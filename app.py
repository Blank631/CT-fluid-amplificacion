import logging
from pathlib import Path
from random import randrange
from zipfile import BadZipFile

import nibabel as nib
import streamlit as st

from convert import to_nifti
from inference import amplify, display_slice, load_amplification_model

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "aa.h5"
PANELS = {
    "Original CT": "original_ct.png",
    "AI-processed CT": "ai_processed_ct.png",
}

cached_model = st.cache_resource(load_amplification_model)


def example_image(filename):
    return next((path for path in (ROOT / filename, ROOT / "assets" / filename) if path.is_file()), None)


st.set_page_config(page_title="CT Fluid Amplification", page_icon="🩻", layout="wide")
st.title("CT Fluid Amplification")
st.markdown("**Research demonstration · Sagittal lumbar spine CT**")
st.write(
    "Compare an original sagittal CT image with its corresponding image "
    "after AI-based fluid amplification."
)

st.subheader("Upload a CT image or volume")
with st.form("ct_upload"):
    upload = st.file_uploader(
        "Anonymized CT file (.jpg, .jpeg, .dcm, .zip, .nii, or .nii.gz)",
        type=["jpg", "jpeg", "dcm", "zip", "nii", "nii.gz"],
        help="For a DICOM series, upload one ZIP. JPEGs and single-slice DICOMs can be converted to NIfTI.",
    )
    submitted = st.form_submit_button("Convert to .nii")
    process = st.form_submit_button("Run AI amplification", disabled=not MODEL_PATH.is_file())

st.caption("If your CT cannot be processed, convert it to .nii in 3D Slicer and try again.")
if not MODEL_PATH.is_file():
    st.info("Add aa.h5 beside app.py to enable AI amplification.")
if submitted or process:
    st.session_state.pop("converted_ct", None)
    st.session_state.pop("amplified_ct", None)
    st.session_state.pop("example_slice", None)
    if upload is None:
        st.error("Select a CT volume first.")
    else:
        try:
            with st.spinner("Converting CT volume..."):
                st.session_state["converted_ct"] = to_nifti(upload.name, upload)
            if process:
                source = nib.Nifti1Image.from_bytes(st.session_state["converted_ct"])
                if upload.name.lower().endswith((".jpg", ".jpeg")) or min(source.shape) < 2:
                    st.info("The file was converted. AI amplification requires a CT volume with multiple slices.")
                else:
                    with st.spinner("Running 3D AI amplification..."):
                        st.session_state["amplified_ct"] = amplify(
                            st.session_state["converted_ct"], cached_model(MODEL_PATH)
                        )
        except (RuntimeError, ValueError, OSError, BadZipFile, TypeError) as error:
            LOGGER.exception("CT upload failed during conversion or amplification")
            if "converted_ct" in st.session_state:
                st.error("The CT was converted, but AI amplification failed. Check the app logs.")
            else:
                st.error(str(error) if type(error) is ValueError else "The CT volume could not be read.")

if "converted_ct" in st.session_state:
    st.download_button(
        "Download .nii", st.session_state["converted_ct"],
        file_name="converted_ct.nii", mime="application/octet-stream",
    )
st.divider()
st.subheader("AI-processed example")
if "amplified_ct" in st.session_state:
    original = nib.as_closest_canonical(nib.Nifti1Image.from_bytes(st.session_state["converted_ct"]))
    amplified = nib.as_closest_canonical(nib.Nifti1Image.from_bytes(st.session_state["amplified_ct"]))
    if "example_slice" not in st.session_state:
        st.session_state["example_slice"] = randrange(original.shape[0])
    index = st.session_state["example_slice"]
    st.caption(f"Random sagittal slice from the uploaded study ({index + 1} of {original.shape[0]}).")
    original_panel, amplified_panel = st.columns(2)
    original_panel.image(display_slice(original.dataobj, index, (-150, 250)), caption="Original CT · soft tissue window")
    amplified_panel.image(display_slice(amplified.dataobj, index, (0, 1)), caption="AI amplification map · normalized 0–1")
    st.download_button(
        "Download AI amplification map (.nii)", st.session_state["amplified_ct"],
        file_name="amplified_ct.nii", mime="application/octet-stream",
    )
else:
    for column, (label, filename) in zip(st.columns(2), PANELS.items()):
        with column:
            path = example_image(filename)
            if path is None:
                st.info(f"Example image unavailable ({filename}).")
            else:
                st.image(str(path), caption=label, width="stretch")
    st.caption("Preprocessed example for research illustration.")

st.link_button("View the research code on GitHub", "https://github.com/Blank631/CT-fluid-amplificacion")
