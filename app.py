import logging
from pathlib import Path
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

st.subheader("Upload a CT volume")
with st.form("ct_upload"):
    upload = st.file_uploader(
        "Anonymized CT volume (.zip with one DICOM series, .nii, or .nii.gz)",
        type=["zip", "nii", "nii.gz"],
    )
    submitted = st.form_submit_button("Convert to .nii")
    process = st.form_submit_button("Run AI amplification", disabled=not MODEL_PATH.is_file())

st.caption("If your CT cannot be processed, convert it to .nii in 3D Slicer and try again.")
if not MODEL_PATH.is_file():
    st.info("Add aa.h5 beside app.py to enable AI amplification.")
if submitted or process:
    st.session_state.pop("converted_ct", None)
    st.session_state.pop("amplified_ct", None)
    if upload is None:
        st.error("Select a CT volume first.")
    else:
        try:
            with st.spinner("Converting CT volume..."):
                st.session_state["converted_ct"] = to_nifti(upload.name, upload)
            if process:
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
if "amplified_ct" in st.session_state:
    st.subheader("Your CT · Before and after")
    original = nib.as_closest_canonical(nib.Nifti1Image.from_bytes(st.session_state["converted_ct"]))
    amplified = nib.as_closest_canonical(nib.Nifti1Image.from_bytes(st.session_state["amplified_ct"]))
    index = st.slider("Sagittal slice index", 0, original.shape[0] - 1, original.shape[0] // 2)
    original_panel, amplified_panel = st.columns(2)
    original_panel.image(display_slice(original.dataobj, index, (-150, 250)), caption="Original CT · soft tissue window")
    amplified_panel.image(display_slice(amplified.dataobj, index, (0, 1)), caption="AI amplification map · normalized 0–1")
    st.download_button(
        "Download AI amplification map (.nii)", st.session_state["amplified_ct"],
        file_name="amplified_ct.nii", mime="application/octet-stream",
    )
st.divider()
st.subheader("AI-processed example")

mode = st.radio("Viewing mode", ("Side by side", "Switch between images"), horizontal=True)

if mode == "Side by side":
    for column, (label, filename) in zip(st.columns(2), PANELS.items()):
        with column:
            st.subheader(label)
            path = example_image(filename)
            if path is None:
                st.info(f"Example image unavailable ({filename}).")
            else:
                st.image(str(path), width="stretch")
else:
    label = st.radio("Image", tuple(PANELS), horizontal=True)
    st.subheader(label)
    path = example_image(PANELS[label])
    if path is None:
        st.info(f"Example image unavailable ({PANELS[label]}).")
    else:
        st.image(str(path), width=460)

st.caption("The demonstration displays a preprocessed example for research illustration.")
st.link_button("View the research code on GitHub", "https://github.com/Blank631/CT-fluid-amplificacion")
