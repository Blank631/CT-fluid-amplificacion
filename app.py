import logging
from pathlib import Path
from random import sample
from zipfile import BadZipFile

import nibabel as nib
import numpy as np
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

st.caption(
    "If your CT cannot be processed, convert it to .nii in 3D Slicer and try again. "
    "If the study is too large, use 3D Slicer to crop the volume or keep fewer slices, "
    "then save the smaller volume as .nii and upload it again."
)
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
    count = original.shape[0]
    show_another = st.button("Show another slice")
    if "example_slice" not in st.session_state or show_another:
        first, last = count // 4, count - count // 4
        central = range(first, last)
        candidates = sample(central, k=min(24, len(central)))
        previous = st.session_state.get("example_slice")
        visible = []
        for index in candidates:
            ct = display_slice(original.dataobj, index, (-150, 250))
            result = display_slice(amplified.dataobj, index, (0, 1))
            if np.ptp(ct) >= 12 and np.ptp(result) >= 12:
                visible.append(index)
        st.session_state["example_slice"] = next(
            (index for index in visible if index != previous),
            visible[0] if visible else count // 2,
        )
    index = st.session_state["example_slice"]
    st.caption(f"Sagittal slice from the uploaded study ({index + 1} of {count}).")
    ct = display_slice(original.dataobj, index, (-150, 250))
    result = display_slice(amplified.dataobj, index, (0, 1))
    original_panel, amplified_panel = st.columns(2)
    original_panel.image(ct, caption="Original CT · soft tissue window", width="stretch")
    amplified_panel.image(result, caption="AI amplification map · normalized 0–1", width="stretch")
    if np.ptp(result) < 12:
        st.caption("The amplification map has little or no visible contrast in this slice.")
    st.download_button(
        "Download AI amplification map (.nii)", st.session_state["amplified_ct"],
        file_name="amplified_ct.nii", mime="application/octet-stream",
    )
elif "converted_ct" in st.session_state:
    converted = nib.Nifti1Image.from_bytes(st.session_state["converted_ct"])
    if converted.shape[2] == 1:
        preview = np.asarray(converted.dataobj[:, :, 0], dtype=np.float32).T
        low, high = float(preview.min()), float(preview.max())
        if high > low:
            preview = (preview - low) / (high - low)
        else:
            preview = np.full_like(preview, 0.5)
        st.image(preview, caption="Uploaded image · converted to NIfTI", width=460)
        st.caption("AI amplification requires a CT volume with multiple slices.")
    else:
        original = nib.as_closest_canonical(converted)
        index = original.shape[0] // 2
        st.image(display_slice(original.dataobj, index, (-150, 250)), caption="Original CT · sagittal preview", width=460)
        st.caption("Use Run AI amplification to generate the matching amplification map.")
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
