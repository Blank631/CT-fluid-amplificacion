from pathlib import Path

import streamlit as st
from zipfile import BadZipFile

from convert import to_nifti

ASSETS = Path(__file__).resolve().parent / "assets"
PANELS = {
    "Original CT": "original_ct.png",
    "AI-processed CT": "ai_processed_ct.png",
}

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

st.caption("If your CT cannot be processed, convert it to .nii in 3D Slicer and try again.")
if submitted:
    st.session_state.pop("converted_ct", None)
    if upload is None:
        st.error("Select a CT volume first.")
    else:
        try:
            with st.spinner("Converting CT volume..."):
                st.session_state["converted_ct"] = to_nifti(upload.name, upload)
        except (RuntimeError, ValueError, OSError, BadZipFile) as error:
            message = str(error) if type(error) is ValueError else "The CT volume could not be read."
            st.error(message)

if "converted_ct" in st.session_state:
    st.download_button(
        "Download .nii", st.session_state["converted_ct"],
        file_name="converted_ct.nii", mime="application/octet-stream",
    )
st.caption("AI processing of uploaded volumes requires the model integration.")
st.divider()
st.subheader("AI-processed example")

mode = st.radio("Viewing mode", ("Side by side", "Switch between images"), horizontal=True)

if mode == "Side by side":
    for column, (label, filename) in zip(st.columns(2), PANELS.items()):
        with column:
            st.subheader(label)
            st.image(str(ASSETS / filename), use_container_width=True)
else:
    label = st.radio("Image", tuple(PANELS), horizontal=True)
    st.subheader(label)
    st.image(str(ASSETS / PANELS[label]), width=460)

st.caption("The demonstration displays a preprocessed example for research illustration.")
st.link_button("View the research code on GitHub", "https://github.com/Blank631/CT-fluid-amplificacion")
