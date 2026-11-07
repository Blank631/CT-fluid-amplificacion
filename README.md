# CT Fluid Amplification

A Streamlit viewer for a paired sagittal lumbar CT example. The AI-processed panel uses the supplied model output. The app converts anonymized CT volumes (one DICOM series in a ZIP, .nii, or .nii.gz) to downloadable .nii. The paired example illustrates AI processing. Inference on uploaded volumes requires integrating aa.h5 and its inference pipeline.

## Run locally

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Extract the ZIP and add `app.py`, `convert.py`, `requirements.txt`, and the `assets/` folder to a GitHub repository. In Streamlit Community Cloud, select the repository and branch, then set `app.py` as the entrypoint. Confirm the example is cleared for public display before publishing.
