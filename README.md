# CT Fluid Amplification

Upload an anonymized CT as a JPEG, an individual CT DICOM (.dcm), a ZIP containing one DICOM series, a .nii, or a .nii.gz. JPEGs become grayscale single-slice NIfTI images with default geometry. AI amplification requires a multi-slice CT volume. Convert it to .nii or run AI amplification with the repository's aa.h5 model. AI-processed example compares a random sagittal slice from the uploaded CT with the matching normalized amplification map.

## Deploy on Streamlit Community Cloud

Replace app.py, convert.py, inference.py, and requirements.txt. Keep aa.h5 in that directory. The example PNGs can be in that directory or in assets/. Choose Python 3.12 in Advanced settings and use app.py as the entrypoint.

## Run locally

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
