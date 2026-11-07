# CT Fluid Amplification

Upload an anonymized CT as a ZIP containing one DICOM series, a .nii, or a .nii.gz. Convert it to .nii or run AI amplification with the repository's aa.h5 model. The result viewer compares matching sagittal slices from the original CT and the normalized amplification map.

## Deploy on Streamlit Community Cloud

Replace app.py and requirements.txt, and add inference.py next to them. Keep convert.py and aa.h5 in that directory. The example PNGs can be in that directory or in assets/. Choose Python 3.12 in Advanced settings and use app.py as the entrypoint.

## Run locally

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
