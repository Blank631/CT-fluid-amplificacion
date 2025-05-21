

This repository provides a complete pipeline for converting DICOM image series to NIfTI format, processing them as NumPy arrays, applying a deep learning model for segmentation (e.g., free fluid in abdominal CT), and restoring the predictions back to the original shape and resolution.
Includes a pre-trained model (aa.h5) along with the code required for processing abdominal CT scans.
## 🔧 Workflow Overview

1. **DICOM to NIfTI Conversion** using SimpleITK
2. **NIfTI to NumPy (.npy)** array transformation
3. **Resizing** to a standard 3D shape (128×128×128)
4. **Applying a trained U-Net model** for segmentation
5. **Visualization** of image slices and predicted masks
6. **Restoring predictions** to the original resolution and saving as NIfTI

You can find CT scans for processing at https://www.kaggle.com/code/ayushs9020/understanding-the-competition-rsna 
