# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 23:56:47 2025

@author: Azul8
"""
##The sample CT provided in this repository is already in NIfTI format. If you have your own dataset in DICOM format, you will need to convert it to NIfTI before processing.
##If you already have a NIfTI (.nii) file, you can skip the DICOM to NIfTI conversion step.
##For datasets in DICOM format, you need to convert them into NIfTI before processing.  
##This can be done either programmatically (OPTION 1) or using [3D Slicer](https://www.slicer.org/) OPTION 2, which provides a straightforward GUI-based conversion.

##OPTION 1

## DICOM TO NIFTY
import SimpleITK as sitk
import os

# Path to the folder containing the DICOMs (a full series)
dicom_folder = 'C:dicomfile'  # Replace with your DICOM archive

# Read the DICOM series
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
reader.SetFileNames(dicom_names)

# Convert to image
image = reader.Execute()

# Save as NIfTI (.nii)
output_path = 'C:converted_image.nii'  # replace with your localization
sitk.WriteImage(image, output_path)

print(f'NIfTI (.nii) file saved at: {output_path}')

##OPTION 2

##Open 3D Slicer.

##Load the DICOM series using the DICOM module.

##Export the volume as NIfTI (converted_image.nii).
############################################################################################
# NII TO NUMPY
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

# Path to the image 
TC_path = 'C:convertido.nii' ##test image

def load_volume(path):
    """
    Load an image volume from a NIfTI file.
    """
    nii = nib.load(path)
    volume = nii.get_fdata()
    return volume


def resize_volume(volume, new_shape=(128, 128, 128)):
    """
    Resize a volume to new_shape.
    You can adjust this function to preserve aspect ratio or use different interpolation.
    """
    # Compute zoom factors
    zh, zw, zd = np.array(new_shape) / np.array(volume.shape)
    # Apply zoom (simplified, consider adjusting interpolation as needed)
    resized_volume = zoom(volume, (zh, zw, zd), order=1)  # order=1 (bilinear) is generally sufficient
    return resized_volume

# Load volumes
image_volume = load_volume(TC_path)

# Resize volumes
image_resized = resize_volume(image_volume)

# Expand dimensions to meet model input expectations (add a channel axis at the end)
image_resized = np.expand_dims(image_resized, axis=-1)

np.save('C://Users//Azul8//OneDrive//Escritorio//unet imagenes//archivos numpy//TC.npy', image_resized)  ##New archive .npy
##########################################################################
### APPLY MODEL
from tensorflow.keras.models import load_model
my_model = load_model('aa.h5', compile=False)#This model is in the main menu

test_img_input = np.expand_dims(image_resized, axis=0)
test_prediction = my_model.predict(test_img_input)
test_prediction = 1 - test_prediction  # may be annuled depending on what you want to visualize

predicted_mask_binary = (test_prediction > 0.5).astype(bool)

################################
tr_path = 'C://Users//Azul8//OneDrive//Escritorio//unet imagenes//archivos numpy//TC.npy'

# Load .npy files as NumPy arrays
tr = np.load(tr_path)
import matplotlib.pyplot as plt

# Select a random slice
# n_slice = np.random.randint(0, tr.shape[2])
n_slice =30

# Extract slice from original image
img_slice = tr[:, :, n_slice, 0]  # tr shape is (128,128,128,1)

# Extract slice from prediction
pred_slice = test_prediction[0, :, :, n_slice, 0]  # test_prediction shape is (1,128,128,128,1)

# Display images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_slice, cmap='gray')
plt.title('CT Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(pred_slice, cmap='binary')
plt.title('Amplified Fluid')  # FF = Free Fluid
plt.axis('off')

plt.tight_layout()
plt.show()
################################################################
# CONVERT .NPY BACK TO NIfTI WITH ORIGINAL SHAPE AND RESOLUTION

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tensorflow.keras.models import load_model

# --- 1) Paths ---
ref_nii_path = r'C:\Users\Azul8\Downloads\CT-fluid-amplificacion-main\CT-fluid-amplificacion-main\CT_test\convertido.nii'
npy_path    = r'C:\Users\Azul8\OneDrive\Escritorio\unet imagenes\archivos numpy\TC.npy'
model_path  = r'aa.h5'
out_path    = r'C:\Users\Azul8\OneDrive\Escritorio\unet imagenes\archivos nifty\amplified_fluid_prob.nii'

# --- 2) Load reference NIfTI (for affine/spacing) ---
ref = nib.load(ref_nii_path)
ref_shape  = np.array(ref.shape)
ref_affine = ref.affine
ref_header = ref.header

# --- 3) Load preprocessed volume (128x128x128x1) and run prediction ---
vol128 = np.load(npy_path)                     # (128,128,128,1)
if vol128.ndim == 4 and vol128.shape[-1] == 1:
    vol128 = np.squeeze(vol128, axis=-1)       # (128,128,128)

model = load_model(model_path, compile=False)
pred = model.predict(np.expand_dims(vol128, axis=(0, -1)))  # -> (1,128,128,128,1)
pred = np.squeeze(pred, axis=(0, -1))          # (128,128,128)

# Optional: invert prediction as in your script
pred = 1.0 - pred

# --- 4) Resample to the original NIfTI size (for 1:1 overlay in Slicer) ---
factors = ref_shape / np.array(pred.shape, dtype=float)
pred_resampled = zoom(pred, factors, order=1)  # lineal para mapa continuo

# --- 5) Normalize (0..1) so “high intensity = fluid” ---
pmin, pmax = float(pred_resampled.min()), float(pred_resampled.max())
if pmax > pmin:
    pred_norm = (pred_resampled - pmin) / (pmax - pmin)
else:
    pred_norm = np.zeros_like(pred_resampled, dtype=np.float32)
pred_norm = pred_norm.astype(np.float32)

# --- 6) Save as NIfTI with study geometry ---
nii = nib.Nifti1Image(pred_norm, affine=ref_affine, header=ref_header)
nii.set_sform(ref_affine, code=1)
nii.set_qform(ref_affine, code=1)
nib.save(nii, out_path)

print(f"✅ Guardado: {out_path}")
