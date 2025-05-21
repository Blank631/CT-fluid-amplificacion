# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 23:56:47 2025

@author: Azul8
"""
##If you already have a NIfTI (.nii) file, you can skip the DICOM to NIfTI conversion step.

## DICOM TO NIFTY
import SimpleITK as sitk
import os

# Path to the folder containing the DICOMs (a full series)
dicom_folder = 'C://Users//Azul8//OneDrive//Escritorio//unet imagenes//Rosalba//10000184'  # Replace with your DICOM archive

# Read the DICOM series
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
reader.SetFileNames(dicom_names)

# Convert to image
image = reader.Execute()

# Save as NIfTI (.nii)
output_path = 'C://Users//Azul8//OneDrive//Escritorio//unet imagenes//archivos nifty//imagen_convertida.nii'  # replace with your localization
sitk.WriteImage(image, output_path)

print(f'NIfTI (.nii) file saved at: {output_path}')
############################################################################################
# NII TO NUMPY
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

# Path to the image 
TC_path = 'C://Users//Azul8//OneDrive//Escritorio//unet imagenes//archivos nifty//imagen_convertida.nii'

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

np.save('C://Users//Azul8//OneDrive//Escritorio//unet imagenes//archivos numpy//imagen_3001.npy', image_resized)
##########################################################################
### APPLY MODEL
from tensorflow.keras.models import load_model
my_model = load_model('aa.h5', compile=False)#This model is in the main menu

test_img_input = np.expand_dims(image_resized, axis=0)
test_prediction = my_model.predict(test_img_input)
test_prediction = 1 - test_prediction  # may be annuled depending on what you want to visualize

predicted_mask_binary = (test_prediction > 0.5).astype(bool)

################################
tr_path = 'C://Users//Azul8//OneDrive//Escritorio//unet imagenes//archivos numpy//imagen_3001.npy'

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
# CONVERT BACK TO ORIGINAL SHAPE NUMPY AND RESOLUTION

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

# Path to the original NIfTI file
original_nii_path = 'C://Users//Azul8//OneDrive//Escritorio//unet imagenes//archivos nifty//imagen_convertida.nii'
original_nii = nib.load(original_nii_path)
original_shape = original_nii.shape
original_affine = original_nii.affine

# Path to the .npy file (prediction or resized image)
npy_path = 'C://Users//Azul8//OneDrive//Escritorio//unet imagenes//archivos numpy//imagen_3001.npy'
npy_array = np.load(npy_path)

# Remove channel if present (from shape (128,128,128,1) to (128,128,128))
if npy_array.shape[-1] == 1:
    npy_array = np.squeeze(npy_array, axis=-1)

# Resize back to the original shape
def resize_to_original(volume, target_shape):
    factors = np.array(target_shape) / np.array(volume.shape)
    return zoom(volume, factors, order=1)  # Bilinear interpolation

resized_to_original = resize_to_original(npy_array, original_shape)

# Create new NIfTI object with the resized matrix and original affine
new_nii = nib.Nifti1Image(resized_to_original, affine=original_affine)

# Save the new NIfTI file
output_path = 'C://Users//Azul8//OneDrive//Escritorio//unet imagenes//archivos nifty//prediccion_restaurada.nii'
nib.save(new_nii, output_path)

print(f"Restored NIfTI file saved at: {output_path}")
