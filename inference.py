import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def load_amplification_model(model_path):
    from tensorflow.keras.layers import Conv3DTranspose, Rescaling
    from tensorflow.keras.models import load_model

    class DivideBy255(Rescaling):
        @classmethod
        def from_config(cls, config):
            return cls(scale=1 / 255, name=config["name"], dtype=config["dtype"])

    class LegacyConv3DTranspose(Conv3DTranspose):
        @classmethod
        def from_config(cls, config):
            if config.pop("groups", 1) != 1:
                raise ValueError("Unexpected grouped transposed convolution in aa.h5.")
            return super().from_config(config)

    return load_model(
        model_path, compile=False,
        custom_objects={"Lambda": DivideBy255, "Conv3DTranspose": LegacyConv3DTranspose},
    )


def amplify(source_bytes, model):
    source = nib.Nifti1Image.from_bytes(source_bytes)
    volume = source.get_fdata(dtype=np.float32, caching="unchanged")
    if not np.isfinite(volume).all():
        raise ValueError("The CT contains non-finite voxel values.")
    resized = zoom(volume, np.array([128, 128, 128]) / volume.shape, order=1)
    prediction = np.asarray(model(resized[None, ..., None], training=False))
    if prediction.shape != (1, 128, 128, 128, 1) or not np.isfinite(prediction).all():
        raise ValueError("The model returned an invalid output volume.")
    output = zoom(1.0 - prediction[0, ..., 0], np.array(source.shape) / (128, 128, 128), order=1)
    low, high = float(output.min()), float(output.max())
    output -= low
    if high > low:
        output /= high - low
    result = nib.Nifti1Image(output, source.affine)
    result.header.set_xyzt_units(*source.header.get_xyzt_units())
    return result.to_bytes()


def display_slice(data, index, bounds):
    low, high = bounds
    scaled = np.clip((data[index, :, :].T[::-1, :] - low) / (high - low), 0, 1)
    return (scaled * 255).astype(np.uint8)
