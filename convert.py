from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from PIL import Image
import SimpleITK as sitk

MAX_DICOM_BYTES = 512 * 1024 * 1024


def to_nifti(name, data):
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        name = name.lower()
        if name.endswith((".nii", ".nii.gz")):
            source = folder / ("input.nii.gz" if name.endswith(".gz") else "input.nii")
            source.write_bytes(data.getbuffer())
            volume = sitk.ReadImage(str(source), imageIO="NiftiImageIO")
        elif name.endswith((".jpg", ".jpeg")):
            with Image.open(data, formats=["JPEG"]) as image:
                if image.width * image.height > 16_000_000:
                    raise ValueError("The JPEG image is too large.")
                volume = sitk.GetImageFromArray(np.asarray(image.convert("L"))[None, ...])
        elif name.endswith(".dcm"):
            source = folder / "input.dcm"
            source.write_bytes(data.getbuffer())
            reader = sitk.ImageFileReader()
            reader.SetImageIO("GDCMImageIO")
            reader.SetFileName(str(source))
            reader.ReadImageInformation()
            if not reader.HasMetaDataKey("0008|0060") or reader.GetMetaData("0008|0060").strip() != "CT":
                raise ValueError("The DICOM file must be from a CT scan.")
            volume = reader.Execute()
            if volume.GetDimension() == 2:
                volume = sitk.JoinSeries(volume)
        elif name.endswith(".zip"):
            with ZipFile(data) as archive:
                files = [item for item in archive.infolist() if not item.is_dir()]
                if len(files) > 4096 or sum(item.file_size for item in files) > MAX_DICOM_BYTES:
                    raise ValueError("The DICOM archive is too large.")
                for i, item in enumerate(files):
                    with archive.open(item) as src, (folder / f"slice_{i}").open("wb") as dst:
                        shutil.copyfileobj(src, dst)
            reader = sitk.ImageSeriesReader()
            candidates = []
            for uid in reader.GetGDCMSeriesIDs(str(folder)) or ():
                files = reader.GetGDCMSeriesFileNames(str(folder), uid)
                info = sitk.ImageFileReader()
                info.SetFileName(files[0])
                info.ReadImageInformation()
                if info.HasMetaDataKey("0008|0060") and info.GetMetaData("0008|0060").strip() == "CT":
                    candidates.append(files)
            if len(candidates) != 1:
                raise ValueError("The ZIP must contain exactly one CT series.")
            reader.SetFileNames(candidates[0])
            volume = reader.Execute()
        else:
            raise ValueError("Use a JPEG, DICOM, DICOM ZIP, or NIfTI file.")
        if volume.GetDimension() != 3 or min(volume.GetSize()[:2]) < 2 or volume.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError("Upload a scalar CT image or volume.")
        volume = sitk.Image(volume)
        for key in volume.GetMetaDataKeys():
            volume.EraseMetaData(key)
        output = folder / "converted_ct.nii"
        sitk.WriteImage(volume, str(output))
        return output.read_bytes()
