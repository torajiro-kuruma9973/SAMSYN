import SimpleITK as sitk
from pathlib import Path
import torch 
from collections import OrderedDict
from PIL import Image
import numpy as np

def get_normed_tensor_from_dcm(dicom_dir): #normed result into [0, 1]
    sitk_image = sitk.ReadImage(dicom_dir)
    numpy_array = sitk.GetArrayFromImage(sitk_image)
    
    assert len(numpy_array.shape) ==  3 # 3-dim is mandatory

    tnsr = torch.tensor(numpy_array) # shape: [1, h, w]
    min_val = torch.min(tnsr.to(torch.float32))
    max_val = torch.max(tnsr.to(torch.float32))
    
    if max_val - min_val > 0:
        normalized_slice = (tnsr - min_val) / (max_val - min_val)
    else:
        normalized_slice = tnsr - min_val

    return normalized_slice

def dcm2jpg(dicom_dir, output_dir):
    name = Path(dicom_dir).stem
    name = output_dir + name + ".jpg"
    slice = get_normed_tensor_from_dcm(dicom_dir)
    slice = slice * 255
    tensor_uint8 = slice.to(torch.uint8).cpu()
    numpy_array = tensor_uint8.squeeze().numpy() #[1, h, w]  -->  [h, w]

    img = Image.fromarray(numpy_array)
    img.save(name, format="JPEG")
    

def convert_dcm_file(dicom_dir, output_dir):
    folder = Path(dicom_dir)
    file_paths = sorted(folder.glob('*.dcm'))

    if not file_paths:
        print(f"ERROR: cannot find any dcm file in {dicom_dir} ...")
        return None
    
    for file_path in file_paths:
        dcm2jpg(file_path, output_dir)
        

def dicom_to_nifti_sitk(dicom_dir, output_filepath):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    
    if not dicom_names:
        print("ERROR: no dcm file is located...")
        return
        
    reader.SetFileNames(dicom_names)
    
    sitk_image = reader.Execute()
    
    sitk.WriteImage(sitk_image, output_filepath)
    print(f"Done! The file is saved in: {output_filepath}")

if __name__=='__main__':
    convert_dcm_file("raw_datasets/BrainTumorMRI/", "jpg_datasets/")