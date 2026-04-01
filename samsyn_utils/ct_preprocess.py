import os
import numpy as np
import SimpleITK as sitk

# CT re-sampling, HU threshold, normed.
def preprocess_ct(ct_dicom_dir, output_nii_path, target_spacing=(2.0, 2.0, 2.0), clip_min=-1000, clip_max=1000):
 
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(ct_dicom_dir)
    reader.SetFileNames(dicom_names)
    ct_image = reader.Execute()
    
    original_spacing = ct_image.GetSpacing()
    original_size = ct_image.GetSize()
    
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / target_spacing[2])))
    ]
    
    print(f"  Original spacing: {original_spacing}, original size: {original_size}")
    print(f"  target spacing: {target_spacing}, new size: {new_size}")
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(ct_image.GetDirection())
    resampler.SetOutputOrigin(ct_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1000) # zero if out of VOF
    resampler.SetInterpolator(sitk.sitkBSpline) 
    
    resampled_ct = resampler.Execute(ct_image)
    
    print(f"  HU clip: [{clip_min}, {clip_max}]")
    ct_array = sitk.GetArrayFromImage(resampled_ct)
    
    ct_clipped = np.clip(ct_array, clip_min, clip_max)
    
    print("  Normed to [0, 1]...")
    ct_normalized = (ct_clipped - clip_min) / (clip_max - clip_min)
    
    final_ct_img = sitk.GetImageFromArray(ct_normalized.astype(np.float32))
    final_ct_img.CopyInformation(resampled_ct)
    
    sitk.WriteImage(final_ct_img, output_nii_path)

# ct_dir = "path/to/CT_folder"
# output_path = "patient_001_input_CT.nii.gz"
# preprocess_ct_for_dl(ct_dir, output_path, target_spacing=(2.0, 2.0, 2.0))