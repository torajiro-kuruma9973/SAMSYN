import os
import pydicom
import numpy as np
import SimpleITK as sitk
import pet_preprocess as pp
import ct_preprocess as cp

pp.preprocess_pet(ct_dicom_dir, pet_dicom_dir, output_nii_path)

cp.preprocess_ct(ct_dicom_dir, output_nii_path, target_spacing=(2.0, 2.0, 2.0), clip_min=-1000, clip_max=1000)

# 4. (可选) 处理 CT 自身 (截断 HU、归一化等，如果是跨患者统一尺寸的话)
target_ct = process_CT(ct_img)

# 5. 分别保存，大功告成！
sitk.WriteImage(target_ct, "patient_01_0000.nii.gz")
sitk.WriteImage(target_pet, "patient_01_0001.nii.gz")
sitk.WriteImage(target_seg, "patient_01_seg.nii.gz")