import os
import pydicom
import numpy as np
import SimpleITK as sitk
from datetime import datetime
import math

# Note: SimpleITK typically applies the RescaleSlope automatically when reading images. 
# Therefore, the image is already in Bq/mL, and we only need to calculate the SUV factor.
def calculate_suv_factor(dicom_file_path):
    ds = pydicom.dcmread(dicom_file_path, stop_before_pixels=True)
    
    try:
        weight_kg = float(ds.PatientWeight)
        weight_g = weight_kg * 1000.0
        
        radio_seq = ds.RadiopharmaceuticalInformationSequence[0]
        total_dose = float(radio_seq.RadionuclideTotalDose) 
        half_life = float(radio_seq.RadionuclideHalfLife)   
        
        start_time_str = radio_seq.RadiopharmaceuticalStartTime
        scan_time_str = ds.AcquisitionTime 
        
        # for convinience, we only care about hour, min, and sec.
        t_inject = datetime.strptime(start_time_str.split('.')[0], "%H%M%S")
        t_scan = datetime.strptime(scan_time_str.split('.')[0], "%H%M%S")
        
        delta_time_seconds = (t_scan - t_inject).total_seconds()
        
        # Dose_current = Dose_initial * exp(-ln(2) * t / half_life)
        decay_corrected_dose = total_dose * math.exp(-math.log(2) * delta_time_seconds / half_life)
        
        suv_factor = weight_g / decay_corrected_dose
        print(f"SUV: {suv_factor:.6f}")
        return suv_factor
        
    except Exception as e:
        print(f"something went wrong: {e}")
        return 1.0
    
# PET image --> CT phy space
def resample_pet_to_ct(pet_image, ct_image):

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image)      # Based on CT
    resampler.SetInterpolator(sitk.sitkLinear) 
    resampler.SetDefaultPixelValue(0)          
    
    resampled_pet = resampler.Execute(pet_image)
    print(f"Sampling is done: {resampled_pet.GetSize()}")
    return resampled_pet

def extract_body_mask(ct_image):
    mask = sitk.BinaryThreshold(ct_image, lowerThreshold=-500, upperThreshold=3000, insideValue=1, outsideValue=0)
    
    morph_filter = sitk.BinaryMorphologicalClosingImageFilter()
    morph_filter.SetKernelRadius([5, 5, 5])
    morph_filter.SetForegroundValue(1)
    mask_closed = morph_filter.Execute(mask)
    
    print("Body contour mask extraction complete.")
    return mask_closed

def preprocess_pet(ct_dicom_dir, pet_dicom_dir, output_nii_path):
    print("Reading DICOM series...")
    reader = sitk.ImageSeriesReader()
    
    ct_dicom_names = reader.GetGDCMSeriesFileNames(ct_dicom_dir)
    reader.SetFileNames(ct_dicom_names)
    ct_image = reader.Execute()
    
    pet_dicom_names = reader.GetGDCMSeriesFileNames(pet_dicom_dir)
    reader.SetFileNames(pet_dicom_names)
    pet_image = reader.Execute()
    
    aligned_pet = resample_pet_to_ct(pet_image, ct_image)
    
    pet_array = sitk.GetArrayFromImage(aligned_pet)
    
    #print("\nread Metadata and cal SUV...")
    sample_pet_dcm = pet_dicom_names[0] 
    suv_factor = calculate_suv_factor(sample_pet_dcm)
    
    pet_suv_array = pet_array * suv_factor
    pet_suv_array[pet_suv_array < 0] = 0 # Eliminate negative values caused by reconstruction artifacts
    

    print("\nbody contor mask...")
    body_mask_img = extract_body_mask(ct_image)
    body_mask_array = sitk.GetArrayFromImage(body_mask_img)
    
    pet_suv_array = pet_suv_array * body_mask_array
    
    print("\nNorm...")
    pet_log_array = np.log1p(pet_suv_array) # log1p = log(1 + x)
    
    min_val = np.min(pet_log_array)
    max_val = np.max(pet_log_array)
    pet_normalized = (pet_log_array - min_val) / (max_val - min_val + 1e-8)
    
    #print(f"min: {np.min(pet_normalized):.4f}, max: {np.max(pet_normalized):.4f}, mean: {np.mean(pet_normalized):.4f}")
    
    print("\nSaving...")
    final_pet_img = sitk.GetImageFromArray(pet_normalized.astype(np.float32))
    
    final_pet_img.CopyInformation(ct_image)
    
    sitk.WriteImage(final_pet_img, output_nii_path)

# only align size based on CT space. no transformation. Just for test.
def pet_aligh_ct_test(pet_dir, ct_dir, output_folder):
   
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def read_series(directory):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(directory)
        reader.SetFileNames(dicom_names)
        return reader.Execute(), dicom_names

    print("Loading CT and PET 3D Vols...")
    ct_volume, ct_file_list = read_series(ct_dir)
    pet_volume, _ = read_series(pet_dir)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_volume)  
    resampler.SetInterpolator(sitk.sitkLinear)  
    resampler.SetDefaultPixelValue(0)

    aligned_pet_volume = resampler.Execute(pet_volume)

    aligned_array = sitk.GetArrayFromImage(aligned_pet_volume)
    
    for i, ct_file_path in enumerate(ct_file_list):
    
        ds = pydicom.dcmread(ct_file_path)
        
        pet_slice = aligned_array[i].astype(np.uint16)
        
        ds.PixelData = pet_slice.tobytes()
        ds.Rows, ds.Columns = pet_slice.shape
        ds.Modality = "PT"
        ds.SeriesDescription = "Aligned PET to CT"
        
        original_name = os.path.basename(ct_file_path)
        save_path = os.path.join(output_folder, f"aligned_PET_{original_name}")
        ds.save_as(save_path)

if __name__=='__main__':
    pet_aligh_ct_test("temp/3.000000-PET-01743", "temp/4.000000-CT-00452", "temp_rst")