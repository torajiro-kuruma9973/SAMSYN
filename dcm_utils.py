import SimpleITK as sitk
from pathlib import Path
import torch 
from collections import OrderedDict
from PIL import Image
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import pydicom

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

# special_str: the prefix or other str in file name to be removed
def dcm2jpg(dicom_dir, output_dir):
    name = Path(dicom_dir).stem
    name = output_dir + name + ".jpg"
    
    slice = get_normed_tensor_from_dcm(dicom_dir)
    slice = slice * 255
    tensor_uint8 = slice.to(torch.uint8).cpu()
    numpy_array = tensor_uint8.squeeze().numpy() #[1, h, w]  -->  [h, w]

    img = Image.fromarray(numpy_array)
    img.save(name, format="JPEG")
    

def convert_all_dcm_files(dicom_dir, output_dir):
    folder = Path(dicom_dir)
    file_paths = sorted(folder.glob('*.dcm'))

    if not file_paths:
        print(f"ERROR: cannot find any dcm file in {dicom_dir} ...")
        return None
    
    for file_path in file_paths:
        dcm2jpg(file_path, output_dir)


def file_name_process(dicom_dir):
    pattern = re.compile(r'^IM-(\d+)-(\d+)\.dcm$')
    valid_files = []
    for filename in os.listdir(dicom_dir):
        match = pattern.match(filename)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            valid_files.append((x, y, filename))
    valid_files.sort(key=lambda item: (item[0], item[1]))
    for index, item in enumerate(valid_files, start=0):
        original_name = item[2]
        new_name = f"{index}.dcm"
        
        old_filepath = os.path.join(dicom_dir, original_name)
        new_filepath = os.path.join(dicom_dir, new_name)
        
        try:
            os.rename(old_filepath, new_filepath)
            print(f"done: {original_name} -> {new_name}")
        except FileExistsError:
            print(f"ERROR: {new_name} existed，skip {original_name}")

def order_file_names(folder):
    frame_names = [
    p for p in os.listdir(folder)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda name: int(name[:-4]))
    ordered_file_names = []
    for name in frame_names:
        ordered_file_names.append(folder + name)
    return ordered_file_names
        

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


def is_folder_empty_pathlib(folder_path):
    folder = Path(folder_path)
    
    return not any(folder.iterdir())

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


# list all the ref mapping between mask slice to PET slice
# e.g. idx 10 in masks is the segmentation of idx 50 of PET slice.
def get_referenced_imaging_info(seg_dcm_path):
 
    if not os.path.exists(seg_dcm_path):
        print(f"Error: cannot find {seg_dcm_path}")
        return

    ds = pydicom.dcmread(seg_dcm_path)

    print(f"{'='*60}")
    print(f"Analyzing masks: {os.path.basename(seg_dcm_path)}")
    print(f"{'='*60}")

    rows = getattr(ds, 'Rows', '未知')
    cols = getattr(ds, 'Columns', '未知')
    num_frames = getattr(ds, 'NumberOfFrames', 1)
    
    print(f"[-] matix shape: {rows} x {cols}")
    print(f"[-] Mask Slices: {num_frames}")

    try:
        ref_series_seq = ds.ReferencedSeriesSequence[0]
        ref_series_uid = ref_series_seq.SeriesInstanceUID
        
        print(f"    >>> Original PET Series UID: {ref_series_uid}")
        
        if hasattr(ref_series_seq, 'ReferencedInstanceSequence'):
            num_ref_instances = len(ref_series_seq.ReferencedInstanceSequence)
            print(f"    >>> num of slice reffered: {num_ref_instances}")
            
    except (AttributeError, IndexError):
        # in some non-standard cases
        if hasattr(ds, 'ReferencedImageSequence'):
            ref_image = ds.ReferencedImageSequence[0]
            print(f"\n[?] cannot find Series UID，but find image instance refferd ID:")
            print(f"    >>> Referenced SOP Instance UID: {ref_image.ReferencedSOPInstanceUID}")
        else:
            print("\n[x] Warning: This DICOM does not contain any ReferencedSeriesSequence labels")
    
    print(f"{'='*60}\n")

# segmentation file sometimes is shaped different then PET file by whiich is refered,
# due to sparse annotation.
# e.g. the seg shape is [33. 200, 200], while the PET file is [300, 200, 300].
# So we should add whole black frame in seg file to make which has the same shape as the PET file.
def convert_seg_to_nifti_sitk(seg_path, dicom_series_dir, output_path):
    """
    Use SimpleITK to align DICOM-SEG and PET
    :param seg_path: Segmentation folder
    :param dicom_series_dir: PET folder
    :param output_path: output path
    """
    
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_series_dir)
    reader.SetFileNames(dicom_names)
    reference_img = reader.Execute()
    
    ref_uids = [pydicom.dcmread(f).SOPInstanceUID for f in dicom_names]
    
    mask_array = np.zeros(reference_img.GetSize()[::-1], dtype=np.uint8) 

    seg_ds = pydicom.dcmread(seg_path)
    seg_array = seg_ds.pixel_array
    
    per_frame_seq = seg_ds.PerFrameFunctionalGroupsSequence
    
    for i, frame in enumerate(per_frame_seq):
        referenced_uid = frame.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
        print(f"@@ {i} : {referenced_uid}")
        if referenced_uid in ref_uids:
            z_index = ref_uids.index(referenced_uid)
            print(f"idx: {z_index}")
            mask_array[z_index, :, :] = seg_array[i]
            
    final_mask = sitk.GetImageFromArray(mask_array)
    final_mask.CopyInformation(reference_img)
    
    sitk.WriteImage(final_mask, output_path)
    print(f"Done! Saved to: {output_path}")
    print(f"Mask shape: {final_mask.GetSize()}, num of non-all-zero intervals: {np.sum(np.any(mask_array, axis=(1,2)))}")

def find_slice_index(dicom_dir, target_uid):
    files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    
    slice_info = []
    for f in files:
        ds = pydicom.dcmread(f)
        # ImagePositionPatient[2]
        slice_info.append((ds.ImagePositionPatient[2], ds.SOPInstanceUID, os.path.basename(f)))
    
    # order by z
    slice_info.sort(key=lambda x: x[0])
    
    for index, (pos, uid, name) in enumerate(slice_info):
        if uid == target_uid:
            return index, name, pos
    return None

def read_segment_sequence(seg_dcm_path):
    ds = pydicom.pydicom.dcmread(seg_dcm_path)
    
    # check if Segment Sequence (0062, 0002) exsited.
    if hasattr(ds, 'SegmentSequence'):
        segments = ds.SegmentSequence
        print(f"this file contains {len(segments)} defined Segment(s):\n")
        
        for i, segment in enumerate(segments):
           
            seg_num = getattr(segment, 'SegmentNumber', i + 1)
            
            # Segment Label
            label = getattr(segment, 'SegmentLabel', 'Unamed')
            
            description = getattr(segment, 'SegmentDescription', 'No description')
            
            color = getattr(segment, 'RecommendedDisplayCIELabValue', 'default color')

            print(f"--- Segment {seg_num} ---")
            print(f"    Label: {label}")
            print(f"    Descrption: {description}")
            
            if hasattr(segment, 'SegmentedPropertyCategoryCodeSequence'):
                category = segment.SegmentedPropertyCategoryCodeSequence[0].CodeMeaning
                print(f"    Category: {category}")
                
            print("-" * 20)
    else:
        print("No Segment Sequence is found...")

# how many objs in this 3D masks.
def count_lesions(nifti_mask_path):
    mask_img = sitk.ReadImage(nifti_mask_path)
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    label_img = cc_filter.Execute(mask_img)
    
    num_lesions = cc_filter.GetObjectCount()
    
    print(f"There is/are {num_lesions} tumors in total...")
    return num_lesions

if __name__=='__main__':
    #file_name_process("raw_datasets/BrainTumorMRI/")
    #dicom_to_nifti_sitk("PET_raw_files/3.000000-PET-01743", "nii_files/PET_nii_files/PSMA_0a3fdc59c5e700d8PET.nii.gz")
    #get_referenced_imaging_info("./1-1.dcm")
    seg_file = "./1-1.dcm"
    ref_dir = "PET_raw_files/3.000000-PET-01743" 
    output_nii = "./case_0001seg.nii.gz"
    #convert_seg_to_nifti_sitk(seg_file, ref_dir, output_nii)
    #index, filename, z_pos = find_slice_index(ref_dir, "1.3.6.1.4.1.14519.5.2.1.162115129127399650767843649836237257916")
    #print(f"该 UID 对应物理层序第 {index} 层，文件名是 {filename}")
    #read_segment_sequence("./1-1.dcm")
    count_lesions("case_0001seg.nii.gz")