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

def dump_dicom_metadata(dcm_path):
    
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    
    print("-" * 80)
    print(f"{'Tag':<15} | {'VR':<4} | {'Name':<35} | {'Value'}")
    print("-" * 80)
    
    for elem in ds:
        tag_str = f"({elem.tag.group:04x},{elem.tag.element:04x})".upper()
        
        name = elem.name[:33] + ".." if len(elem.name) > 35 else elem.name
        vr = elem.VR
        
        value_str = str(elem.value)
        if len(value_str) > 50:
            value_str = value_str[:47] + "..."
            
        print(f"{tag_str:<15} | {vr:<4} | {name:<35} | {value_str}")
        
    print("-" * 80)

# some dcm file contain extra info which is not mandatory.
# e.g. in the PSMA datasets, PET dcm contains segmentation info.
def print_segment_sequence_details(dcm_path):
    
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    
    if 'SegmentSequence' not in ds:
        print("Warning: This DICOM does NOT contain Segment Sequence (0062,0002)。")
        return
        
    seq = ds.SegmentSequence
    print(f"\nSegment Sequence contains {len(seq)} Segments.\n")
    print("=" * 60)
    
    for i, item in enumerate(seq):
        print(f" Segment #{i+1}")
        print("-" * 60)
        
        seg_num = item.get('SegmentNumber', 'Unknown')
        seg_label = item.get('SegmentLabel', 'No label')
        seg_desc = item.get('SegmentDescription', 'No description')
        
        print(f"  Segment Number  : {seg_num}")
        print(f"  Segment Label   : {seg_label}")
        print(f"  Description     : {seg_desc}")
        
        algo_type = item.get('SegmentAlgorithmType', 'Unknown')
        algo_name = item.get('SegmentAlgorithmName', 'Unrecorded')
        print(f"  Algorithm Type  : {algo_type}")
        if algo_type != 'MANUAL':
            print(f"  Algorithm Name  : {algo_name}")
            
        if 'SegmentedPropertyCategoryCodeSequence' in item:
            cat_seq = item.SegmentedPropertyCategoryCodeSequence[0]
            cat_meaning = cat_seq.get('CodeMeaning', 'Unknown')
            print(f"  Category        : {cat_meaning}")
            
        if 'SegmentedPropertyTypeCodeSequence' in item:
            type_seq = item.SegmentedPropertyTypeCodeSequence[0]
            type_meaning = type_seq.get('CodeMeaning', 'Unknown')
            print(f"  Property Type   : {type_meaning}")
            
        print("=" * 60 + "\n")


# output all the points whose value >= value_threshold
def map_pixels_to_physical_coords(dcm_path, value_threshold=10000):
    ds = pydicom.dcmread(dcm_path)
    pixel_matrix = ds.pixel_array
    
    # Image Position (Patient): [X0, Y0, Z0]
    origin_x, origin_y, origin_z = [float(val) for val in ds.ImagePositionPatient]
    
    # Pixel Spacing: [Row Spacing (Y), Column Spacing (X)]
    spacing_y, spacing_x = [float(val) for val in ds.PixelSpacing]
    
    print("====== Coordinators mapping info ======")
    print(f"O (X0, Y0, Z0): ({origin_x:.2f}, {origin_y:.2f}, {origin_z:.2f}) mm")
    print(f"pix dist (X, Y): ({spacing_x:.2f}, {spacing_y:.2f}) mm")
    print(f"threshold: Only output values > {value_threshold} \n")
    print(f"{'pix coordinators (row, col)':<15} | {'phy coordinators (X, Y, Z) mm':<35} | {'original pix val'}")
    print("-" * 75)
    
    rows, cols = np.where(pixel_matrix > value_threshold)
    
    point_count = len(rows)
    
    if point_count == 0:
        print("No points are found, please devrease the threshold.")
        return
        
    #max_print = min(point_count, 50) # for in case there are too many points to be printed
    max_print = point_count
    
    for i in range(max_print):
        r = rows[i]
        c = cols[i]
        val = pixel_matrix[r, c]
        
        phys_x = origin_x + c * spacing_x
        phys_y = origin_y + r * spacing_y
        phys_z = origin_z
        
        pixel_str = f"({r:3d}, {c:3d})"
        phys_str = f"({phys_x:7.2f}, {phys_y:7.2f}, {phys_z:7.2f})"
        print(f"{pixel_str:<15} | {phys_str:<35} | {val}")
        
    print("-" * 75)
    print(f"Find {point_count} points >= {value_threshold}")

# read the dcm file to get the first pixel's phy coordinates
def get_dcm_x0y0_info(dcm_file):
    try:
        # 读取 DICOM 文件
        ds = pydicom.dcmread(dcm_file)
        
        # 普通单帧 DICOM 的物理坐标通常直接存储在 ImagePositionPatient 标签中
        if hasattr(ds, 'ImagePositionPatient'):
            origin_pt = ds.ImagePositionPatient
            
            # 转换为 Python 原生 float 类型
            phys_x = float(origin_pt[0])
            phys_y = float(origin_pt[1])
            phys_z = float(origin_pt[2])
            
            print(f"---- DICOM 空间信息 ----")
            print(f"像素坐标 (0, 0) 对应的物理空间坐标 (LPS):")
            print(f"X (Right->Left) : {phys_x}")
            print(f"Y (Anterior->Posterior) : {phys_y}")
            print(f"Z (Inferior->Superior) : {phys_z}  <-- 这就是 Slice Location (SL)")
            print(f"------------------------")
            
            return (phys_x, phys_y, phys_z)
        else:
            print(f"警告: 在文件 {dcm_file} 中未找到 ImagePositionPatient 标签。这可能不是一个包含空间信息的图像。")
            return None
            
    except Exception as e:
        print(f"读取文件出错: {e}")
        return None
    

def batch_get_info(folder_path):
    """
    遍历文件夹，打印每个 DICOM 文件的文件名及其左上角第一个像素 (0,0) 的物理坐标。
    """
    # 1. 获取文件夹下所有文件并排序（确保按 1-186, 1-187 这样的顺序排列）
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.dcm')]
    files.sort() # 默认按字母/数字顺序排序
    
    if not files:
        print(f"在路径 {folder_path} 下未找到任何 .dcm 文件。")
        return

    print(f"{'文件名':<20} | {'物理坐标 (X, Y, Z)':<40}")
    print("-" * 70)

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True) # 仅读取头部，速度极快
            
            if hasattr(ds, 'ImagePositionPatient'):
                pos = ds.ImagePositionPatient
                # 转换为 Python float 并保留 2 位小数
                pos_str = f"({float(pos[0]):.2f}, {float(pos[1]):.2f}, {float(pos[2]):.2f})"
                print(f"{filename:<20} | {pos_str}")
            else:
                print(f"{filename:<20} | 未找到 ImagePositionPatient 标签")
                
        except Exception as e:
            print(f"{filename:<20} | 读取失败: {e}")

def analyze_psma_dataset(root_path):
    project_with_others = []
    total_pet_folders = 0
    pure_1_folders = 0
    other_prefix_folders = 0

    # 遍历 Project ID 层 (PSMA_xxxx)
    for project_id in os.listdir(root_path):
        project_path = os.path.join(root_path, project_id)
        if not os.path.isdir(project_path):
            continue

        # 遍历 Study 层 (日期-NA-PETCT...)
        for study_id in os.listdir(project_path):
            study_path = os.path.join(project_path, study_id)
            if not os.path.isdir(study_path):
                continue

            # 遍历 Series 层，寻找包含 "PET" 字样的文件夹
            for series_id in os.listdir(study_path):
                if "PET" in series_id.upper():
                    pet_path = os.path.join(study_path, series_id)
                    if not os.path.isdir(pet_path):
                        continue

                    total_pet_folders += 1
                    
                    # 获取该 PET 文件夹下所有 dcm 文件
                    dcm_files = [f for f in os.listdir(pet_path) if f.lower().endswith('.dcm')]
                    if not dcm_files:
                        continue

                    # 检查前缀
                    has_others = False
                    for f in dcm_files:
                        if not f.startswith("1-"):
                            has_others = True
                            break
                    
                    if has_others:
                        other_prefix_folders += 1
                        project_with_others.append((project_id, study_id, series_id))
                    else:
                        pure_1_folders += 1

    # --- 打印结果 ---
    print("--- 含有 '1-' 以外前缀的文件夹详情 ---")
    print(f"{'Project ID':<25} | {'Study ID':<40} | {'Series Folder'}")
    print("-" * 100)
    for p_id, s_id, ser_id in project_with_others:
        print(f"{p_id:<25} | {s_id:<40} | {ser_id}")

    print("\n" + "="*30)
    print("--- 统计数据汇总 ---")
    print(f"总 PET 文件夹数量:    {total_pet_folders}")
    print(f"全部为 '1-' 的文件夹:  {pure_1_folders}")
    print(f"含有其它前缀的文件夹: {other_prefix_folders}")
    print("="*30)
    

if __name__=='__main__':
    #file_name_process("raw_datasets/BrainTumorMRI/")
    #get_referenced_imaging_info("./1-1.dcm")
    seg_file = "temp/1-1.dcm"
    ref_dir = "temp/3.000000-PET-01743" 
    output_nii = "./case_0001seg.nii.gz"
    #dicom_to_nifti_sitk("temp/4.000000-CT-00452", "./01_ct.nii.gz")
    #get_referenced_imaging_info("temp/1-1.dcm")
    #index, filename, z_pos = find_slice_index(ref_dir, "1.3.6.1.4.1.14519.5.2.1.162115129127399650767843649836237257916")
    #print(f"该 UID 对应物理层序第 {index} 层，文件名是 {filename}")
    #read_segment_sequence("./1-1.dcm")
    #count_lesions("case_0001seg.nii.gz")
    #ds = pydicom.dcmread("temp/4.000000-CT-00452/1-001.dcm")
    #print(f"像素矩阵 Shape: {ds.pixel_array.shape}")
    #visualize_pet_slice("temp/3.000000-PET-01743/1-168.dcm")
    #dump_dicom_metadata("temp/102.000000-PET-11700/2-001.dcm")
    #print_segment_sequence_details("temp/3.000000-PET-01743/1-168.dcm")
    #map_pixels_to_physical_coords("temp/3.000000-PET-01743/1-168.dcm")
    #batch_get_info("temp/102.000000-PET-11700")