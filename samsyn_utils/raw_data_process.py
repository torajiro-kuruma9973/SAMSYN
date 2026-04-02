import os
import pydicom
import numpy as np
import SimpleITK as sitk
import pet_preprocess as pp
import ct_preprocess as cp
import seg_process as sp
import json

# save the lasions pix coordinates on CT file as json.
import os
import json

def get_lasions_info_from_ct(root_path, output_json_path):

    all_results = {}

    projects = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    print(f"find {len(projects)} project folders，processing...")

    for project_id in projects:
        project_dir = os.path.join(root_path, project_id)
        
        petct_dir = None
        ct_dir = None
        seg_file_path = None
        # there is a "study id" project under "project id" folder.
        # so we need to continue to look for the ct foloer, pet folder and seg folder
        for sub_folder in os.listdir(project_dir):
            if "PETCT" in sub_folder.upper() and os.path.isdir(os.path.join(project_dir, sub_folder)):
                petct_dir = os.path.join(project_dir, sub_folder)
                break  
        
        if not petct_dir:
            print(f"sikp [{project_id}]: cannot find 'PETCT' sub folders。")
            continue

    
        for leaf_folder in os.listdir(petct_dir):
            leaf_path = os.path.join(petct_dir, leaf_folder)
            if not os.path.isdir(leaf_path):
                continue
                
            leaf_upper = leaf_folder.upper()
            
    
            if "SEGMENTATION" in leaf_upper:
               
                seg_files = [f for f in os.listdir(leaf_path) if f.lower().endswith('.dcm')]
                if seg_files:
                    seg_file_path = os.path.join(leaf_path, seg_files[0])
            elif "PET" in leaf_upper:
                pass
            elif "CT" in leaf_upper:
                ct_dir = leaf_path

        if not ct_dir or not seg_file_path:
            print(f"skip [{project_id}]: lacks of CT folder or SEG folder.")
            continue

        try:
            print(f"processing: {project_id} ...")
            
            space_info_dict = sp.extract_seg_physical_coords(seg_file_path)
            
            ct_pixel_coords = cp.map_physical_to_ct_pixels(space_info_dict, ct_dir)
            
            all_results[project_id] = ct_pixel_coords
            
        except Exception as e:
            print(f"failed: [{project_id}]: {e}")

    try:
        out_dir = os.path.dirname(output_json_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4)
        print(f"\nDone! json file is saved in: {output_json_path}")
    except Exception as e:
        print(f"\nFailed!!: {e}")

def parse_json_to_dict(json_file_path):
    """
    读取并解析 JSON 文件，返回一个字典。
    
    参数:
        json_file_path (str): JSON 文件的路径
        
    返回:
        dict: 解析后的字典。如果读取失败，则返回空字典 {}
    """
    try:
        # 使用 utf-8 编码打开文件，防止读取中文时报错
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data_dict = json.load(file)
            return data_dict
            
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 '{json_file_path}'，请检查路径是否正确。")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ 错误: '{json_file_path}' 不是有效的 JSON 格式。详细信息: {e}")
        return {}
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        return {}

# 1. Pick the folders only contain PET files with prefix "1-". (done in Windows)
# 2. Abstract the foreground pix coordinates on CT, by uising seg files.
# save as json file.
#get_lasions_info_from_ct("raw_datasets", "temp_results/lesion_pixel_coords.json")
rst = parse_json_to_dict("temp_results/lesion_pixel_coords.json")
print(rst)


# pp.preprocess_pet(ct_dicom_dir, pet_dicom_dir, output_nii_path)

# cp.preprocess_ct(ct_dicom_dir, output_nii_path, target_spacing=(2.0, 2.0, 2.0), clip_min=-1000, clip_max=1000)

# # 4. (可选) 处理 CT 自身 (截断 HU、归一化等，如果是跨患者统一尺寸的话)
# target_ct = process_CT(ct_img)

# # 5. 分别保存，大功告成！
# sitk.WriteImage(target_ct, "patient_01_0000.nii.gz")
# sitk.WriteImage(target_pet, "patient_01_0001.nii.gz")
# sitk.WriteImage(target_seg, "patient_01_seg.nii.gz")