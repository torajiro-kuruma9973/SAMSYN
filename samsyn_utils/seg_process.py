import os
import pydicom
import numpy as np
import SimpleITK as sitk
from collections import defaultdict

# seg file relies on PET file.
# this function lists all the referrenced PET dcm files
def show_ref_pet(seg_file_path, pet_dir):
    pet_uid_to_filename = {}
    
    for f in os.listdir(pet_dir):
        if f.lower().endswith('.dcm'):
            pet_path = os.path.join(pet_dir, f)
            try:
                # read metadata
                ds = pydicom.dcmread(pet_path, stop_before_pixels=True)
                pet_uid_to_filename[ds.SOPInstanceUID] = f
            except Exception as e:
                pass

    print(f"Get {len(pet_uid_to_filename)} PET files info")
    
    try:
        seg_ds = pydicom.dcmread(seg_file_path, stop_before_pixels=True)
        num_frames = getattr(seg_ds, 'NumberOfFrames', 0)
    except Exception as e:
        print(f"Error: {e}")
        return

    if num_frames == 0:
        print("This is not a 3D DICOM-SEG file...")
        return

    print("-" * 60)
    print(f"{'SEG idx':<22} | {'PET filename'}")
    print("-" * 60)
    
    for i in range(num_frames):
        try:
            frame_seq = seg_ds.PerFrameFunctionalGroupsSequence[i]
            
            ref_uid = frame_seq.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
            
            if ref_uid in pet_uid_to_filename:
                target_filename = pet_uid_to_filename[ref_uid]
                print(f"SEG Frame [{i:03d}]          --->   {target_filename}")
            else:
                print(f"SEG Frame [{i:03d}]          --->   Lack of (UID: {ref_uid})")
                
        except AttributeError:
            print(f"SEG Frame [{i:03d}]          --->   cannot get correct path")

    print("-" * 60)

import pydicom
import numpy as np
import os

def get_frontground_from_seg(seg_file):
   
    ds_seg = pydicom.dcmread(seg_file)
    seg_array = ds_seg.pixel_array # (Frames, Rows, Cols)
    
    shared_group = ds_seg.SharedFunctionalGroupsSequence[0]
    orientation = shared_group.PlaneOrientationSequence[0].ImageOrientationPatient

    row_dir = np.array([float(v) for v in orientation[:3]]) # x axis
    col_dir = np.array([float(v) for v in orientation[3:]]) # y axis
    
    spacing = shared_group.PixelMeasuresSequence[0].PixelSpacing
    dy, dx = float(spacing[0]), float(spacing[1])
    
    space_info = {}
    
    for frame_idx in range(seg_array.shape[0]):
        frame_data = seg_array[frame_idx]
        
        y_indices, x_indices = np.where(frame_data != 0)
        if len(y_indices) == 0:
            continue
            
        frame_group = ds_seg.PerFrameFunctionalGroupsSequence[frame_idx]
        origin_pt = np.array([float(v) for v in frame_group.PlanePositionSequence[0].ImagePositionPatient])
        
        frame_points = []
        for y, x in zip(y_indices, x_indices):
         
            phys_pt = origin_pt + (x * dx * row_dir) + (y * dy * col_dir)
            
            frame_points.append((
                float(round(phys_pt[0], 4)), 
                float(round(phys_pt[1], 4)), 
                float(round(phys_pt[2], 4))
            ))
            
        space_info[frame_idx] = frame_points
        
    return space_info

# mapping the lasions pos to CT pix space.
# space_dict: return from "get_frontground_from_seg"
def locate_pos(space_dict, ct_dir):
  
    ct_index = {} # { z_coord: {"name": str, "origin": array, "spacing": (dy, dx)} }
    
    ct_files = [f for f in os.listdir(ct_dir) if f.lower().endswith('.dcm')]
    for f in ct_files:
        path = os.path.join(ct_dir, f)
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        
        if hasattr(ds, 'ImagePositionPatient'):
            pos = [float(v) for v in ds.ImagePositionPatient]
            z_val = round(pos[2], 2) # 取两位小数防止浮点误差
            spacing = [float(v) for v in ds.PixelSpacing] # [dy, dx]
            
            ct_index[z_val] = {
                "name": f,
                "origin": np.array(pos),
                "spacing": spacing
            }

    final_mapping = {}

    for seg_idx, points in space_dict.items():
        for (wx, wy, wz) in points:
        
            target_z = round(wz, 2)
            
            if target_z in ct_index:
                ct_info = ct_index[target_z]
                ct_name = ct_info["name"]
                origin = ct_info["origin"]
                dy, dx = ct_info["spacing"]
                
                px = int(round((wx - origin[0]) / dx))
                py = int(round((wy - origin[1]) / dy))
                
                if ct_name not in final_mapping:
                    final_mapping[ct_name] = []
                
                if 0 <= px < 512 and 0 <= py < 512:
                    final_mapping[ct_name].append((px, py))

    for ct_name in final_mapping:
        final_mapping[ct_name] = list(set(final_mapping[ct_name]))
        
    return final_mapping

if __name__=='__main__':
    seg_rst = get_frontground_from_seg("temp/1-1.dcm")
    rst = locate_pos(seg_rst, "temp/4.000000-CT-00452")
    for key in rst.keys():
        print(f"@@ {key} @@")
        print(rst[key])
        print("----")

    show_ref_pet("temp/1-1.dcm", "temp/3.000000-PET-01743")