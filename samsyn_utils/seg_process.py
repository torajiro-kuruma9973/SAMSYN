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


def extract_seg_physical_coords(seg_path):
    """
    遍历 SEG 文件，提取不为 0 的前景点，并严格按照 DICOM Header 的变换矩阵计算其物理坐标。
    
    参数:
        seg_path (str): 3D volume 的 DICOM SEG 文件路径。
        
    返回:
        dict: { frame_idx (int): [(x, y, z), (x, y, z), ...] } 
              其中坐标均为 float 类型，保留 4 位小数。
    """
    # 1. 读取 DICOM SEG 文件
    ds = pydicom.dcmread(seg_path)
    pixel_array = ds.pixel_array  # 形状通常为 (Frames, Rows, Columns)
    
    # 2. 获取【全局共享】的空间变换参数
    # SEG 文件强制使用多帧结构，公共参数存储在 SharedFunctionalGroupsSequence 中
    shared_group = ds.SharedFunctionalGroupsSequence[0]
    
    # 方向余弦向量 (ImageOrientationPatient)
    # 前 3 个值代表行延伸方向 (Row direction, 对应列索引变化 / X轴)
    # 后 3 个值代表列延伸方向 (Column direction, 对应行索引变化 / Y轴)
    orientation = shared_group.PlaneOrientationSequence[0].ImageOrientationPatient
    row_dir = np.array([float(v) for v in orientation[:3]])
    col_dir = np.array([float(v) for v in orientation[3:]])
    
    # 像素间距 (PixelSpacing): [相邻行的中心距(dy), 相邻列的中心距(dx)]
    spacing = shared_group.PixelMeasuresSequence[0].PixelSpacing
    dy = float(spacing[0])
    dx = float(spacing[1])
    
    results = {}
    
    # 3. 遍历每一帧
    for frame_idx in range(pixel_array.shape[0]):
        frame_data = pixel_array[frame_idx]
        
        # 找到所有不为 0 的像素索引 (y对应行/Row, x对应列/Column)
        y_indices, x_indices = np.where(frame_data != 0)
        
        # 如果全为0 (无病灶)，直接跳过该帧
        if len(y_indices) == 0:
            continue
            
        # 4. 获取【当前帧专属】的物理原点
        # 即当前帧左上角第一个像素 (0,0) 的 LPS 绝对三维坐标
        frame_group = ds.PerFrameFunctionalGroupsSequence[frame_idx]
        origin = np.array([float(v) for v in frame_group.PlanePositionSequence[0].ImagePositionPatient])
        
        # 5. 核心：计算实际物理空间坐标 (利用 Numpy 向量化加速)
        # DICOM 坐标公式: P(x,y) = Origin + (x * dx * RowDir) + (y * dy * ColDir)
        
        # 将 x, y 偏移量转为列向量，形状从 (N,) 变为 (N, 1) 以便与形状为 (3,) 的方向向量广播相乘
        x_offsets = (x_indices * dx)[:, np.newaxis]
        y_offsets = (y_indices * dy)[:, np.newaxis]
        
        # 向量化计算当前帧所有前景点的 3D 坐标
        # world_coords 的形状将是 (N, 3)
        world_coords = origin + (x_offsets * row_dir) + (y_offsets * col_dir)
        
        # 6. 将 Numpy 数组转换为 Python 原生 list(tuple) 并保留精度
        # 这样在后续作为 JSON 导出时不会报错
        frame_points = [
            (round(pt[0], 4), round(pt[1], 4), round(pt[2], 4)) 
            for pt in world_coords.tolist()
        ]
        
        results[frame_idx] = frame_points
        
    return results

if __name__=='__main__':
    seg_rst = extract_seg_physical_coords("raw_datasets/PSMA_2e5119d4ac37d41d/06-29-1999-NA-PETCT whole-body PSMA-90640/300.000000-Segmentation - Tumor lesions detected-49927/1-1.dcm")
    #rst = locate_pos(seg_rst, "temp/4.000000-CT-00452")
    for key in seg_rst.keys():
        print(f"@@ {key} @@")
        print(seg_rst[key])
        print("----")
