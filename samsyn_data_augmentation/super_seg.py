import os
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import pack_bits

def enhance_segmentation_with_pet(root_dir):
    """
    遍历整个根目录，找到配对的 PET 和 Segmentation 文件夹。
    逐帧匹配，将 PET 中代谢值高于当前前景最大值的像素补充进 SEG 掩码中，
    并在同目录下保存为 super.dcm。
    """
    print(f"🚀 开始深度扫描根目录: {root_dir}")
    
    # 步骤 1：遍历根目录，寻找每一个 Study 层级
    # os.walk 会遍历每一个子目录。当我们在某个子目录下同时发现 pet 和 seg 文件夹时，
    # 说明我们到达了正确的 Study 层级（例如 04-02-2003-NA-PETCT...）
    for current_root, dir_names, file_names in os.walk(root_dir):
        pet_dir = None
        seg_dir = None
        
        for dir_name in dir_names:
            lower_name = dir_name.lower()
            full_path = os.path.join(current_root, dir_name)
            
            # 识别 Seg 和 PET 文件夹
            if "segmentation" in lower_name:
                seg_dir = full_path
            elif "pet" in lower_name:
                pet_dir = full_path
                
        # 如果在当前目录下同时找到了 PET 和 SEG 文件夹，就开始处理这组数据！
        if pet_dir and seg_dir:
            print(f"\n=======================================================")
            print(f"📂 发现待处理的 Study: {os.path.basename(current_root)}")
            print(f"   ➤ PET 目录: {os.path.basename(pet_dir)}")
            print(f"   ➤ SEG 目录: {os.path.basename(seg_dir)}")
            
            try:
                _process_single_study(pet_dir, seg_dir)
            except Exception as e:
                print(f"❌ 处理 {current_root} 时发生错误: {str(e)}")

def _process_single_study(pet_dir, seg_dir):
    """
    处理单个 Study 目录下的数据：匹配 UID、扩充掩码、保存 super.dcm
    """
    # ---------------------------------------------------------
    # 1. 扫描 PET 文件夹，建立 UID -> 文件路径 的映射字典
    # ---------------------------------------------------------
    pet_uid_to_path = {}
    for filename in os.listdir(pet_dir):
        if filename.endswith(".dcm") or filename.endswith(".IMA"):
            file_path = os.path.join(pet_dir, filename)
            # 仅读取头部信息，极大加快扫描速度
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            if hasattr(ds, 'SOPInstanceUID'):
                pet_uid_to_path[ds.SOPInstanceUID] = file_path

    # ---------------------------------------------------------
    # 2. 读取 SEG 掩码文件
    # ---------------------------------------------------------
    seg_files = [f for f in os.listdir(seg_dir) if f.endswith(".dcm") or f.endswith(".IMA")]
    # 过滤掉我们自己生成的 super.dcm (防止重复处理)
    seg_files = [f for f in seg_files if f != "super.dcm"]
    
    if not seg_files:
        print(f"⚠️ 在 {seg_dir} 中没有找到原始 SEG 文件，已跳过。")
        return
        
    seg_path = os.path.join(seg_dir, seg_files[0])
    seg_ds = pydicom.dcmread(seg_path)
    
    # 确保数据被解压，并拷贝出一份可修改的 NumPy 数组
    if seg_ds.file_meta.TransferSyntaxUID.is_compressed:
        seg_ds.decompress()
    seg_pixel_array = seg_ds.pixel_array.copy()
    
    total_frames = seg_pixel_array.shape[0]
    enhanced_frames_count = 0

    print(f"⏳ 开始逐帧处理 (共 {total_frames} 帧)...")
    
    # ---------------------------------------------------------
    # 3. 逐帧遍历并进行物理数值比对
    # ---------------------------------------------------------
    for frame_idx, frame_item in enumerate(seg_ds.PerFrameFunctionalGroupsSequence):
        try:
            # 提取本帧参考的 PET 切片 UID
            ref_uid = frame_item.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
        except AttributeError:
            continue
            
        pet_path = pet_uid_to_path.get(ref_uid)
        if not pet_path:
            continue
            
        # 读取完整 PET 切片数据
        pet_ds = pydicom.dcmread(pet_path)
        pet_img = pet_ds.pixel_array.astype(np.float64)
        
        # 还原真实的物理数值 (SUV 或 放射性浓度)
        slope = getattr(pet_ds, 'RescaleSlope', 1.0)
        intercept = getattr(pet_ds, 'RescaleIntercept', 0.0)
        pet_img = pet_img * slope + intercept
        
        # 提取当前帧的掩码
        seg_frame = seg_pixel_array[frame_idx]
        
        # 寻找当前帧已有的前景点 (mask > 0)
        fg_mask = seg_frame > 0
        
        if not fg_mask.any():
            # 本帧没有前景点，直接跳过
            continue
            
        # 找到前景点对应在 PET 上的最大物理数值
        max_fg_val = float(np.max(pet_img[fg_mask]))
        
        # 🌟 核心逻辑：找出 PET 中所有严格大于 max_fg_val 的区域
        new_hotspots_mask = pet_img > max_fg_val
        
        # 如果确实找到了新的高代谢点，把它们标记为前景点 (1)
        if new_hotspots_mask.any():
            seg_frame[new_hotspots_mask] = 1
            enhanced_frames_count += 1
            # 覆盖回原数组
            seg_pixel_array[frame_idx] = seg_frame

    # ---------------------------------------------------------
    # 4. 将修改后的数组重新打包并保存为 super.dcm
    # ---------------------------------------------------------
    # 医学 DICOM SEG 文件经常是 1-bit 压缩存储的。
    # 我们必须根据 BitsAllocated 决定如何把 NumPy 转回 Bytes，否则图片会花屏！
    if getattr(seg_ds, 'BitsAllocated', 8) == 1:
        seg_ds.PixelData = pack_bits(seg_pixel_array)
    else:
        seg_ds.PixelData = seg_pixel_array.tobytes()
        
    output_path = os.path.join(seg_dir, "super.dcm")
    seg_ds.save_as(output_path)
    
    print(f"✅ 处理完毕！")
    print(f"   ➤ 共有 {enhanced_frames_count} / {total_frames} 帧被高代谢点增强。")
    print(f"   ➤ 增强后的文件已保存至: {output_path}")


def compare_seg_dicoms(file1_path, file2_path):
    """
    对比两个 3D SEG DICOM 文件的每一帧，找出像素值不同的坐标点。
    
    参数:
        file1_path (str): 第一个 SEG 文件的路径
        file2_path (str): 第二个 SEG 文件的路径
        
    返回:
        dict: 包含差异详情的字典。格式为 {frame_index: [(y, x, val1, val2), ...]}
    """
    print(f"⏳ 正在读取文件 1: {file1_path}")
    ds1 = pydicom.dcmread(file1_path)
    # 对于部分压缩的 SEG 文件，确保解压
    if ds1.file_meta.TransferSyntaxUID.is_compressed:
        ds1.decompress()
    arr1 = ds1.pixel_array
    
    print(f"⏳ 正在读取文件 2: {file2_path}")
    ds2 = pydicom.dcmread(file2_path)
    if ds2.file_meta.TransferSyntaxUID.is_compressed:
        ds2.decompress()
    arr2 = ds2.pixel_array

    # 1. 基础校验：确保帧数和分辨率完全一致
    if arr1.shape != arr2.shape:
        raise ValueError(f"❌ 形状不匹配！文件1形状: {arr1.shape}, 文件2形状: {arr2.shape}")
        
    num_frames = arr1.shape[0]
    print(f"\n✅ 文件读取成功，开始比对 (共 {num_frames} 帧)...\n")
    
    # 用于存储所有差异的字典
    all_differences = {}
    total_diff_frames = 0

    # 2. 逐帧遍历比对
    for i in range(num_frames):
        frame1 = arr1[i]
        frame2 = arr2[i]
        
        # 使用 numpy 的矢量化操作，瞬间找出不相等的像素坐标
        diff_mask = frame1 != frame2
        diff_y, diff_x = np.where(diff_mask)
        
        diff_count = len(diff_y)
        
        if diff_count == 0:
            print(f"✅ 帧 {i:03d}: 完美匹配，没有任何差异。")
        else:
            total_diff_frames += 1
            print(f"⚠️ 帧 {i:03d}: 发现 {diff_count} 个像素不一致！")
            
            frame_diff_list = []
            # 提取具体的数值
            vals1 = frame1[diff_y, diff_x]
            vals2 = frame2[diff_y, diff_x]
            
            for idx in range(diff_count):
                y, x = diff_y[idx], diff_x[idx]
                v1, v2 = vals1[idx], vals2[idx]
                
                # 存入列表，转为 Python 原生 int 格式以便后续使用
                frame_diff_list.append((int(y), int(x), int(v1), int(v2)))
                
                # 为了防止终端刷屏，每帧最多只打印前 3 个差异点作为示例
                if idx < 3:
                    print(f"     -> 坐标 (y={y:4d}, x={x:4d}): 文件1 = {v1}, 文件2 = {v2}")
            
            if diff_count > 3:
                print(f"     -> ... (已省略其余 {diff_count - 3} 个不同点，详情已存入返回的字典中)")
                
            all_differences[i] = frame_diff_list

    # 3. 打印最终总结
    print("\n=========================================")
    if total_diff_frames == 0:
        print("🎉 恭喜！这两个 SEG 文件的所有帧完全一模一样！")
    else:
        print(f"📊 比对完成。共有 {total_diff_frames} 帧存在差异。")
        
    return all_differences

# ===============================
# 运行入口
# ===============================
if __name__ == "__main__":
    # 只需要把你的根目录路径传进去即可
    # 比如: target_directory = r"C:\Users\YourName\Data\PSMA_Dataset"
    #target_directory = "./PSMA-PET-CT-Lesions" 
    
    #enhance_segmentation_with_pet(target_directory)
    #print("\n🎉 所有 Study 文件夹处理完毕！")

    file_a = "./temp/1-1.dcm"
    file_b = "./temp/super.dcm"
    
    diff_dict = compare_seg_dicoms(file_a, file_b)