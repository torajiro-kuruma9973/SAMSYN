import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

def load_resize_and_check_nii(file_path, target_size=(1024, 1024), is_label=False):
    print(f"📂 正在读取并处理: {file_path}")
    print("-" * 50)
    
    # 1. 读取并强转 float32
    img = nib.load(file_path)
    data_fp32 = img.get_fdata().astype(np.float32)
    
    # 假设原始形状是 (H, W, Slices)，例如 (512, 512, 150)
    print(f"1️⃣ 原始 float32 数据:")
    print(f"   - 形状: {data_fp32.shape}")
    print(f"   - 内存: {data_fp32.nbytes / (1024**2):.2f} MB")
    
    # 2. 维度重排：(H, W, Slices) -> (Slices, H, W)
    # 深度学习中，通常把切片数当作 Batch 或 Sequence 处理，放在第 0 维
    data_transposed = np.transpose(data_fp32, (2, 0, 1))
    
    # 转为 PyTorch Tensor，并增加通道维度 -> (Slices, 1, H, W)
    tensor_data = torch.from_numpy(data_transposed).unsqueeze(1)
    
    # 3. 执行插值缩放 (Resize) 到 1024x1024
    # 注意：如果是图像原图用双线性插值(bilinear)，如果是Label掩码必须用最近邻(nearest)防止出现小数！
    interp_mode = 'nearest' if is_label else 'bilinear'
    align = None if is_label else False
    
    tensor_resized = F.interpolate(
        tensor_data, 
        size=target_size, 
        mode=interp_mode, 
        align_corners=align
    )
    
    # 计算当前 Tensor 占用的内存 (元素个数 * 每个元素占用的字节数)
    resized_mb = tensor_resized.numel() * tensor_resized.element_size() / (1024**2)
    print(f"\n2️⃣ Resize 到 {target_size} 后 (单通道):")
    print(f"   - 形状: {tuple(tensor_resized.shape)}")
    print(f"   - 内存: {resized_mb:.2f} MB")
    
    # 4. Tile 成 3 通道 (在 PyTorch 里叫 repeat)
    # 沿着第1个维度（通道维）复制 3 遍，形状变成 (Slices, 3, 1024, 1024)
    tensor_final = tensor_resized.repeat(1, 3, 1, 1)
    
    final_mb = tensor_final.numel() * tensor_final.element_size() / (1024**2)
    print(f"\n3️⃣ Tile 成 3 通道后 (最终喂给模型的数据):")
    print(f"   - 形状: {tuple(tensor_final.shape)}")
    print(f"   - 内存: {final_mb:.2f} MB (📈 内存暴增处！)")
    print("-" * 50)
    
    return tensor_final

# ================= 测试运行 =================
# 如果处理的是 CT/MRI 原图：
image_tensor = load_resize_and_check_nii("samsyn_dataset/data/20.nii.gz", is_label=False)

# 如果处理的是对应的 Mask 标签：
# label_tensor = load_resize_and_check_nii("your_label.nii.gz", is_label=True)