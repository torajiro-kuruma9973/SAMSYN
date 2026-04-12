#!/bin/bash

# ================= 配置区 =================
N=12                                  # 设定的阈值 N
SRC_DIR="samsyn_dataset/labels"      # 源文件夹绝对路径 (替换为实际路径)
DEST_DIR="samsyn_datasets_backup/labels_backup"     # 目标文件夹绝对路径 (替换为实际路径)
# ==========================================

# 确保目标文件夹存在，不存在则自动创建
mkdir -p "$DEST_DIR"

echo "🚀 开始扫描并移动大于 $N 的文件..."

# 遍历源文件夹下所有的 .nii.gz 文件
for file in "$SRC_DIR"/*.nii.gz; do
    
    # 1. 提取纯文件名 (例如把 /path/12.nii.gz 变成 12.nii.gz)
    filename=$(basename "$file")
    
    # 2. 提取文件名前缀的数字 n (利用 bash 字符串截取，去掉第一个点之后的所有内容)
    n="${filename%%.*}"
    
    # 3. 核心判断逻辑
    #    条件一：确保提取出的 n 是纯数字 (防止混入如 mask.nii.gz 导致报错)
    #    条件二：判断数字 n 是否严格大于 N (-gt 代表 greater than)
    if [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -gt "$N" ]; then
        echo "  ➡️ 正在移动: $filename"
        mv "$file" "$DEST_DIR/"
    fi
    
done

echo "✅ 任务完成！"