import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from samsyn_train_utils import FocalLoss, DiceLoss
import warnings

class PETSynthesisLoss(nn.Module):
   
    def __init__(self, lambda_l1=5.0, lambda_ssim=10.0, lambda_high_suv=20.0, data_range=1.0):
        """
        初始化加权损失函数
        :param lambda_l1: L1 损失的权重 (基础权重)
        :param lambda_ssim: SSIM 损失的权重 (通常设得比较大，因为 SSIM 梯度的数值较小)
        """
        super(PETSynthesisLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_high_suv = lambda_high_suv
        
        self.data_range = data_range

    def calc_l1_loss(self, pred, gt):
        """
        计算全局 L1 损失，约束整体像素强度的平均偏差。
        输入 shape 可以是 [8, 1024, 1024] 或 [8, 1, 1024, 1024]
        """
        return F.l1_loss(pred, gt)
    
    def topk_brightness_l1_loss(self, pred: torch.Tensor, gt: torch.Tensor, a: float = 0.1) -> float:
        """
        对每一帧，取 gt 中最亮的前 a 比例像素的坐标，
        仅在这些坐标上计算 pred 与 gt 的 L1 loss，
        最后对所有帧、所有选中像素求平均（等价于 F.l1_loss 的 mean reduction）。

        Args:
            pred (torch.Tensor): 模型推理结果，形状 [N, 1, 1024, 1024]，
                                数值范围 [0, 1]。
            gt (torch.Tensor): 真实值，形状 [N, 1, 1024, 1024]，
                                数值范围 [0, 1]。
            a (float): 最亮像素比例，默认 0.2，取值范围 (0, 1]。

        Returns:
            float: 标量 loss 值。
        """
        assert 0.0 < a <= 1.0, f"a must be in (0, 1], got {a}"

        # ---- 形状 / 数值范围校验 ----
        assert pred.shape == gt.shape, (
            f"pred and gt must have the same shape, got {pred.shape} vs {gt.shape}"
        )
        N, C, H, W = pred.shape
        assert C == 1 and H == 1024 and W == 1024, (
            f"expected shape [N, 1, 1024, 1024], got {pred.shape}"
        )
        assert 1 <= N <= 8, f"N must satisfy 1 <= N <= 8, got {N}"

        # 数值应在 [0, 1]（已做归一化）。这里用 assert 做一次防御性检查，
        # 如果你的 pipeline 里存在浮点误差导致轻微越界，可把 assert 换成 clamp。
        assert pred.min() >= 0.0 and pred.max() <= 1.0, "pred values must be in [0, 1]"
        assert gt.min() >= 0.0 and gt.max() <= 1.0, "gt values must be in [0, 1]"

        # ---- gt 梯度检查（仅提示，不强制阻断训练）----
        if gt.requires_grad:
            warnings.warn(
                "gt.requires_grad=True. topk_brightness_l1_loss 内部会对 gt 调用 detach() "
                "来选取最亮像素坐标，因此不会有梯度经由『选择索引』这一步回传，"
                "但请确认这是你期望的行为（gt 通常不需要梯度）。"
            )

        total_pixels = H * W  # 1024 * 1024
        k = int(total_pixels * a)  # 向下取整
        assert k > 0, f"a={a} too small, computed k=0 (total_pixels={total_pixels})"

        # 用于挑选坐标的 gt 必须 detach，保证"选择"这一步不参与反向传播
        gt_for_index = gt.detach()

        pred_flat = pred.reshape(N, total_pixels)
        gt_flat_for_index = gt_for_index.reshape(N, total_pixels)
        gt_flat = gt.reshape(N, total_pixels)

        # 每帧独立选出最亮的 k 个坐标（严格 k 个，largest=True）
        _, topk_idx = torch.topk(
            gt_flat_for_index, k=k, dim=1, largest=True, sorted=False
        )

        pred_topk = torch.gather(pred_flat, dim=1, index=topk_idx)  # [N, k]
        gt_topk = torch.gather(gt_flat, dim=1, index=topk_idx)      # [N, k]

        # 直接复用 F.l1_loss 的 mean reduction：
        # 等价于 sum(|pred_topk - gt_topk|) / (N * k)，
        # 与 F.l1_loss(pred, gt) 对整张图求 mean 的语义完全一致
        loss = F.l1_loss(pred_topk, gt_topk, reduction="mean")
        return loss
    
    def calc_ssim_loss(self, pred, gt, data_range=1.0):
        """
        计算结构相似性损失。SSIM 值越大越好（最高为1），所以 Loss = 1 - SSIM。
        ⚠️ data_range 是你的图像像素值范围：
        - 如果你归一化到了 [0, 1]，填 1.0
        - 如果你归一化到了 [-1, 1]，填 2.0
        - 如果是原始 SUV 值，填你的最大截断值
        """
        # SSIM 强制要求 4D 输入 (B, C, H, W)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
            gt = gt.unsqueeze(1)
            
        # 计算 SSIM，size_average=True 表示对整个 Batch 求平均
        ssim_val = ssim(pred, gt, data_range=data_range, size_average=True)
        return 1.0 - ssim_val
    
    # def calc_lesion_aware_loss(self, pred, gt, mask, loss_type='l1'):
    #     """
    #     计算仅在病灶（前景）区域内的平均损失值。
        
    #     参数:
    #         pred (torch.Tensor): 模型的预测输出，例如形状 [B, 1, H, W] 或 [B, 1, D, H, W]
    #         gt (torch.Tensor): 真实标签 (Ground Truth)，形状需与 pred 一致
    #         mask (torch.Tensor): 病灶掩码，1 表示病灶，0 表示背景，形状需与 pred 一致
    #         loss_type (str): 损失类型，目前支持 'l1' (MAE) 和 'mse' (L2)
            
    #     返回:
    #         torch.Tensor: 病灶区域的平均损失标量
    #     """
    #     # 1. 计算元素级 (Element-wise) 的全局 Loss
    #     # 注意：务必使用 reduction='none'，这样才会返回每个像素的独立 loss，而不是直接求平均
    #     if loss_type.lower() == 'l1':
    #         base_loss = F.l1_loss(pred, gt, reduction='none')
    #     elif loss_type.lower() == 'mse':
    #         base_loss = F.mse_loss(pred, gt, reduction='none')
    #     else:
    #         raise ValueError(f"❌ 不支持的 Loss 类型: {loss_type}")
            
    #     # 2. 确保 mask 是浮点型且与 base_loss 形状对齐
    #     mask = mask.to(base_loss.dtype)
        
    #     # 3. 将 Loss 过滤，只保留病灶区域 (背景的 loss 全被乘 0 抹去)
    #     lesion_loss_map = base_loss * mask
        
    #     # 4. 计算病灶区域的总像素个数
    #     lesion_pixel_count = torch.sum(mask)
        
    #     # 5. 计算平均值并处理边界情况
    #     # 如果当前 Batch 或切片中完全没有病灶，直接除以 0 会产生 NaN (Not a Number)
    #     if lesion_pixel_count > 0:
    #         # 只在有病灶的像素上求平均
    #         final_loss = torch.sum(lesion_loss_map) / lesion_pixel_count
    #     else:
    #         # 如果没有病灶，返回 0.0。
    #         # ⚠️ 关键点：必须使用 requires_grad=True 保持计算图不中断，否则反向传播 (backward) 会报错
    #         final_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
            
    #     return final_loss


    def forward(self, pred, gt):
        """
        前向传播计算总 Loss
        :param pred: 模型生成的 PET [8, 1024, 1024]
        :param gt: 真实的 PET [8, 1024, 1024]
        """
        
        loss_l1 = self.calc_l1_loss(pred, gt)
        loss_ssim = self.calc_ssim_loss(pred, gt, data_range=self.data_range)
        loss_high_suv = self.topk_brightness_l1_loss(pred, gt)
        
        total_loss = (self.lambda_l1 * loss_l1) + (self.lambda_ssim * loss_ssim) + (self.lambda_high_suv * loss_high_suv)
        #print(f"self.lambda_l1 = {self.lambda_l1}, self.lambda_ssim = {self.lambda_ssim}, self.lambda_high_suv = {self.lambda_high_suv}")
        
        return total_loss, loss_l1, loss_ssim, loss_high_suv