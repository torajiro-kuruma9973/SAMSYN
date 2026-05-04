import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from samsyn_train_utils import FocalLoss, DiceLoss

class PETSynthesisLoss(nn.Module):
    #def __init__(self, lambda_l1=10.0, lambda_ssim=5.0, lambda_lesion=0.0, lambda_focal=5.0, lambda_dice=5.0, data_range=1.0):
    def __init__(self, lambda_l1=10.0, lambda_ssim=5.0, lambda_lesion=0.0, data_range=1.0):
        """
        初始化加权损失函数
        :param lambda_l1: L1 损失的权重 (基础权重)
        :param lambda_ssim: SSIM 损失的权重 (通常设得比较大，因为 SSIM 梯度的数值较小)
        :param lambda_lesion: 病灶区域的惩罚权重 (极高，强迫模型关注高代谢区域)
        """
        super(PETSynthesisLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_lesion = lambda_lesion
        self.data_range = data_range
        # self.lambda_focal = lambda_focal
        # self.lambda_dice = lambda_dice

        # self.focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
        # self.dice_loss = DiceLoss(smooth=1.0)

    def calc_l1_loss(self, pred, gt):
        """
        计算全局 L1 损失，约束整体像素强度的平均偏差。
        输入 shape 可以是 [8, 1024, 1024] 或 [8, 1, 1024, 1024]
        """
        return F.l1_loss(pred, gt)
    
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
    
    # def calc_lesion_aware_loss(self, pred, gt, lesion_mask):
    #     """
    #     仅针对病灶区域计算高权重的 L1 损失。
    #     lesion_mask: 形状应与 pred 相同，病灶区域为 1，非病灶区域为 0。
    #                 (这就是你之前用 SAM 2 提取出来的 Label!)
    #     """
    #     # 计算绝对误差图
    #     diff = torch.abs(pred - gt)
        
    #     # 用 mask 过滤掉非病灶区域 (只保留病灶处的误差)
    #     lesion_diff = diff * lesion_mask
        
    #     # 计算病灶区域的平均误差
    #     # ⚠️ 加上 1e-8 是为了防止分母为 0 (万一这个 Batch 里刚好没有病灶)
    #     lesion_pixels_count = lesion_mask.sum() + 1e-8
    #     print(f"@@@@@@@@ count = {lesion_pixels_count} @@@@@@@@@@@@")
        
    #     return lesion_diff.sum() / lesion_pixels_count

    # def calc_lesion_aware_loss(self, pred, gt, lesion_mask):
    #     """
    #     改良版：全局稳定的 Lesion Aware Loss
    #     """
    #     # 1. 计算绝对误差图
    #     diff = torch.abs(pred - gt)
        
    #     # 2. 用 mask 过滤掉非病灶区域
    #     lesion_diff = (diff * lesion_mask) * 500
        
    #     # 3. 🌟 直接在全局求平均 (除以 Batch 的总像素数)
    #     # 这样病灶越大，额外惩罚越多；病灶越小或没有，惩罚趋近于0。极其稳定！
    #     #print(lesion_mask.sum())
    #     return lesion_diff.mean()
    
    def calc_lesion_aware_loss(self, pred, gt, mask, loss_type='mse'):
        """
        计算仅在病灶（前景）区域内的平均损失值。
        
        参数:
            pred (torch.Tensor): 模型的预测输出，例如形状 [B, 1, H, W] 或 [B, 1, D, H, W]
            gt (torch.Tensor): 真实标签 (Ground Truth)，形状需与 pred 一致
            mask (torch.Tensor): 病灶掩码，1 表示病灶，0 表示背景，形状需与 pred 一致
            loss_type (str): 损失类型，目前支持 'l1' (MAE) 和 'mse' (L2)
            
        返回:
            torch.Tensor: 病灶区域的平均损失标量
        """
        # 1. 计算元素级 (Element-wise) 的全局 Loss
        # 注意：务必使用 reduction='none'，这样才会返回每个像素的独立 loss，而不是直接求平均
        if loss_type.lower() == 'l1':
            base_loss = F.l1_loss(pred, gt, reduction='none')
        elif loss_type.lower() == 'mse':
            base_loss = F.mse_loss(pred, gt, reduction='none')
        else:
            raise ValueError(f"❌ 不支持的 Loss 类型: {loss_type}")
            
        # 2. 确保 mask 是浮点型且与 base_loss 形状对齐
        mask = mask.to(base_loss.dtype)
        
        # 3. 将 Loss 过滤，只保留病灶区域 (背景的 loss 全被乘 0 抹去)
        lesion_loss_map = base_loss * mask
        
        # 4. 计算病灶区域的总像素个数
        lesion_pixel_count = torch.sum(mask)
        
        # 5. 计算平均值并处理边界情况
        # 如果当前 Batch 或切片中完全没有病灶，直接除以 0 会产生 NaN (Not a Number)
        if lesion_pixel_count > 0:
            # 只在有病灶的像素上求平均
            final_loss = torch.sum(lesion_loss_map) / lesion_pixel_count
        else:
            # 如果没有病灶，返回 0.0。
            # ⚠️ 关键点：必须使用 requires_grad=True 保持计算图不中断，否则反向传播 (backward) 会报错
            final_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
            
        return final_loss

    def forward(self, pred, gt, lesion_mask=None):
        """
        前向传播计算总 Loss
        :param pred: 模型生成的 PET [8, 1024, 1024]
        :param gt: 真实的 PET [8, 1024, 1024]
        :param lesion_mask: 病灶掩码 [8, 1024, 1024]
        """
        
        loss_l1 = self.calc_l1_loss(pred, gt)
        loss_ssim = self.calc_ssim_loss(pred, gt, data_range=self.data_range)
        
        if lesion_mask is not None:
            loss_lesion = self.calc_lesion_aware_loss(pred, gt, lesion_mask)
        else:
            # 如果某个样本没有提供 mask，此项 Loss 为 0
            loss_lesion = torch.tensor(0.0, device=pred.device)

        # loss_focal = self.focal_loss(pred, gt)
        # loss_dice = self.dice_loss(pred, gt)

        # 2. 按照 lambda 权重进行加权求和
        # total_loss = (self.lambda_l1 * loss_l1) + \
        #              (self.lambda_ssim * loss_ssim) + \
        #              (self.lambda_lesion * loss_lesion) + \
        #              (self.lambda_dice * loss_dice) + \
        #              (self.lambda_focal * loss_focal)
        
        total_loss = (self.lambda_l1 * loss_l1) + \
                     (self.lambda_ssim * loss_ssim) + \
                     (self.lambda_lesion * loss_lesion)
                     
        
        #return total_loss, loss_l1, loss_ssim, loss_lesion, loss_focal, loss_dice
        return total_loss, loss_l1, loss_ssim, loss_lesion