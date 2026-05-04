# set up environment
import numpy as np
import random 
import matplotlib.pyplot as plt
import os
join = os.path.join
import samsyn_cfg
from tqdm import tqdm
from torch.backends import cudnn
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from samsyn_test_dataloaders.dataloader import get_dataset_3d
from samsyn_train_utils import DiceLoss, build_model, get_logger
import nibabel as nib  
import cv2
from torch.nn import CrossEntropyLoss
from samsyn_losses import PETSynthesisLoss

from torch.utils.tensorboard import SummaryWriter
from samsyn_json_metadata import utils

import warnings
warnings.filterwarnings("ignore") 
warnings.filterwarnings("ignore", category=UserWarning)

import SimpleITK as sitk
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default=samsyn_cfg.test_results_path)
parser.add_argument('--task_name', type=str, default='PET_test')
#load data
#parser.add_argument("--data_root", type = str, default='datasets/BraTS2020/FLAIR')
parser.add_argument("--data_root", type = str, default=samsyn_cfg.test_dataset_path)
parser.add_argument('--image_size', type=int, default=samsyn_cfg.image_size)
parser.add_argument('--slice_length', type=int, default=samsyn_cfg.interval_thickness)
parser.add_argument('--mode', type = str, default='test')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_intervals', type=int, default=samsyn_cfg.num_intervals)
parser.add_argument('--num_objs', type=int, default=1)
#parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=1)
#load model
parser.add_argument("--model_type", type = str, default='sam2')
parser.add_argument("--model_cfg", type = str, default=samsyn_cfg.model_cfg_path)
parser.add_argument("--sam_med2_ckpt", type = str, default=samsyn_cfg.sam2_checkpoint_path)
# train
parser.add_argument('--pretrain_path', type=str, default=samsyn_cfg.test_checkpoint_path)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_epochs', type=int, default=samsyn_cfg.num_epochs)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0,1,2,3])
#parser.add_argument('--multi_gpu', action='store_true', default=True)
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--lr_scheduler', type=str, default='cosinelr', help='multisteplr, cosinelr')
parser.add_argument('--step_size', type=list, default=[20, 35, 60]) 
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=samsyn_cfg.lr)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--port', type=int, default=11365)
#parser.add_argument('--dist', dest='dist', type=bool, default=True, help='distributed training or not')
parser.add_argument('--dist', dest='dist', type=bool, default=False, help='distributed training or not')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])

device = args.device
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def save_tensors_as_jpg(tensor1, tensor2, name1, name2, save_dir="./debug_images"):
    """
    将两个形状为 [B, 1, H, W] 的 Tensor 按批次依次保存为 JPG 图像。
    
    参数:
        tensor1: 第一个张量, 形状如 [8, 1, 1024, 1024], 值域应在 [0, 1]
        tensor2: 第二个张量, 形状如 [8, 1, 1024, 1024], 值域应在 [0, 1]
        name1: tensor1 生成文件的前缀名
        name2: tensor2 生成文件的前缀名
        save_dir: 图像保存的目录路径
    """
    # 1. 确保输出目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. 剥离计算图，转移到 CPU，并转为 Numpy 数组
    # (如果输入已经是 numpy array，这段代码依然能兼容)
    if isinstance(tensor1, torch.Tensor):
        arr1 = tensor1.detach().cpu().numpy()
        arr2 = tensor2.detach().cpu().numpy()
    else:
        arr1, arr2 = tensor1, tensor2
        
    batch_size = arr1.shape[0]
    
    for i in range(batch_size):
        # 3. 取出单张切片并去掉 Channel 维度: [1, 1024, 1024] -> [1024, 1024]
        img1_np = np.squeeze(arr1[i], axis=0)
        img2_np = np.squeeze(arr2[i], axis=0)
        
        # 4. 强行截断在 [0, 1] 之间（防止有微小的越界），并缩放到 [0, 255]
        img1_uint8 = (np.clip(img1_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        img2_uint8 = (np.clip(img2_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        
        # 5. 拼凑文件名，使用 :02d 保证两位数对齐 (例如 00, 01, 02)
        path1 = os.path.join(save_dir, f"{name1}_{i:02d}.jpg")
        path2 = os.path.join(save_dir, f"{name2}_{i:02d}.jpg")
        
        # 6. 转为 PIL Image (mode='L' 代表 8-bit 灰度图) 并保存
        Image.fromarray(img1_uint8, mode='L').save(path1)
        Image.fromarray(img2_uint8, mode='L').save(path2)
        
    print(f"✅ 成功将 {batch_size} 对对比图保存至目录: {os.path.abspath(save_dir)}")

def tensor_to_pet_nifti(pred_tensor, reference_nii_path, inv_meta, save_path, output_suv=True):
        """
        将高分辨率 [0, 1] Tensor 下采样并逆变换为真实的 3D PET NIfTI 图像。
        
        参数:
            pred_tensor: 模型输出的 tensor，形状类似 [8, 1, 1024, 1024]
            reference_nii_path: 真实的 PET NIfTI 参考文件路径
            inv_meta: 预处理时生成的字典
            save_path: NIfTI 保存路径
            output_suv: 是否保留为真实的 SUV 物理值
        """
        # 1. 加载参考模板，获取真实的物理空间信息和形状
        ref_sitk = sitk.ReadImage(reference_nii_path)
        # SimpleITK 的 GetSize() 返回的是 (X, Y, Z)
        ref_size_x, ref_size_y, ref_size_z = ref_sitk.GetSize() 
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(save_path)
        # 2. 动态下采样 (Downsample)
        # 确保 tensor 在进行插值时是 float 类型
        pred_tensor = pred_tensor.float()
        
        # 你的 pred_tensor 是 [8, 1, 1024, 1024] = [Batch(Z), Channel, Height(Y), Width(X)]
        # 我们使用双线性插值 (bilinear)，将其平滑缩放回真实的分辨率
        if pred_tensor.shape[-2:] != (ref_size_y, ref_size_x):
            pred_tensor = F.interpolate(
                pred_tensor, 
                size=(ref_size_y, ref_size_x), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 3. 剥离 Tensor 到 Numpy，并挤压掉多余的 Channel 维度
        img_array = pred_tensor.detach().cpu().numpy()
        img_array = np.squeeze(img_array, axis=1) # 变成 [8, Y_真实, X_真实]

        # --- 后续完全是之前推导的数学逆变换 ---
        # log_min = inv_meta["log_min"]
        # log_max = inv_meta["log_max"]
        # suv_factor = inv_meta["suv_factor"]

        log_min = 0.0
        log_max = 2.772588722239781
        suv_factor = 0.0006631289109771332

        # 逆归一化 -> 逆对数
        img_array = img_array * (log_max - log_min) + log_min
        img_array = np.expm1(img_array)

        if not output_suv:
            img_array = img_array / suv_factor
            
        img_array = img_array.astype(np.float32)

        # 4. 把 Numpy 数组转回 SimpleITK 对象
        pred_sitk = sitk.GetImageFromArray(img_array)

        # 5. 🌟 灵魂一步：完美拷贝所有的物理信息 (Origin, Spacing, Direction)
        # 此时因为 pred_sitk 的 Shape 已经通过 interpolate 变得和 ref_sitk 完全一致
        # 这一步拷贝将会天衣无缝！
        #pred_sitk.CopyInformation(ref_sitk)

        # 6. 落盘保存
        sitk.WriteImage(pred_sitk, save_path)
        print(f"✅ 生成的 PET NIfTI 已下采样至 {ref_size_x}x{ref_size_y} 并保存至: {save_path}")

def save_dict_to_disk(data_dict, save_path):
    """
    将包含 Tensor 和其他基础类型（如字符串）的字典安全地保存到本地磁盘。
    
    参数:
        data_dict (dict): 要保存的字典，例如 {"data": tensor, "case_name": "patient_01"}
        save_path (str): 保存的文件路径，建议以 .pt 或 .pth 结尾
    """
    # 1. 确保父目录存在
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # 2. 遍历字典，安全处理里面的每一个元素
    safe_dict = {}
    for key, value in data_dict.items():
        # 如果这个值是 Tensor，就剥离计算图并移到 CPU
        if isinstance(value, torch.Tensor):
            safe_dict[key] = value.detach().cpu()
        # 如果是其他类型（字符串、数字、列表等），直接原样保留
        else:
            safe_dict[key] = value
            
    # 3. 直接保存整个安全字典
    torch.save(safe_dict, save_path)
    
    # 打印一些友好的提示信息
    keys_list = list(safe_dict.keys())
    print(f"✅ 字典已成功保存至: {save_path}")
    print(f"   包含的键: {keys_list}")


class BaseTester:
    def __init__(self, model, test_dataloaders, args):
        self.model = model
        self.test_dataloaders = test_dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_L1 = 0.0
        self.best_ssim = 0.0
        self.loss = []
        self.L1 = []
        self.ssim = []
        self.set_loss_fn()
        self.name_mapping_dict = utils.read_json_to_dict(samsyn_cfg.studyId_to_nii_idx_json)
        self.pipepine_info_dict = utils.read_json_to_dict(samsyn_cfg.pet_pipline_info)

        self.model = self.model.module if self.args.multi_gpu else self.model

        if args.pretrain_path is not None:
            self.load_checkpoint(args.pretrain_path, args.resume)
            print("!!!!!!!!!!!!!!!! Loading trained PT !!!!!!!!!!!!!!!!")
        else:
            self.start_epoch = 0

    def set_loss_fn(self):
        l_l1 = getattr(self.args, 'lambda_l1', 10.0)
        l_ssim = getattr(self.args, 'lambda_ssim', 5.0)
        l_lesion = getattr(self.args, 'lambda_lesion', 20.0)
        print(f"🔧 初始化 Loss: L1({l_l1}), SSIM({l_ssim}), Lesion({l_lesion})")
        
        self.criterion = PETSynthesisLoss(
            lambda_l1=l_l1,
            lambda_ssim=l_ssim,
            lambda_lesion=l_lesion,
            data_range=1.0 
        ).to(device)

    def load_checkpoint(self, ckp_path, resume):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
            if 'step_' in ckp_path:
                self.start_step = int(ckp_path.split('step_')[-1].split('.pth')[0])
            else:
                self.start_step = 0

            last_ckpt = torch.load(ckp_path, map_location=self.args.device, weights_only=False)

        if last_ckpt:
            try:
                self.model.load_state_dict(last_ckpt['model_state_dict'])
            except Exception as e:
                print(f"Failed to load model state dict: {e}")
                self.model.load_state_dict(last_ckpt['model_state_dict'], False)

            if resume:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                if self.lr_scheduler and 'lr_scheduler_state_dict' in last_ckpt:
                    self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.loss = last_ckpt['loss']
                self.L1 = last_ckpt['L1']
                self.ssim = last_ckpt['ssim']
                self.best_loss = last_ckpt['best_loss']
                #self.best_dice = last_ckpt['best_dice']
            else:
                self.start_epoch = 0
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch}, step: {self.start_step})")
            
        else:
            self.start_epoch, self.start_step = 0, 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")
    
    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "loss": self.loss,
            "L1": self.L1,
            "ssim": self.ssim,
            "best_loss": self.best_loss,
            "best_L1": self.best_L1,
            "best_ssim": self.best_ssim,
            "args": self.args,
        }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))


    def test_epoch(self, epoch):
        self.model.eval()
        l = len(self.test_dataloaders)
        tbar = tqdm(self.test_dataloaders, desc=f'Epoch {epoch+1} / {self.args.num_epochs}')
        epoch_loss, epoch_L1, epoch_ssim, epoch_focal, epoch_dice, epoch_lesion = 0, 0, 0, 0, 0, 0
        for step, batch_input in enumerate(tbar): 
            print(f"@@@@@  test step {step} @@@@@")
            batch_loss, batch_L1, batch_ssim, batch_focal, batch_dice, batch_lesion = [], [], [], [], [], []
            data_intervals_list = batch_input["data_intervals_list"]
            prompts_coords_list = batch_input["prompts_coords_list"]
            prompts_objs_list = batch_input["prompts_objs_list"]
            ground_truth_list = batch_input["ground_truth_list"]
            conditioned_frame_idx_list = batch_input["conditioned_frame_idx_list"]
            case_name_list = batch_input["case_name_list"]
            interval_seg_list = batch_input["interval_seg_list"]
            segs_are_full_list = batch_input["segs_are_full_list"]

            interval_idx = random.randrange(len(data_intervals_list)) # randmly pick up an interval idx
            
            curent_data_interval = data_intervals_list[interval_idx].to(device)
            current_prompts_coords = prompts_coords_list[interval_idx]
            current_prompts_obj_classes = prompts_objs_list[interval_idx]
            current_gt = ground_truth_list[interval_idx].to(device)
            conditioned_frame_offset_in_nii = conditioned_frame_idx_list[interval_idx]
            curent_interval_thinckness = curent_data_interval.shape[0]
            #env_info = utils.read_json_to_dict("samsyn_json_metadata/pet_inv_meta.json")
            current_conditioned_frame_idx = 0 # this is reletive idx in a small interval. The above one is absolute idx in an NII file
            current_case_name = case_name_list[interval_idx]
            current_segs_are_full = segs_are_full_list[interval_idx]
            if current_segs_are_full:
                current_interval_seg = interval_seg_list[interval_idx].to(device)
            else:
                current_interval_seg = None
            obj_id = 1 # hardcode here!!!!!!!!!! Will be modified
            predict_labels = {}
            train_state = self.model.train_init_state(curent_data_interval)
            
            with torch.no_grad():
                
                _, _, conditioned_out_mask_logits = self.model.train_add_new_points_or_box(  
                                inference_state=train_state, frame_idx=current_conditioned_frame_idx, obj_id=obj_id,  
                                points=current_prompts_coords, labels=current_prompts_obj_classes, clear_old_points=False  
                )   
                start_slice = current_conditioned_frame_idx
                    
                for out_frame_idx, out_obj_ids, out_mask_logits in self.model.train_propagate_in_video(  
                train_state, start_frame_idx=start_slice, reverse=False):  
                    # out_mask_logits type is tensor, and the shape is [1,1,1024,1024]
                    predict_labels[out_frame_idx] = out_mask_logits 
                
                gt3d = current_gt[:, 0:1, :, :] # original gt is [8, 3, 1024, 1024] 
                predict_labels = list(predict_labels.values())
                predict3d = torch.cat(predict_labels, dim=0)
                predict3d = torch.sigmoid(predict3d)
                fname = samsyn_cfg.test_results_path + str(current_case_name) + "_tensor.pth"
                key = str(current_case_name) + ".nii.gz"
                studyID = self.name_mapping_dict[key]
                suv = self.pipepine_info_dict[studyID]["suv_factor"]
                log_min = self.pipepine_info_dict[studyID]["log_min"]
                log_max = self.pipepine_info_dict[studyID]["log_max"]
                d = {"case_name": current_case_name,
                     "slice_offset": conditioned_frame_offset_in_nii,
                     "thickness": curent_interval_thinckness,
                     "tensor": predict3d,
                     "studyID": studyID,
                     "suv": suv,
                     "min": log_min,
                     "max": log_max
                     }
                save_dict_to_disk(d, fname)
                  
                #total_loss, L1, ssim, lesion, focal, dice  = self.criterion(predict3d, gt3d, lesion_mask=current_interval_seg)
                total_loss, L1, ssim, lesion  = self.criterion(predict3d, gt3d, lesion_mask=current_interval_seg)
     
                self.model.reset_state(train_state)  

                batch_loss.append(total_loss.item())  
                batch_L1.append(L1.item())  
                batch_ssim.append(ssim.item())   
                # batch_focal.append(focal.item()) 
                # batch_dice.append(dice.item()) 
                batch_lesion.append(lesion.item()) 

            epoch_loss += np.mean(batch_loss)
            epoch_L1 += np.mean(batch_L1)
            epoch_ssim += np.mean(batch_ssim)
            # epoch_focal += np.mean(batch_focal)
            # epoch_dice += np.mean(batch_dice)
            epoch_lesion += np.mean(batch_lesion)

        #avg_loss, avg_L1, avg_ssim, avg_focal, avg_dice, avg_lesion = epoch_loss / l, epoch_L1 / l, epoch_ssim / l, epoch_focal / l, epoch_dice / l, epoch_lesion / l
        avg_loss, avg_L1, avg_ssim, avg_lesion = epoch_loss / l, epoch_L1 / l, epoch_ssim / l, epoch_lesion / l
        
        #return avg_loss, avg_L1, avg_ssim, avg_focal, avg_dice, avg_lesion, current_case_name
        return avg_loss, avg_L1, avg_ssim, avg_lesion, current_case_name


    def test(self):
        self.scaler = amp.GradScaler()
        
        torch.cuda.empty_cache()

        #test_loss, test_L1, test_ssim, test_focal, test_dice, test_lesion, current_case_name = self.test_epoch(1)
        test_loss, test_L1, test_ssim, test_lesion, current_case_name = self.test_epoch(1)
        print("VAL END...")
        print("===== Test result: =====")
        #print(f"case name: {current_case_name}, Loss: {test_loss}, L1: {test_L1}, SIMM: {test_ssim}, focal: {test_focal}, dice: {test_dice}, lesion: {test_lesion}")
        print(f"case name: {current_case_name}, Loss: {test_loss}, L1: {test_L1}, SIMM: {test_ssim}, lesion: {test_lesion}")

            
      


########################################## Trainer ##########################################
def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            # args.multi_gpu = False
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            # args.multi_gpu = True
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node
    except RuntimeError as e:
        print(e)

def main():
    for key, value in vars(args).items():
        print(key + ': ' + str(value))

    mp.set_sharing_strategy('file_system')
    device_config(args)

    logger = get_logger(args)
    args.logger = logger
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    # Load datasets
    test_dataloaders = get_dataset_3d(args)
    
    # Build model
    model = build_model(args)
    # Create trainer
    tester = BaseTester(model, test_dataloaders, args)
    # Train
    tester.test()


def setup(rank, world_size):
    # initialize the process group
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = f'{args.port}'
    dist.init_process_group(backend='NCCL', init_method='env://', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
