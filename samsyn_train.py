# set up environment
import numpy as np
import random 
import matplotlib.pyplot as plt
import os
join = os.path.join
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
from samsyn_dataloders.dataloader import get_dataset_3d
from samsyn_train_utils import DiceLoss, build_model, get_logger
import nibabel as nib  
import cv2
from torch.nn import CrossEntropyLoss
import samsyn_cfg

import warnings
warnings.filterwarnings("ignore") 
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default='work_dir')
parser.add_argument('--task_name', type=str, default='sam2_Synthesizing')
#load data
#parser.add_argument("--data_root", type = str, default='datasets/BraTS2020/FLAIR')
parser.add_argument("--data_root", type = str, default=samsyn_cfg.dataset_path)
parser.add_argument('--image_size', type=int, default=samsyn_cfg.image_size)
parser.add_argument('--slice_length', type=int, default=samsyn_cfg.interval_thickness)
parser.add_argument('--mode', type = str, default='training')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_intervals', type=int, default=samsyn_cfg.num_intervals)
parser.add_argument('--num_objs', type=int, default=1)
#parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--lasion_ct_pix_json', type=str, default=samsyn_cfg.lasions_coords_info_json) # this file specifies the lasions coords in traning data.
parser.add_argument('--rename_json', type=str, default=samsyn_cfg.studyId_to_nii_idx_json) # this file specifies the mapping between study id and nii idx.
#load model
parser.add_argument("--model_type", type = str, default='sam2')
parser.add_argument("--model_cfg", type = str, default=samsyn_cfg.model_cfg_path)
parser.add_argument("--sam_med2_ckpt", type = str, default=samsyn_cfg.sam2_checkpoint_path)
#parser.add_argument("--sam_med2_ckpt", type = str, default='checkpoints/sam-med2d_b.pth')
# train
parser.add_argument('--pretrain_path', type=str, default=None)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0,1,2,3])
#parser.add_argument('--multi_gpu', action='store_true', default=True)
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--lr_scheduler', type=str, default='cosinelr', help='multisteplr, cosinelr')
parser.add_argument('--step_size', type=list, default=[20, 35, 60]) 
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--port', type=int, default=11365)
#parser.add_argument('--dist', dest='dist', type=bool, default=True, help='distributed training or not')
parser.add_argument('--dist', dest='dist', type=bool, default=False, help='distributed training or not')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])

device = args.device
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


def cosine_lr_factor(total_steps, warmup_steps):
    def _lr_adjuster(step):
        if step < warmup_steps:
            lr_factor = 1.0  # Keep the learning rate fixed during warmup
        else:
            e = step - warmup_steps
            es = total_steps - warmup_steps
            lr_factor = 0.5 * (1 + np.cos(np.pi * e / es))  # Cosine annealing factor
        return lr_factor
    return _lr_adjuster


class BaseTrainer:
    def __init__(self, model, dataloaders, test_dataloaders, args):
        self.model = model
        self.dataloaders = dataloaders
        self.test_dataloaders = test_dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.best_iou = 0.0
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()

        self.model = self.model.module if self.args.multi_gpu else self.model

        if args.pretrain_path is not None:
            self.load_checkpoint(args.pretrain_path, args.resume)
        else:
            self.start_epoch = 0

    def set_loss_fn(self):
        self.seg_loss = DiceLoss()
        self.ce_loss = CrossEntropyLoss()

    def set_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay) # 

    def set_lr_scheduler(self):
        # Ensure self.dataloaders refers to the training DataLoader
        train_loader = self.dataloaders  # Adjust if self.dataloaders is a dict, e.g., self.dataloaders['train']
        steps_per_epoch = len(train_loader) * self.args.num_intervals
        total_steps = steps_per_epoch * self.args.num_epochs
        
        if self.args.lr_scheduler.lower() == "multisteplr":
            milestones_epochs = self.args.step_size  # e.g., [30, 60, 90]
            milestones_steps = [epoch * steps_per_epoch for epoch in milestones_epochs]
            
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones_steps,
                gamma=self.args.gamma
            )
            print(f"Using MultiStepLR scheduler with milestones at steps: {milestones_steps} and gamma: {self.args.gamma:.8f}")
        
        elif self.args.lr_scheduler.lower() == "cosinelr":
            warmup_steps = steps_per_epoch*5  # Example: 5 epochs of warmup
            if warmup_steps >= total_steps:
                raise ValueError("Warmup steps must be less than total training steps.")
            # Initialize LambdaLR with cosine annealing factor
            cosine_factor = cosine_lr_factor(total_steps, warmup_steps)
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=cosine_factor
            )
            print(f"Using CosineLR scheduler with total steps: {total_steps} and warmup steps: {warmup_steps}")
        else:
            self.lr_scheduler = None
            print("No learning rate scheduler is being used.")

    def update_learning_rate(self, current_step):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            # Assuming a single parameter group
            self.current_lr = self.lr_scheduler.get_last_lr()[0]
            # if self.args.rank == 0:
            #     print(f"Steps: {current_step} Learning rate updated to: {self.current_lr}")
        else:
            self.current_lr = self.args.lr
                # print(f"Using fixed learning rate: {self.current_lr}")


    def load_checkpoint(self, ckp_path, resume):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
            if 'step_' in ckp_path:
                self.start_step = int(ckp_path.split('step_')[-1].split('.pth')[0])
            else:
                self.start_step = 0

            last_ckpt = torch.load(ckp_path, map_location=self.args.device)

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
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                self.ious = last_ckpt['ious']
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']
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
            "losses": self.losses,
            "ious": self.ious,
            "dices": self.dices,
            "best_loss": self.best_loss,
            "best_iou": self.best_iou,
            "best_dice": self.best_dice,
            "args": self.args,
        }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))


    def get_iou_and_dice(self, pred, label):
        assert pred.shape == label.shape
        pred = (torch.sigmoid(pred) > 0.5)
        label = (label > 0)
        intersection = torch.logical_and(pred, label).sum(dim=(1,2,3)) 
        union = torch.logical_or(pred, label).sum(dim=(1,2,3))  
        iou = intersection.float() / (union.float() + 1e-8) 
        dice = (2 * intersection.float()) / (pred.sum(dim=(1,2,3)) + label.sum(dim=(1,2,3)) + 1e-8) 
        return iou.mean().item(), dice.mean().item()


    def plot_result(self, plot_data, description, save_name):
        # plot_data 应该是一个包含训练和验证损失的字典，格式如: {'train': [loss_values], 'val': [loss_values]}
        # 提取训练和验证损失
        train_data = [data['train'] for data in plot_data]
        val_data = [data['val'] for data in plot_data]
        # 绘制训练损失和验证损失的线
        plt.plot(train_data, label='Train')
        plt.plot(val_data, label='Validation')
        # 设置图表的标题和标签
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        # 添加图例
        plt.legend()
        # 保存图像
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()

    def extract_features(self, train_state):
        backbone_fpns = dict(sorted(train_state['backbone_fpns'].items()))
        upscaled_embeds = dict(sorted(train_state['upscaled_embeds'].items()))
        low_res_masks = dict(sorted(train_state['low_res_masks'].items()))

        low_res_masks = torch.sigmoid(torch.cat([v for v in low_res_masks.values()], dim=0)).permute(1, 0, 2, 3).unsqueeze(0)
        
        expanded_backbone_fpns = {
            key: torch.cat([v[i] for v in backbone_fpns.values()], dim=0)
                    .permute(1, 0, 2, 3)
                    .unsqueeze(0)
            for i, key in enumerate(['down_x4', 'down_x8', 'down_x16'])
        }
        expanded_upscaled_embeds = {
            up_scale: torch.cat(
                [embed[up_scale] for embed in upscaled_embeds.values()],
                dim=0
            ).permute(1, 0, 2, 3).unsqueeze(0)
            for up_scale in ['upscaled_x16', 'upscaled_x8', 'upscaled_x4']
        }

        return {
            'encoder_feats': expanded_backbone_fpns,
            'decoder_feats': expanded_upscaled_embeds,
            'low_res_masks': low_res_masks
        }


    def train_epoch(self, epoch):
        self.model.train()
        l = len(self.dataloaders)
        
        tbar = tqdm(self.dataloaders, desc=f'Epoch {epoch+1} / {self.args.num_epochs}')
        epoch_loss, epoch_iou, epoch_dice = 0, 0, 0
        
        for step, batch_input in enumerate(tbar): 
            
            batch_loss, batch_iou, batch_dice = [], [], []
            obj_to_class = batch_input["obj_to_class"]
            interval_prompts = batch_input["pre_interval_obj_prompt"]
            interval_labels = batch_input["pre_interval_obj_label"]
            interval_images_flair, interval_images_t1 = batch_input["pre_interval_image_flair"], batch_input["pre_interval_image_t1"]
            
            for interval in range(interval_images_flair.shape[0]):
                print(f"interval = {interval}")
                current_step = epoch * l + step * self.args.num_intervals + interval
                images_f = interval_images_flair[interval]
                images_t = interval_images_t1[interval]
                images = torch.cat([images_f, images_t], dim=1).to(device) 
                labels = interval_labels[interval]
                prompts = interval_prompts[interval]
                train_state = self.model.train_init_state(images)
                
                obj_list = list(labels.keys())
                obj_segments = {}
                prompt_labels = []
                output_dict = {obj_id: [] for obj_id in obj_list} 
                
                for obj_id, obj_data in prompts.items():
                    print(f"~~~ {obj_id}:{list(obj_data.keys())} ~~~")
                    obj_label = labels[obj_id].to(device).type(torch.long) 
                    if random.random() > 0.5:  
                        for slice_idx, points in obj_data["point_coords"].items():
                            print(f"@slice_idx={slice_idx}")  
                            point_labels = prompts[obj_id]["point_labels"][slice_idx]
                            _, _, out_mask_logits = self.model.train_add_new_points_or_box(  
                                inference_state=train_state, frame_idx=slice_idx, obj_id=obj_id,  
                                points=points, labels=point_labels, clear_old_points=False  
                            )  
                            prompt_labels.append(obj_label[slice_idx])
                    else:  
                        for slice_idx, bbox in obj_data["bboxes"].items():  
                            print(f"#slice_idx={slice_idx}")
                            _, _, out_mask_logits = self.model.train_add_new_points_or_box(  
                                inference_state=train_state, frame_idx=slice_idx, obj_id=obj_id,  
                                points=None, labels=None, box=bbox, clear_old_points=True  
                            )  
                            prompt_labels.append(obj_label[slice_idx]) 
                
                
                prompt_label = torch.stack(prompt_labels, dim=0)  
                prompt_loss = self.seg_loss(out_mask_logits, prompt_label.unsqueeze(1))  
                prompt_iou, prompt_dice = self.get_iou_and_dice(out_mask_logits, prompt_label.unsqueeze(1))  
                # print(f'Prompt metrics, IoU: {prompt_iou:.4f}, Dice: {prompt_dice:.4f}')
                
                if prompt_dice < 0.45:  #0.6
                    print("yes, <")
                    self.optimizer.zero_grad()  
                    self.scaler.scale(prompt_loss).backward()  
                    self.scaler.step(self.optimizer)  
                    self.scaler.update()  
                    self.model.reset_state(train_state) 

                    batch_loss.append(prompt_loss.item())  
                    batch_iou.append(prompt_iou)  
                    batch_dice.append(prompt_dice)  
                    self.update_learning_rate(current_step)
                    continue
                else:
                    print("No, >")

                start_slice = min(slice_idx for slice_idx in prompts[next(iter(prompts))]["point_coords"].keys())
                
                for direction in [False, True]:  # 正向和反向  
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.model.train_propagate_in_video(  
                        train_state, start_frame_idx=start_slice, reverse=direction):  
                        obj_segments[out_frame_idx] = {out_obj_id: out_mask_logits[i] for i, out_obj_id in enumerate(out_obj_ids)}  
                
                for out_frame_idx in range(images.shape[0]):  
                    for out_obj_id, out_mask in obj_segments[out_frame_idx].items():  
                        output_dict[out_obj_id].append(out_mask)  
                
                enhance_mask = self.model.head_3d(self.extract_features(train_state), mode='training')
                
                outputs_ = [torch.cat(masks, dim=0) for masks in output_dict.values()]  
                mask = torch.stack(outputs_, dim=0)  

                labels_ = [labels[obj_id].to(device).type(torch.long) for obj_id in output_dict.keys()]  
                label = torch.stack(labels_, dim=0)  
                total_loss = self.seg_loss(enhance_mask, label) + prompt_loss  
                self.optimizer.zero_grad()  
                self.scaler.scale(total_loss).backward()  
                self.scaler.step(self.optimizer)  
                self.scaler.update()  
                
                self.model.reset_state(train_state)  
                iou, dice = self.get_iou_and_dice(enhance_mask, label) 
                batch_loss.append(total_loss.item())  
                batch_iou.append(iou)  
                batch_dice.append(dice)  

                self.update_learning_rate(current_step)
                
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if (step+1) % 50 == 0:
                    self.args.logger.info(f'Epoch: {epoch+1}, Step: {step+1}, lr: {self.current_lr:.8f}, loss: {np.mean(batch_loss):.4f}, iou: {np.mean(batch_iou):.4f}, dice: {np.mean(batch_dice):.4f}')
                    state_dict = self.model.state_dict()
                    self.save_checkpoint(epoch, state_dict, describe='setp')

            epoch_loss += np.mean(batch_loss)
            epoch_iou += np.mean(batch_iou)
            epoch_dice += np.mean(batch_dice)

        if self.args.multi_gpu:
            dist.barrier()
            local_loss = torch.tensor([epoch_loss / l]).to(self.args.device)
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM) 
            avg_loss = local_loss.item() / dist.get_world_size()

            local_iou = torch.tensor([epoch_iou / l]).to(self.args.device)
            dist.all_reduce(local_iou, op=dist.ReduceOp.SUM) 
            avg_iou = local_iou.item() / dist.get_world_size()

            local_dice = torch.tensor([epoch_dice / l]).to(self.args.device)
            dist.all_reduce(local_dice, op=dist.ReduceOp.SUM) 
            avg_dice = local_dice.item() / dist.get_world_size()

        # if self.args.multi_gpu and self.args.rank == 0:
        #     avg_loss, avg_iou, avg_dice = epoch_loss / l, epoch_iou / l, epoch_dice / l
        else:
            avg_loss, avg_iou, avg_dice = epoch_loss / l, epoch_iou / l, epoch_dice / l
        
        return avg_loss, avg_iou, avg_dice


    def test_epoch(self, epoch):
        self.model.eval()
        l = len(self.test_dataloaders)
        tbar = tqdm(self.test_dataloaders, desc=f'Epoch {epoch+1} / {self.args.num_epochs}')
        epoch_loss, epoch_iou, epoch_dice = 0, 0, 0
        for step, batch_input in enumerate(tbar): 
            #print(f"@@@@@  step {step} @@@@@")
            batch_loss, batch_iou, batch_dice = [], [], []
            prompts = batch_input["pre_interval_obj_prompt"][0]
            labels = batch_input["pre_interval_obj_label"][0]
            interval_images_flair, interval_images_t1 = batch_input["pre_interval_image_flair"][0], batch_input["pre_interval_image_t1"][0]
            images = torch.cat([interval_images_flair, interval_images_t1], dim=1).to(device) 

            obj_list = list(labels.keys())
            obj_segments = {}
            prompt_labels = []
            output_dict = {obj_id: [] for obj_id in obj_list} 

            train_state = self.model.train_init_state(images)
           
            with torch.no_grad():
                for obj_id, obj_data in prompts.items():
                    obj_label = labels[obj_id].to(device).type(torch.long)  
                    if random.random() > 0.5:  
                        for slice_idx, points in obj_data["point_coords"].items():  
                            point_labels = prompts[obj_id]["point_labels"][slice_idx]
                            _, _, out_mask_logits = self.model.train_add_new_points_or_box(  
                                inference_state=train_state, frame_idx=slice_idx, obj_id=obj_id,  
                                points=points, labels=point_labels, clear_old_points=False  
                            )  
                            prompt_labels.append(obj_label[slice_idx])
                    else:  
                        for slice_idx, bbox in obj_data["bboxes"].items():  
                            _, _, out_mask_logits = self.model.train_add_new_points_or_box(  
                                inference_state=train_state, frame_idx=slice_idx, obj_id=obj_id,  
                                points=None, labels=None, box=bbox, clear_old_points=True  
                            )  
                            prompt_labels.append(obj_label[slice_idx]) 

                    prompt_label = torch.stack(prompt_labels, dim=0)  
                    prompt_loss = self.seg_loss(out_mask_logits, prompt_label.unsqueeze(1))  

                    start_slice = min(slice_idx for slice_idx in prompts[next(iter(prompts))]["point_coords"].keys())
                    for direction in [False, True]:  # 正向和反向  
                        for out_frame_idx, out_obj_ids, out_mask_logits in self.model.train_propagate_in_video(  
                            train_state, start_frame_idx=start_slice, reverse=direction):  
                            obj_segments[out_frame_idx] = {out_obj_id: out_mask_logits[i] for i, out_obj_id in enumerate(out_obj_ids)}  

                    for out_frame_idx in range(images.shape[0]):  
                        for out_obj_id, out_mask in obj_segments[out_frame_idx].items():  
                            output_dict[out_obj_id].append(out_mask)  

                
                enhance_mask = self.model.head_3d(self.extract_features(train_state), mode='test')
                outputs_ = [torch.cat(masks, dim=0) for masks in output_dict.values()]  
                labels_ = [labels[obj_id].to(device).type(torch.long) for obj_id in output_dict.keys()]  
                mask = torch.stack(outputs_, dim=0)  
                label = torch.stack(labels_, dim=0)  
                total_loss = self.seg_loss(enhance_mask, label) + prompt_loss  
     
                self.model.reset_state(train_state)  

                iou, dice = self.get_iou_and_dice(enhance_mask, label) 
                batch_loss.append(total_loss.item())  
                batch_iou.append(iou)  
                batch_dice.append(dice)  
                
            epoch_loss += np.mean(batch_loss)
            epoch_iou += np.mean(batch_iou)
            epoch_dice += np.mean(batch_dice)

        if self.args.multi_gpu:
            dist.barrier()
            local_loss = torch.tensor([epoch_loss / l]).to(self.args.device)
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM) 
            avg_loss = local_loss.item() / dist.get_world_size()

            local_iou = torch.tensor([epoch_iou / l]).to(self.args.device)
            dist.all_reduce(local_iou, op=dist.ReduceOp.SUM) 
            avg_iou = local_iou.item() / dist.get_world_size()

            local_dice = torch.tensor([epoch_dice / l]).to(self.args.device)
            dist.all_reduce(local_dice, op=dist.ReduceOp.SUM) 
            avg_dice = local_dice.item() / dist.get_world_size()

        else:
            avg_loss, avg_iou, avg_dice = epoch_loss / l, epoch_iou / l, epoch_dice / l
        
        return avg_loss, avg_iou, avg_dice


    def train(self):
        self.scaler = amp.GradScaler()
        best_dice_epoch = 1
        for epoch in range(self.start_epoch, self.args.num_epochs):
            
            torch.cuda.empty_cache()
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                print(f'Epoch: {epoch+1}/{self.args.num_epochs}')

            if self.args.multi_gpu:
                # dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)
            print("TRAIN START...")
            avg_loss, avg_iou, avg_dice = self.train_epoch(epoch)
            print("TRAIN END...")
            test_loss, test_iou, test_dice = self.test_epoch(epoch)
            print("VAL END...")
            
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append({'train':avg_loss, 'val': test_loss})
                self.ious.append({'train':avg_iou, 'val': test_iou})
                self.dices.append({'train':avg_dice, 'val': test_dice})

                self.args.logger.info(f'Epoch: {epoch+1} LR: {self.current_lr:.8f}: Train loss: {avg_loss:.5f}, IoU: {avg_iou:.5f}, Dice: {avg_dice:.5f} | Test loss: {test_loss:.5f}, IoU: {test_iou:.5f}, Dice: {test_dice:.5f}')
                state_dict = self.model.state_dict()
                 #save latest checkpoint
                self.save_checkpoint(epoch, state_dict, describe='latest')
                
                #save train loss best checkpoint
                if test_loss < self.best_loss: 
                    self.best_loss = test_loss
                    # self.save_checkpoint(epoch, state_dict, describe='loss_best')
                
                if test_iou > self.best_iou: 
                    self.best_iou = test_iou
                    # self.save_checkpoint(epoch, state_dict, describe='iou_best')

                # save train dice best checkpoint
                if test_dice > self.best_dice: 
                    best_dice_epoch = epoch
                    self.best_dice = test_dice
                    self.save_checkpoint(epoch, state_dict, describe='dice_best')

                self.plot_result(self.losses, 'Loss', 'Loss')
                self.plot_result(self.dices, 'Dice', 'Dice')
                self.plot_result(self.ious, 'IoU', 'IoU')
      
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            self.args.logger.info('=====================================================================')
            for key, value in vars(self.args).items():
                self.args.logger.info(key + ': ' + str(value))
            self.args.logger.info(f'Best loss: {self.best_loss}, Best iou: {self.best_iou}, Best dice: {self.best_dice}, Best dice epoch: {best_dice_epoch}')
            self.args.logger.info('=====================================================================')


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

    if args.multi_gpu:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args, ))
    else:
        logger = get_logger(args)
        args.logger = logger
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        # Load datasets
        dataloaders = get_dataset_3d(args, mode='training')
        test_dataloaders = get_dataset_3d(args, mode='val')
        # Build model
        model = build_model(args)
        # Create trainer
        trainer = BaseTrainer(model, dataloaders, test_dataloaders, args)
        # Train
        trainer.train()

def main_worker(rank, args):
    setup(rank, args.world_size)
    torch.cuda.set_device(rank)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank
    args.gpu_info = {"gpu_count":args.world_size, 'gpu_name':rank}
    init_seeds(2024 + rank)
    if rank == 0:
        logger = get_logger(args)
        args.logger = logger
    dataloaders = get_dataset_3d(args, mode='training')
    test_dataloaders = get_dataset_3d(args, mode='val')
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, test_dataloaders, args)
    trainer.train()
    cleanup()


def setup(rank, world_size):
    # initialize the process group
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = f'{args.port}'
    dist.init_process_group(backend='NCCL', init_method='env://', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
