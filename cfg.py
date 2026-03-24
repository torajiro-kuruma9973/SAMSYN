sam2_checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
model_cfg_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
data_dir = "raw_datasets/BrainTumorMRI"
con_frame_idx = 0  # the frame index we interact with
obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
nii_mask_path = "/home/will/Work/CVEnv1/SAM2/sam2-main/notebooks/videos/BraTS20_Training_001_seg.nii"
comman_img_mask_path = "/home/will/Work/CVEnv1/SAM2/sam2-main/notebooks/videos/bird_masks"
