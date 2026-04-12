sam2_checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
model_cfg_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
con_frame_idx = 0  # the frame index we interact with
obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

dataset_path = "samsyn_dataset/data/" # nii.gz files.
labels_path = "samsyn_dataset/labels/" # nii.gz files.
prompts_path = "samsyn_dataset/segs/" # nii.gz files.

lasions_coords_info_json = "samsyn_json_metadata/lesion_ct_pixel_coords.json"
studyId_to_nii_idx_json = "samsyn_json_metadata/rename_mapping.json"
ct_slice_counts_json = "samsyn_json_metadata/dcm_counts.json" # how many slices in a ct nii
nii_idx_with_prompts_coords_json = "samsyn_json_metadata/nii_idx_with_prompts_coords.json"
interval_info = "samsyn_json_metadata/nii_idx_intervals.json"

interval_thickness = 8 # there are 8 frames in a "short video"

points_num = 4 # number of prompts of points

num_intervals = 10

image_size = 1024
