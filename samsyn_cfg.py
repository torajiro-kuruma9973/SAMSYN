sam2_checkpoint_path = "checkpoints/sam2.1_hiera_base_plus.pt"
#sam2_checkpoint_path = "checkpoints/sam2.1_hiera_small.pt"
model_cfg_path = "configs/sam2.1/sam2.1_hiera_b+.yaml"
#model_cfg_path = "configs/sam2.1/sam2.1_hiera_s.yaml"
con_frame_idx = 0  # the frame index we interact with
obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

dataset_path = "samsyn_dataset/data/" # nii.gz files.
labels_path = "samsyn_dataset/labels/" # nii.gz files.
prompts_path = "samsyn_dataset/segs" # nii.gz files. "DONT MODIFY THIS PATH!!!!!!!!!!!!"

starter_end_info = "samsyn_json_metadata/new_version_starters_ends_info.json"

#pet_pipline_info = "samsyn_json_metadata/pet_inv_meta.json"

test_starter_end_info = "samsyn_json_metadata/enhanced_orinal_intv_info.json"

interval_thickness = 8 # there are 8 frames in a "short video"

points_num = 4 # number of prompts of points

foreground_rate = 0.5

distant = 6

num_intervals = 300 #just let model use all pics

image_size = 1024

num_epochs = 200

lr = 1e-5

condition_frame_pass_threshold = 2.0

summary_writer_log_path = "summary_writer_logs"

#================ Test ================
test_dataset_path = "samsyn_dataset/test_data/" # nii.gz files.
test_labels_path = "samsyn_dataset/test_labels/" # nii.gz files.
test_prompts_path = "samsyn_dataset/test_segs/" # nii.gz files.

test_checkpoint_path = "checkpoints/new_version_sam_model_loss_best.pth"

studyId_to_nii_idx_json = "samsyn_json_metadata/name_mapping.json"

test_results_path = "test_results/"