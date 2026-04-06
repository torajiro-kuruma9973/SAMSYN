import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib
from torchvision import transforms
import samsyn_cfg
from samsyn_utils import dcm_utils

# build model

# cuda ONLY
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

else:
    print("ERROR: device type is undefined!!")

from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor(samsyn_cfg.model_cfg_path, samsyn_cfg.sam2_checkpoint_path, device=device)

# pre-process dcm files

if any("IM" in filename for filename in os.listdir(samsyn_cfg.raw_data_dir)):
    dcm_utils.file_name_process(samsyn_cfg.raw_data_dir)
# convert dcm data tp jpg files then store them in a folder specified.
# if the files have existed in the data folder, do nothing. (wont do it again in your test.)
if len(os.listdir(samsyn_cfg.test_data_folder)) == 0:
    dcm_utils.convert_all_dcm_files(samsyn_cfg.raw_data_dir, samsyn_cfg.test_data_folder)

# add prompts
ann_frame_idx = 18
ann_obj_id = 1

# scan all the JPEG frame names in this directory
frame_names = dcm_utils.order_file_names(samsyn_cfg.test_data_folder)
# # take a look the first video frame
# frame_idx = ann_frame_idx
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {frame_idx}")
# plt.imshow(Image.open(frame_names[frame_idx]), cmap='gray')
# plt.show()

# init state
inference_state = predictor.init_state(video_path = samsyn_cfg.test_data_folder)

points = np.array([[128, 212], [145, 249], [222, 286]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 0], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx}")
# #plt.imshow(Image.open(os.path.join(cfg.test_data_folder, frame_names[ann_frame_idx])))
# plt.imshow(Image.open(frame_names[ann_frame_idx]), cmap='gray')
# dcm_utils.show_points(points, labels, plt.gca())
# dcm_utils.show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
# plt.show()

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 1
start_frame = ann_frame_idx
plt.close("all")
for out_frame_idx in range(start_frame, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(frame_names[out_frame_idx]), cmap='gray')

    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        dcm_utils.show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
plt.show()
