import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib
from torchvision import transforms
import cfg


sam2_checkpoint = cfg.sam2_checkpoint_path
model_cfg = cfg.model_cfg_path

video_dir = cfg.data_dir
ann_frame_idx = 0  
ann_obj_id = 1
#nii_mask_path = "/home/will/Work/CVEnv1/SAM2/sam2-main/notebooks/videos/BraTS20_Training_001_seg.nii"
#comman_img_mask_path = "/home/will/Work/CVEnv1/SAM2/sam2-main/notebooks/videos/bird_masks"
last_frame_idx = 0
special = 70
vis_frame_stride = special

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

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
plt.show()
'''
inference_state = predictor.init_state(video_path=video_dir)

#predictor.reset_state(inference_state)

# Let's add a positive click at (x, y) = (210, 350) to get started
# points = np.array([[107, 93]], dtype=np.float32)
# # for labels, `1` means positive click and `0` means negative click
# labels = np.array([1], np.int32)
# _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
# )

# # show the results on the current (interacted) frame
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_points(points, labels, plt.gca())
# show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
# plt.show()

#ann_frame_idx = 28  # the frame index we interact with
#ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
# sending all clicks (and their labels) to `add_new_points_or_box`
#points = np.array([[210, 350], [250, 220]], dtype=np.float32)
points = np.array([[262, 234], [373,226]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1], np.int32)
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
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_points(points, labels, plt.gca())
# show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

# plt.show()

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    #print(f"@@@ {out_frame_idx}, {out_obj_ids} @@@")
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

first_non_zero_frame_idx = ann_frame_idx
sliece_num = max(video_segments.keys()) + 1
pred_data = []
obj_idx = 1
for idx in range(first_non_zero_frame_idx, sliece_num):
    data = video_segments[idx][obj_idx] 
    data = torch.from_numpy(data)
    pred_data.append(data.squeeze(0))
#pred_data = np.array(pred_data)



# gt_lables = extract_mask_coordinates(nii_mask_path, first_non_zero_frame_idx, last_frame_idx)
gt_lables = folder_to_image_tensor_list(comman_img_mask_path)
special_slice = gt_lables[special]
gt_lables = torch.stack(gt_lables, dim=0).permute(1,0,2,3)
#breakpoint()

#gt_lables = gt_lables.unsqueeze(0)
pred_data = torch.stack(pred_data, dim=0)
pred_data = pred_data.unsqueeze(0)
#pred_data = pred_data.permute(1,0,3,2)
score = get_iou_and_dice(pred_data, gt_lables, special)
print(score)
#001: (0.5808679461479187, 0.7348722219467163) no setting "last frame"
#002: (0.46679407358169556, 0.6364820599555969) no setting "last frame"
#002: (0.46679407358169556, 0.6364820599555969) with setting "last frame"

# render the segmentation results every few frames

#special_pred = special_slice
# plt.close("all")
# for out_frame_idx in range(first_non_zero_frame_idx, len(frame_names), vis_frame_stride):
#     plt.figure(figsize=(6, 4))
#     plt.title(f"frame {out_frame_idx}")
#     plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
#     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
#         if(out_frame_idx ==  special + ann_frame_idx):
#             #special_pred = out_mask
#             out_mask = special_slice
#         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

# plt.show()
# special_pred =  special_pred.squeeze(0)
# special_slice =  special_slice.squeeze(0)
# show_diff(special_pred, special_slice)

######################################
'''
'''
#ann_frame_idx = 150  # further refine some details on this frame
ann_frame_idx = 100
ann_obj_id = 1  # give a unique id to the object we interact with (it can be any integers)

# show the segment before further refinement
#plt.figure(figsize=(9, 6))
#plt.title(f"frame {ann_frame_idx} -- before refinement")
#plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
#show_mask(video_segments[ann_frame_idx][ann_obj_id], plt.gca(), obj_id=ann_obj_id)

# Let's add a negative click on this frame at (x, y) = (82, 415) to refine the segment
points = np.array([[1350, 903]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
#labels = np.array([0], np.int32)
labels = np.array([1], np.int32)
_, _, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the segment after the further refinement
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx} -- after refinement")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_points(points, labels, plt.gca())
# show_mask((out_mask_logits > 0.0).cpu().numpy(), plt.gca(), obj_id=ann_obj_id)
    

vis_frame_stride = 10
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

plt.show()
'''



    