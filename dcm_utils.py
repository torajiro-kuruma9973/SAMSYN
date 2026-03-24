import SimpleITK as sitk
from pathlib import Path
import torch 
from collections import OrderedDict
from PIL import Image
import numpy as np

def get_normed_tensor_from_dcm(dicom_dir): #normed result into [0, 1]
    sitk_image = sitk.ReadImage(dicom_dir)
    numpy_array = sitk.GetArrayFromImage(sitk_image)
    
    assert len(numpy_array.shape) ==  3 # 3-dim is mandatory

    tnsr = torch.tensor(numpy_array) # shape: [1, h, w]
    min_val = torch.min(tnsr.to(torch.float32))
    max_val = torch.max(tnsr.to(torch.float32))
    
    if max_val - min_val > 0:
        normalized_slice = (tnsr - min_val) / (max_val - min_val)
    else:
        normalized_slice = tnsr - min_val

    return normalized_slice

# special_str: the prefix or other str in file name to be removed
def dcm2jpg(dicom_dir, output_dir, special_str):
    name = Path(dicom_dir).stem
    name = output_dir + name + ".jpg"
    #name = name.replace(special_str, "")
    
    slice = get_normed_tensor_from_dcm(dicom_dir)
    slice = slice * 255
    tensor_uint8 = slice.to(torch.uint8).cpu()
    numpy_array = tensor_uint8.squeeze().numpy() #[1, h, w]  -->  [h, w]

    img = Image.fromarray(numpy_array)
    img.save(name, format="JPEG")
    

def convert_all_dcm_files(dicom_dir, output_dir, special_str):
    folder = Path(dicom_dir)
    file_paths = sorted(folder.glob('*.dcm'))

    if not file_paths:
        print(f"ERROR: cannot find any dcm file in {dicom_dir} ...")
        return None
    
    for file_path in file_paths:
        dcm2jpg(file_path, output_dir, special_str)
        

def dicom_to_nifti_sitk(dicom_dir, output_filepath):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    
    if not dicom_names:
        print("ERROR: no dcm file is located...")
        return
        
    reader.SetFileNames(dicom_names)
    
    sitk_image = reader.Execute()
    
    sitk.WriteImage(sitk_image, output_filepath)
    print(f"Done! The file is saved in: {output_filepath}")

def file_name_process(folder):
    processed_list = [name.replace("IM-", "") for name in folder] # remove prefix "IM-"
    def get_sort_key(filename):
        core_part = filename.replace(".jpg", "")
        x_str, y_str = core_part.split("-")
        return (int(x_str), int(y_str))
    processed_list.sort(key=get_sort_key)
    rst = []

    return processed_list

def is_folder_empty_pathlib(folder_path):
    folder = Path(folder_path)
    
    return not any(folder.iterdir())

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

if __name__=='__main__':
    convert_all_dcm_files("raw_datasets/BrainTumorMRI/", "jpg_datasets/")