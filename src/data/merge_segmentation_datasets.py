import os
import shutil

synthetic_images = "data/segmentation_dataset/images"
synthetic_masks = "data/segmentation_dataset/masks"

weak_images = "data/weak_labels/images"
weak_masks = "data/weak_labels/masks"

final_images = "data/final_segmentation_dataset/images"
final_masks = "data/final_segmentation_dataset/masks"

os.makedirs(final_images, exist_ok=True)
os.makedirs(final_masks, exist_ok=True)


# copy synthetic dataset
for file in os.listdir(synthetic_images):

    src_img = os.path.join(synthetic_images,file)
    src_mask = os.path.join(synthetic_masks,file.replace(".tif",".png"))

    dst_img = os.path.join(final_images,"synthetic_"+file)
    dst_mask = os.path.join(final_masks,"synthetic_"+file.replace(".tif",".png"))

    shutil.copy(src_img,dst_img)
    shutil.copy(src_mask,dst_mask)


# copy weak dataset
for file in os.listdir(weak_images):

    src_img = os.path.join(weak_images,file)
    src_mask = os.path.join(weak_masks,file.replace(".tif",".png"))

    dst_img = os.path.join(final_images,"weak_"+file)
    dst_mask = os.path.join(final_masks,"weak_"+file.replace(".tif",".png"))

    shutil.copy(src_img,dst_img)
    shutil.copy(src_mask,dst_mask)


print("Dataset merged successfully")