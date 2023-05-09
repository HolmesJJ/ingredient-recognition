import os
import cv2
import glob
import random
import shutil
import numpy as np
import pandas as pd


CATEGORY_PATH = "FoodSeg103/category_id.txt"
RAW_DATA_PATH = "FoodSeg103/"
DATASET_PATH = "dataset/"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2


def get_classes():
    categories = pd.read_csv(CATEGORY_PATH, sep="\t", names=["id", "name"])
    ids = categories["id"].to_list()
    classes = categories["name"].to_list()
    classes = [cls.strip() for cls in classes]
    return ids, classes


def crop_ingredients(indices, dataset, source_folder, target_folder):
    ids, classes = get_classes()
    for idx in indices:
        filename = os.path.basename(dataset[idx])
        image = cv2.imread(RAW_DATA_PATH + "img_dir/" + source_folder + filename)
        annotation = cv2.imread(RAW_DATA_PATH + "ann_dir/" + source_folder + filename)
        ann_b, ann_g, ann_r = cv2.split(annotation)
        unique_channels = np.unique(ann_r)
        for channel in unique_channels:
            mask = cv2.bitwise_and(image, image, mask=np.where(ann_r == channel, 1, 0).astype("uint8"))
            tmp = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
            mask_b, mask_g, mask_r = cv2.split(mask)
            rgba = [mask_b, mask_g, mask_r, alpha]
            masked_img = cv2.merge(rgba, 4)
            contours, _ = cv2.findContours(cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            top_left_x = image.shape[0]
            top_left_y = image.shape[1]
            right_bottom_x = right_bottom_y = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if top_left_x >= x:
                    top_left_x = x
                if top_left_y >= y:
                    top_left_y = y
                if right_bottom_x <= x + w:
                    right_bottom_x = x + w
                if right_bottom_y <= y + h:
                    right_bottom_y = y + h
            cropped_img = masked_img[top_left_y:right_bottom_y, top_left_x:right_bottom_x]
            class_name = classes[ids.index(channel)]
            cv2.imwrite(os.path.join(DATASET_PATH + target_folder + class_name, filename), cropped_img)


def create_dataset():
    ids, classes = get_classes()
    dataset_folders = ["train", "val", "test"]
    os.makedirs(DATASET_PATH)
    for dataset_folder in dataset_folders:
        dataset_folder_path = os.path.join(DATASET_PATH, dataset_folder)
        os.makedirs(dataset_folder_path)
        for cls in classes:
            class_folder = os.path.join(dataset_folder_path, cls)
            os.makedirs(class_folder)

    train_images = glob.glob(f"{RAW_DATA_PATH}img_dir/train/*")
    test_images = glob.glob(f"{RAW_DATA_PATH}img_dir/test/*")
    full_indices = range(len(train_images))
    train_indices = random.sample(full_indices, int(len(full_indices) * TRAIN_RATIO))
    val_indices = list(set(full_indices) - set(train_indices))
    test_indices = range(len(test_images))
    print(len(full_indices), (len(train_indices) + len(val_indices)), len(test_indices))

    crop_ingredients(train_indices, train_images, "train/", "train/")
    crop_ingredients(val_indices, train_images, "train/", "val/")
    crop_ingredients(test_indices, test_images, "test/", "test/")


def convert_jpg_to_png():
    dataset_dirs = ["train", "test"]
    for dataset_dir in dataset_dirs:
        shutil.copytree(f"{RAW_DATA_PATH}img_dir/{dataset_dir}", f"{RAW_DATA_PATH}img_dir/tmp")
        shutil.rmtree(f"{RAW_DATA_PATH}img_dir/{dataset_dir}")
        os.makedirs(f"{RAW_DATA_PATH}img_dir/{dataset_dir}")
        images = glob.glob(f"{RAW_DATA_PATH}img_dir/tmp/*")
        for image in images:
            jpg_img = cv2.imread(image)
            png_fp = os.path.join(f"{RAW_DATA_PATH}img_dir/{dataset_dir}",
                                  os.path.splitext(os.path.basename(image))[0] + ".png")
            cv2.imwrite(png_fp, jpg_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        shutil.rmtree(f"{RAW_DATA_PATH}img_dir/tmp")


if __name__ == "__main__":
    if not os.path.isdir(DATASET_PATH):
        convert_jpg_to_png()
        create_dataset()
