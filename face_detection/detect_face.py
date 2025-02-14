from pathlib import Path
from typing import List

import torch
import numpy as np
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)
from PIL import Image, ExifTags
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model_and_preprocess():
    model = retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    ).to(device)
    preprocess = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT.transforms().to(device)
    model.eval()
    return model, preprocess


def get_img_list(path: Path):
    img_types = [".jpg", ".jpeg", ".png"]
    return (
        list(path.glob("**/*.jpg"))
        + list(path.glob("**/*.jpeg"))
        + list(path.glob("**/*.png"))
    )


def process_image_list(image_path_list: List[Path], model, preprocess):
    processed_img_list = [
        preprocess(Image.open(image_path)).to(device) for image_path in image_path_list
    ]
    predictions = model(processed_img_list)
    boxes_dict = {
        image_path: prediction["boxes"][prediction["scores"] > 0.7]
        for image_path, prediction in enumerate(image_path_list, predictions)
    }

    return boxes_dict


def save_boxes(boxes_dict: dict):
    for image_path, boxes in boxes_dict.items():
        with open(image_path.with_suffix(".txt"), "w") as f:
            for box in boxes:
                f.write(f"{box}\n")
