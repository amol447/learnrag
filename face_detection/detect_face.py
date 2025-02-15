from functools import reduce
from pathlib import Path
from typing import List

import sqlalchemy
import torch
import numpy as np
from sqlalchemy import select
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)
from PIL import Image, ExifTags
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from database.schema_def import face_data, image_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model_and_preprocess():
    model = retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    ).to(device)
    preprocess = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT.transforms().to(device)
    model.eval()
    return model, preprocess


def get_img_list(path: Path):
    img_types = [".jpg", ".jpeg", ".png", "JPG", "JPEG", "PNG"]
    return reduce(
        lambda x, y: x + y, [list(path.glob(f"*{img_type}")) for img_type in img_types]
    )


def process_batch(image_path_list: List[Path], model, preprocess):
    processed_img_list = [
        preprocess(Image.open(image_path)).to(device) for image_path in image_path_list
    ]
    predictions = model(processed_img_list)
    boxes_dict = {
        image_path: prediction["boxes"][
            (prediction["scores"] > 0.5) & (prediction["labels"] == 1)
        ]
        for image_path, prediction in zip(image_path_list, predictions)
    }
    for tensor in processed_img_list:
        del tensor
    torch.cuda.empty_cache()
    return boxes_dict


def chunks(l: List, batch_size):
    # divide the list into chunks of size batch_size
    for i in range(0, len(l), batch_size):  # 0, batch_size, 2*batch_size, ...
        yield l[i : min(i + batch_size, len(l))]


def process_directory(path: Path, engine, batch_size=2):
    model, preprocess = get_model_and_preprocess()
    image_path_list = get_img_list(path)

    for chunk in chunks(image_path_list, batch_size):
        boxes_dict = process_batch(chunk, model, preprocess)
        save_boxes(boxes_dict, engine)
    return boxes_dict


def create_face_data_rows(boxes_dict: dict):
    face_data_rows = [
        {
            "image_id": image_id,
            "face_box": box,
            "face_embedding": None,
            "face_tag_id": None,
        }
        for image_id, boxes in boxes_dict.items()
        for box in boxes
    ]
    return face_data_rows


def save_boxes(boxes_dict: dict, engine: sqlalchemy.engine):
    rows = create_face_data_rows(boxes_dict)
    image_id_cte = (
        select(image_data.c.id, image_data.c.path)
        .where(image_data.c.path.in_(list(boxes_dict.keys())))
        .cte("image_id_cte")
    )
    delete_query = face_data.delete().where(
        face_data.c.image_id.in_(select([image_id_cte.c.id]))
    )
    with engine.connect() as conn:
        conn.execute(delete_query)
        conn.execute(face_data.insert(), rows)
    return None
