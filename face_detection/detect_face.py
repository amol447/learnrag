from pathlib import Path
from typing import List, Dict

import sqlalchemy
import torch
from sqlalchemy import select
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)
from PIL import Image
from database.schema_def import face_data, image_data
from utils.utils import get_img_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model_and_preprocess():
    model = retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    ).to(device)
    preprocess = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT.transforms().to(device)
    model.eval()
    return model, preprocess


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


def process_directory(engine, batch_size=2):
    model, preprocess = get_model_and_preprocess()
    image_query = select([image_data.c.path, image_data.c.id])
    path_id_dict = dict()
    with engine.connect() as conn:
        result = conn.execute(image_query)
        for row in result.fetchall():
            row_dict = dict(row)
            path_id_dict[row_dict["path"]] = row_dict["id"]

    for chunk in chunks(path_id_dict.keys(), batch_size):
        boxes_dict = process_batch(chunk, model, preprocess)
        for image_path in chunk:
            image_id = path_id_dict[image_path]
            boxes_dict[image_id] = boxes_dict.pop(image_path)
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


def save_boxes(boxes_dict: Dict, engine: sqlalchemy.engine):
    rows = create_face_data_rows(boxes_dict)

    delete_query = face_data.delete().where(
        face_data.c.image_id.in_(list(boxes_dict.keys()))
    )
    with engine.connect() as conn:
        conn.execute(delete_query)
        conn.execute(face_data.insert(), rows)
    return None
