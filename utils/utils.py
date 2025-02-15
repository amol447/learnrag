from functools import reduce
from pathlib import Path
from typing import List

from PIL import ExifTags
from exiftool import ExifToolHelper
from database.schema_def import image_data, engine


def find_exif(image_path_list: List[Path]):
    with ExifToolHelper() as et:
        ans = dict(zip(image_path_list, et.get_metadata(image_path_list)))
    return ans


def create_image_path_rows(image_path_list: List[Path]):
    exif_dict = find_exif(image_path_list)
    rows = [
        {
            "path": str(image_path),
            "exif": exif,
            "image_type": exif.get("File:FileType"),
        }
        for image_path, exif in exif_dict.items()
    ]
    return rows


def save_image_paths(path: Path):
    image_paths = get_img_list(path)
    rows = create_image_path_rows(image_paths)
    with engine.connect() as conn:
        conn.execute(image_data.insert().values(rows))
        conn.commit()


def get_img_list(path: Path):
    img_types = [".jpg", ".jpeg", ".png", "JPG", "JPEG", "PNG"]
    return reduce(
        lambda x, y: x + y, [list(path.glob(f"*{img_type}")) for img_type in img_types]
    )
