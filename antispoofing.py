import numpy as np
import cv2
import torch
import pandas as pd
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb
from datasouls_antispoof.pre_trained_models import create_model
from datasouls_antispoof.class_mapping import class_mapping
import json


def antiSpoofing(img):
    try:
        model = create_model("tf_efficientnet_b3_ns")
        model.eval()
        image_replay = load_rgb(img)

        transform = albu.Compose([albu.PadIfNeeded(min_height=400, min_width=400),
                                albu.CenterCrop(height=400, width=400),
                                albu.Normalize(p=1),
                                albu.pytorch.ToTensorV2(p=1)], p=1)

        with torch.no_grad():
            prediction = model(torch.unsqueeze(
                transform(image=image_replay)['image'], 0)).numpy()[0]

        df = pd.DataFrame({"prediction": prediction*100,
                        "class_name": class_mapping.keys()})

        res = df.set_index('class_name').T.to_dict('list')

        return res

    except Exception as e:
        error_json = {
            "Status : ": e
        }
        return error_json


if __name__ == "__main__":
    res = antiSpoofing("Test Cases/ocrImageTestCases/1.jpg")
    print(type(res))
    print(res)
