
import argparse
import gc
import os

import torch

from ultralytics import YOLO
from ultralytics.engine.trainer import get_training_output_dir

# TODO: weight losses for diff heads

data = [
    'ball-tracking',
    "beach_actions_merge_freeball_receive",
    "coco_manual_merge_freeball_receive_minor_ds_fixes_applied",
    "jersey",
    "players_detections"
]



def run_exp(**kwargs):
    try:
        base_model_path = kwargs.pop("base_model", 'yolov8m.pt')
        if kwargs.get("resume", False):
            base_model_path = os.path.join(get_training_output_dir(), kwargs["name"], "weights", "last.pt")
        model = YOLO(base_model_path)
        model.train(data=data, save_period=1, epochs=kwargs.pop("epochs", 50), **kwargs)
    except Exception as e:
        raise
        print(e)
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main(args):
    # run_exp(imgsz=1200, name="img_size_1200", batch=12)
    # run_exp(imgsz=900, name="multi_scale=true", multi_scale=True, batch=10, close_mosaic=0)
    # run_exp(imgsz=900, name="no_mosaic_and_erasing", mosaic=0, erasing=0, close_mosaic=0)
    # run_exp(imgsz=900, name="no_mosaic_and_erasing_shear=0.2_perspective=0.2", mosaic=0, erasing=0, shear=0.2, perspective=0.2,
    #         close_mosaic=0)
    # run_exp(imgsz=1200, name="delete", close_mosaic=0, batch=12)
    # run_exp(imgsz=1200, name="with_players_mot", close_mosaic=0, batch=12)
    # run_exp(imgsz=1200, name="with_beach", close_mosaic=0, batch=12)
    run_exp(imgsz=1200, name="removing_wrong_annotations_except_jersey_2", close_mosaic=0, batch=12, epochs=70)
    # run_exp(imgsz=1200, name="delete", close_mosaic=0, batch=12, epochs=2)
    # run_exp(imgsz=1200, name="img_size_1200_yolov8l",  close_mosaic=0, batch=8, base_model="yolov8l.pt")
    # run_exp(imgsz=1056, name="img_size_1200_half_close_mosaic_last_10", close_mosaic=10, half=True)

    # run_exp(imgsz=900, name="higher_cls_loss_weight_cls=2box=5", cls=2, box=5, close_mosaic=0)




# add the experiment name as argument
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--exp-name", type=str, required=True)
    args = parser.parse_args()
    main(args)


