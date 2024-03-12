import argparse

from ultralytics import YOLO

# TODO: add public volley
# TODO: add sports mot and limit number of samples in a randomize way
data = [
    '/home/tom/Projects/bv-play-break-detection/service/research/scripts/ball.yaml',
    '/home/tom/Projects/bv-play-break-detection/service/research/scripts/actions.yaml',
    '/home/tom/Projects/bv-play-break-detection/service/research/scripts/jerseys.yaml',
    '/home/tom/Projects/bv-play-break-detection/service/research/scripts/players.yaml',

]

def main(args):
    model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

    # model.train(data=data, epochs=40, imgsz=900, name="base")
    model.train(data=data, epochs=40, imgsz=1200, name="img_size_1200", batch=12)
    model.train(data=data, epochs=40, imgsz=900, name="rect=true", rect=True)
    model.train(data=data, epochs=40, imgsz=900, name="multi_scale=true", multi_scale=True)
    model.train(data=data, epochs=40, imgsz=900, name="higher_cls_loss_weight_cls=2box=5", cls=2,box=5)
    model.train(data=data, epochs=40, imgsz=900, name="no_mosaic_and_erasing", mosaic=0, erasing=0)
    model.train(data=data, epochs=40, imgsz=900, name="no_mosaic_and_erasing_shear=0.2_perspective=0.2", mosaic=0, erasing=0,
                shear=0.2, perspective=0.2)


    # TODO: weight losses for diff heads


# add the experiment name as argument
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    args = parser.parse_args()
    main(args)


