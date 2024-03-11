import argparse

from ultralytics import YOLO


# TODO: add public volley
# TODO: add sports mot and limit number of samples in a randomize way

def main(args):
    model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

    model.train(data=
    [
        '/home/tom/Projects/bv-play-break-detection/service/research/scripts/ball.yaml',
        '/home/tom/Projects/bv-play-break-detection/service/research/scripts/actions.yaml',
        '/home/tom/Projects/bv-play-break-detection/service/research/scripts/jerseys.yaml',
        '/home/tom/Projects/bv-play-break-detection/service/research/scripts/players.yaml',

    ], epochs=60, imgsz=900, name=args.exp_name)


# add the experiment name as argument
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    args = parser.parse_args()
    main(args)


