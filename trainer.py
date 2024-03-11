from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data=['/home/tom/Projects/bv-play-break-detection/service/research/scripts/actions.yaml',
                            '/home/tom/Projects/bv-play-break-detection/service/research/scripts/ball.yaml'], epochs=60, imgsz=900)