import torch
import ultralytics as ut
import numpy as np
from SAM.segment_anything import sam_model_registry, SamPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"

def getSAM(modelPath: str = None):
    sam = sam_model_registry["vit_b"](modelPath)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


def getYOLO(modelPath: str):
    # Please train your own model before using this function.
    # And a function to train YOLO is provided below.
    return ut.YOLO(modelPath).to(device)

def trainYOLO(dataYAML : str):
    # This is just a simple example of how to train a YOLO model.
    # Check the official document of ultralytics for more details.
    # https://docs.ultralytics.com/modes/train/
    model = ut.YOLO()
    model.train(dataYAML, epochs=300)


def getMasksByPoints(img:np.ndarray, sam:SamPredictor, points, labels):
    sam.set_image(img)
    masks = []
    for point, label in zip(points, labels):
        p = []
        l = []
        if label[0] != -1:
            p.append(point[0])
            l.append(label[0])
        if label[1] != -1:
            p.append(point[1])
            l.append(label[1])
        p = np.array(p)
        l = np.array(l)
        
        mask, _, _ = sam.predict(point_coords=p, point_labels=l, multimask_output=False)
        masks.append(mask)
    masks = np.array(masks)

    return masks

def getMasksAndBoxes(img:np.ndarray, yolo:ut.YOLO, sam:SamPredictor, iou=0.7, conf=0.3):
    yoloResult = yolo(img, iou=iou, conf=conf)[0]
    boxes = yoloResult.boxes.xyxy.cpu().numpy().round().astype(int)

    sam.set_image(img)
    masks = []
    for box in boxes:
        mask, _, _ = sam.predict(box=box, multimask_output=False)
        masks.append(mask)
    masks = np.array(masks)

    return masks, boxes