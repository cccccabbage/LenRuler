import ultralytics as ut
import numpy as np

def getSAM(modelPath: str = None):
    # the leagle SAM model are available at ultralytics's official website
    # https://docs.ultralytics.com/models/sam/
    if modelPath is not None:
        return ut.SAM(modelPath)
    else:
        # This would automatically download the model from ultralytics's official website,
        # but just the sam-b model. 
        # If a model of larger scale is needed, please download it manually before passing the path.
        return ut.SAM()

def getYOLO(modelPath: str):
    # Please train your own model before using this function.
    # And a function to train YOLO is provided below.
    return ut.YOLO(modelPath)   

def trainYOLO(dataYAML : str):
    # This is just a simple example of how to train a YOLO model.
    # Check the official document of ultralytics for more details.
    # https://docs.ultralytics.com/modes/train/
    model = ut.YOLO()
    model.train(dataYAML, epochs=300)


def getMasks(img:np.ndarray, yolo:ut.YOLO, sam:ut.SAM, iou:float=0.7, conf:float=0.7):
    yoloResult = yolo(img, iou=iou, conf=conf)[0]
    boxes = yoloResult.boxes.xyxy.cpu().numpy()
    samResult = sam(img, bboxes=boxes)[0]
    masks = samResult.masks.data.cpu().numpy()

    return masks

def getMasksByPoints(img:np.ndarray, sam:ut.SAM, points, labels):
    samResult = sam(img, points=points, labels=labels)[0]
    masks = samResult.masks.data.cpu().numpy()

    return masks

def getMasksAndBoxes(img:np.ndarray, yolo:ut.YOLO, sam:ut.SAM, iou=0.7, conf=0.7):
    yoloResult = yolo(img, iou=iou, conf=conf)[0]
    boxes = yoloResult.boxes.xyxy.cpu().numpy().round().astype(int)
    samResult = sam(img, bboxes=boxes)[0]
    masks = samResult.masks.data.cpu().numpy()

    return masks, boxes