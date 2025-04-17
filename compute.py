import models
import gaussian as gs

import os
import cv2
import networkx as nx
import numpy as np
from skimage import measure, morphology

def _writeBool(mask:np.ndarray, path:str):
    mask = mask.astype(np.uint8) * 255
    cv2.imwrite(path, mask)

def _wholeProcess(img, yolo, sam, imgName):
    saveFolder = "output/" + imgName.split(".")[0]
    masks, boxes = models.getMasksAndBoxes(img, yolo, sam)
    boxImg = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(boxImg, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(f"{saveFolder}/box.png", boxImg)

    gsResults = gs.gaussianLabel(img, masks, boxes, normalize=True, erode=True, dilate=True)

    points = []
    labels = []
    for rice, bud in zip(gsResults.rices, gsResults.buds):
        point = [[-1,  -1], [-1, -1]]
        label = [-1, -1]
        if rice.notEmpty:
            point[0] = rice.point
            label[0] = 1
        if bud.notEmpty:
            point[1] = bud.point
            label[1] = 1
        points.append(point)
        labels.append(label)

    masks = models.getMasksByPoints(img, sam, points, labels)

    ricePoints = []
    canvas = img.copy()
    for p in points:
        if p[0][0] != -1:
            cv2.circle(canvas, tuple(p[0]), 3, (0, 255, 0), -1) # green
            ricePoints.append(p[0])
        if p[1][0] != -1:
            cv2.circle(canvas, tuple(p[1]), 3, (0, 0, 255), -1) # red
    cv2.imwrite(f"{saveFolder}/points.png", canvas)

    # classify the masks of instances to rice and bud
    gsResults = gs.gaussianLabel2(img, masks, ricePoints, normalize=True, erode=True, dilate=True, conf=0.006)

    return gsResults


def __getSkl(mask:np.ndarray):
    mask = mask.astype(np.uint8) * 255

    # dilate
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
    mask = mask.astype(bool)

    properties = measure.regionprops(measure.label(mask))
    # sort by area with ascending order
    properties = sorted(properties, key=lambda x: x.area, reverse=True)
    
    if len(properties) == 0:
        return np.zeros_like(mask)
    fullSizeSkl = properties[0].coords

    ret = np.zeros_like(mask, dtype=bool)
    ret[tuple(fullSizeSkl.T)] = True

    fullSizeSkl = morphology.skeletonize(ret).astype(bool)
    return fullSizeSkl


kernel = np.array([[1,  1, 1],
                   [1, 10, 1],
                   [1,  1, 1]], dtype=np.uint8)

# difine the 8 neighbors
neighbors = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1)
]

def __build_graph(points):
    G = nx.Graph()

    for x, y in points:
        neiPoints = []
        for dx, dy in neighbors:
            neiX, neiY = x + dx, y + dy
            neiPoints.append((neiX, neiY))
        for neiX, neiY in neiPoints:
            if (neiX, neiY) in points:
                G.add_edge((x, y), (neiX, neiY))

    return G

def getLen(riceSkl: np.ndarray | None, budSkl: np.ndarray | None) -> float:
    if riceSkl is None or budSkl is None:
        return 0.0
    budSkl = budSkl.astype(np.uint8)

    convolved = cv2.filter2D(budSkl, -1, kernel)

    its = np.where(convolved >= 13)
    ends = np.where(convolved == 11)
    its = list(zip(*its))
    ends = list(zip(*ends))

    allPoints = np.where(budSkl)
    allPoints = list(zip(*allPoints))
    ricePoints = np.where(riceSkl)
    G = __build_graph(allPoints)

    iPoint = ()
    ePoint = ()
    if len(its) == 0 and len(ends) == 0:
        return 0.0
    elif len(its) == 0 or len(ends) == 0:
        minDist = np.inf
        maxDist = 0.0
        for p in allPoints:
            for r in zip(*ricePoints):
                dist = np.linalg.norm(np.array(p) - np.array(r))
                if dist < minDist:
                    minDist = dist
                    iPoint = p
                if dist > maxDist:
                    maxDist = dist
                    ePoint = p
    else:
        # compute the min distance from each intersection to ricePoints
        minDist = np.inf
        for p in its:
            for r in zip(*ricePoints):
                dist = np.linalg.norm(np.array(p) - np.array(r))
                if dist < minDist:
                    minDist = dist
                    iPoint = p

        maxDist = 0.0
        for p in ends:
            for r in zip(*ricePoints):
                dist = np.linalg.norm(np.array(p) - np.array(r))
                if dist > maxDist:
                    maxDist = dist
                    ePoint = p

    shortest_path = nx.shortest_path(G, source=ePoint, target=iPoint)
    # compute the length of the shortest path according to the distance between each two points
    lengths = 0.0
    for i in range(len(shortest_path) - 1):
        lengths += np.linalg.norm(np.array(shortest_path[i]) - np.array(shortest_path[i + 1]))

    return lengths

def __CalcuateLength(gsResults: gs.GaussianResult, retSkl=False) -> list[tuple[int, float]]:
    if retSkl:
        budSkl = np.zeros_like(gsResults.rices[0].mask, dtype=bool)
        riceSkl = np.zeros_like(gsResults.rices[0].mask, dtype=bool)

    lengths = []
    for i in range(len(gsResults.insMasks)):
        riceSklFull = None
        budSklFull = None
        if gsResults.rices[i].notEmpty:
            riceSklFull = __getSkl(gsResults.rices[i].mask[0])
        if gsResults.buds[i].notEmpty:
            budSklFull = __getSkl(gsResults.buds[i].mask[0])

        length = getLen(riceSklFull, budSklFull)

        lengths.append((i, length))

        if retSkl:
            riceSkl = (riceSkl | riceSklFull) if riceSklFull is not None else riceSkl
            budSkl  = (budSkl  | budSklFull) if budSklFull is not None else budSkl

    if retSkl:
        return lengths, (riceSkl, budSkl)
    return lengths

def calPix2MM(img: np.ndarray, realArea:float) -> float:

    def angle_between_vectors(v1, v2):
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        v2 = v2.reshape([2, 1])
        dot_product = np.dot(v1, v2)
        magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
        angle = np.arccos(dot_product / magnitude)
        return np.degrees(angle)

    def is_right_angle(approx):
        for i in range(4):
            v1 = approx[i] - approx[(i - 1) % 4]
            v2 = approx[i] - approx[(i + 1) % 4]
            angle = angle_between_vectors(v1, v2)
            if abs(angle - 90) > 10:  # 10 degree tolerance
                return False
        return True

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=7)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    pixArea = 0.0
    for i, contour in enumerate(contours):
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            if is_right_angle(approx):
                area = cv2.contourArea(approx)
                if area > pixArea:
                    pixArea = area

    return np.sqrt(realArea / pixArea)

def computeOneLength(imgPath, pix2mm, yolo, sam):
    img = cv2.imread(imgPath)
    img = cv2.resize(img, (1920, 1080))
    imgName = imgPath.split("/")[-1].replace(".jpg", ".png")
    saveFolder = "output/" + imgName.split(".")[0]
    os.makedirs(saveFolder, exist_ok=True)
    gsResults = _wholeProcess(img, yolo, sam, imgName=imgName)

    canvas = img.copy()
    budMask = gsResults.allBudMasks()
    riceMask = gsResults.allRiceMasks()
    canvas[riceMask[0]] = (0, 255, 0) # green
    canvas[ budMask[0]] = (0, 0, 255) # red
    cv2.imwrite(f"{saveFolder}/mask.png", canvas)

    lengths, (riceSkl, budSkl) = __CalcuateLength(gsResults, retSkl=True)
    sklImg = img.copy()
    sklImg[ budSkl[0]] = (0, 0, 255)
    sklImg[riceSkl[0]] = (0, 255, 0)
    cv2.imwrite(f"{saveFolder}/skl.png", sklImg)

    lengths = [x[1] * pix2mm for x in lengths]

    if pix2mm == 1.0:
        print("mean(pix): ", np.mean(lengths))
        print("std (pix): ", np.std(lengths))
    else:
        print("mean(mm): ", np.mean(lengths))
        print("std (mm): ", np.std(lengths))
