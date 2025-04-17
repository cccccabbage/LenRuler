import cv2
import numpy as np
from skimage import morphology, measure
from scipy.ndimage import label

def _writeBool(mask:np.ndarray, path:str):
    mask = mask.astype(np.uint8) * 255
    cv2.imwrite(path, mask)

class LabelResult:
    def __init__(self, point, mask):
        self.point = point
        self.mask = mask
        self.notEmpty = len(np.unique(self.mask)) == 2
    
    def __str__(self) -> str:
        return f"{self.point}\n{self.mask}"


class GaussianResult:
    def __init__(self):
        self.insMasks = []

        self.buds = []
        self.rices = []
    
    def __str__(self) -> str:
        return f"instance count: {len(self.insMasks)}"
    
    def allBudMasks(self):
        return np.sum([bud.mask for bud in self.buds if bud.notEmpty], axis=0, dtype=float).astype(bool)

    def allRiceMasks(self):
        return np.sum([rice.mask for rice in self.rices if rice.notEmpty], axis=0, dtype=float).astype(bool)


def _disk5():
    # exactly the same as `strel('disk', 5)` in matlab
    # used for erosion and dilation
    kernel = np.ones((9, 9), np.uint8) * 255
    kernel[0, 0] = 0
    kernel[0, 1] = 0
    kernel[1, 0] = 0

    kernel[8, 0] = 0
    kernel[8, 1] = 0
    kernel[7, 0] = 0

    kernel[0, 8] = 0
    kernel[0, 7] = 0
    kernel[1, 8] = 0

    kernel[8, 8] = 0
    kernel[8, 7] = 0
    kernel[7, 8] = 0

    return kernel


def _removeSmall(oneMask):
    # remove small area of noise in the mask
    m = oneMask.astype(np.uint8)
    labeled_mask, num_features = label(m)
    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0
    largest_label = sizes.argmax()
    largest_componetn = (labeled_mask == largest_label).astype(bool)
    return largest_componetn


def _normalize(mat):
    return (mat - np.min(mat)) / (np.max(mat) - np.min(mat))


# input:
#   img: np.array of shape (H, W, 3)
#   masks: np.array of shape (N, H, W), where N stands for the number of instances
# return:
#   GaussianResult
def gaussianLabel(img, masks, boxes, normalize=False, erode=True, dilate=False, conf=0.006):
    img = img.astype(float)
    if np.max(img) <= 1.0:
        img = img * 255.0

    wholeMask = np.sum(masks, axis=0, dtype=float).astype(bool).astype(np.uint8) * 255
    wholeMask = wholeMask[0]

    if erode:
        kernel = _disk5()
        mask_eroded = cv2.erode(wholeMask, kernel, iterations=1).astype(bool)
    else:
        mask_eroded = np.ones_like(wholeMask, dtype=bool)

    # preserve the masks inside the boxes
    boxMask = np.zeros_like(wholeMask, dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = box
        boxMask[y1:y2, x1:x2] = True
    mask_eroded = mask_eroded & boxMask

    # measure and get centroids
    centroids = []
    for box in boxes:
        x1, y1, x2, y2 = box
        oneBoxMask = np.zeros_like(wholeMask, dtype=bool)
        oneBoxMask[y1:y2, x1:x2] = True
        oneBoxMask = oneBoxMask & mask_eroded
        if len(np.unique(oneBoxMask)) == 1:
            continue

        centroids.append(measure.regionprops(oneBoxMask.astype(np.uint8))[0].centroid)

    # format r-b and g-b data
    b, g, r = cv2.split(img)
    rbgb = img[..., :2].copy()
    if normalize:
        rbgb[..., 0] = _normalize(r - b)
        rbgb[..., 1] = _normalize(g - b)
    else:
        rbgb[..., 0] = r - b
        rbgb[..., 1] = g - b

    # select the representative points and compute the mean vector and inverse covariance matrix
    repPoints = [] # representative points
    for centroid in centroids:
        lt = (int(centroid[1]-1), int(centroid[0]-1))
        rb = (int(centroid[1]+2), int(centroid[0]+2))

        rbVal = rbgb[lt[1]:rb[1], lt[0]:rb[0], 0].ravel()
        gbVal = rbgb[lt[1]:rb[1], lt[0]:rb[0], 1].ravel()
        repPoints.append(np.column_stack((rbVal, gbVal)))
    repPoints = np.vstack(repPoints)
    meanVec = np.mean(repPoints, axis=0)
    invCovMatrix = np.linalg.inv(np.cov(repPoints, rowvar=False))

    maskToLabel = mask_eroded

    # compute the probability of each pixel inside the mask
    indices = np.where(maskToLabel)
    rbgb = rbgb[indices] # shape = (n, 2)
    probs = np.zeros_like(maskToLabel, dtype=float)
    for i in range(len(indices[0])):
        diffVector = rbgb[i] - meanVec
        prob = np.exp(-0.5 * (diffVector @ invCovMatrix @ diffVector))
        probs[indices[0][i], indices[1][i]] = prob
    probs = probs >= conf

    # format bud labels and rice labels
    budMask = ~probs
    riceMask = probs
    riceMask[~maskToLabel] = False
    budMask [~maskToLabel] = False

    if erode and dilate:
        riceMask = riceMask.astype(np.uint8) * 255
        riceMask = cv2.dilate(riceMask, kernel, iterations=1).astype(bool)

    # remove small area
    results = GaussianResult()
    results.insMasks = masks
    for mask in masks:
        oneBudMask = np.zeros_like(mask, dtype=bool)
        oneRiceMask = np.zeros_like(mask, dtype=bool)
        oneBudMask = budMask & mask
        oneRiceMask = riceMask & mask
        oneBudMask = oneBudMask & (~oneRiceMask)

        oneBudMask = _removeSmall(oneBudMask)
        oneRiceMask = _removeSmall(oneRiceMask)

        budCenter = maskCenter(oneBudMask)
        riceCenter = maskCenter(oneRiceMask)
        results.rices.append(LabelResult(riceCenter, oneRiceMask))
        results.buds.append(LabelResult(budCenter, oneBudMask))

    return results

# input:
#   img: np.array of shape (H, W, 3)
#   masks: np.array of shape (N, H, W), where N stands for the number of instances
#   points: np.array of shape (N, 2), where N stands for the number of instances
# return:
#   GaussianResult
def gaussianLabel2(img, masks, points, normalize=False, erode=True, dilate=False, conf=0.006):
    img = img.astype(float)
    if np.max(img) <= 1.0:
        img = img * 255.0

    wholeMask = np.sum(masks, axis=0, dtype=float).astype(bool).astype(np.uint8) * 255
    wholeMask = wholeMask[0]

    if erode:
        kernel = _disk5()
        mask_eroded = cv2.erode(wholeMask, kernel, iterations=1).astype(bool)
    else:
        mask_eroded = np.ones_like(wholeMask, dtype=bool)

    # format r-b and g-b data
    b, g, r = cv2.split(img)
    rbgb = img[..., :2].copy()
    if normalize:
        rbgb[..., 0] = _normalize(r - b)
        rbgb[..., 1] = _normalize(g - b)
    else:
        rbgb[..., 0] = r - b
        rbgb[..., 1] = g - b

    # select the representative points and compute the mean vector and inverse covariance matrix
    repPoints = [] # representative points
    for centroid in points:
        lt = (int(centroid[1]-1), int(centroid[0]-1))
        rb = (int(centroid[1]+2), int(centroid[0]+2))

        rbVal = rbgb[lt[1]:rb[1], lt[0]:rb[0], 0].ravel()
        gbVal = rbgb[lt[1]:rb[1], lt[0]:rb[0], 1].ravel()
        repPoints.append(np.column_stack((rbVal, gbVal)))
    repPoints = np.vstack(repPoints)
    meanVec = np.mean(repPoints, axis=0)
    invCovMatrix = np.linalg.inv(np.cov(repPoints, rowvar=False))

    maskToLabel = wholeMask.astype(bool)

    # compute the probability of each pixel inside the mask
    indices = np.where(maskToLabel)
    rbgb = rbgb[indices] # shape = (n, 2)
    probs = np.zeros_like(maskToLabel, dtype=float)
    for i in range(len(indices[0])):
        diffVector = rbgb[i] - meanVec
        prob = np.exp(-0.5 * (diffVector.T @ invCovMatrix @ diffVector))
        probs[indices[0][i], indices[1][i]] = prob
    probs = probs >= conf

    # format bud labels and rice labels
    budMask = (~probs) & maskToLabel
    riceMask = (probs) & maskToLabel

    if erode and dilate:
        riceMask = riceMask.astype(np.uint8) * 255
        riceMask = cv2.dilate(riceMask, kernel, iterations=1).astype(bool)

    # remove small area
    results = GaussianResult()
    results.insMasks = masks
    for mask in masks:
        oneBudMask = budMask & mask
        oneRiceMask = riceMask & mask
        oneBudMask = oneBudMask & (~oneRiceMask)

        oneBudMask = _removeSmall(oneBudMask)
        oneRiceMask = _removeSmall(oneRiceMask)

        budCenter = maskCenter(oneBudMask)
        riceCenter = maskCenter(oneRiceMask)
        results.buds.append(LabelResult(riceCenter, oneRiceMask))
        results.rices.append(LabelResult(budCenter, oneBudMask))

    return results


# input mask: np.array of shape (H, W)
# notice: the input mask must be a mask of one instance
def maskCenter(mask: np.array):
    assert len(mask) == 1
    properties = measure.regionprops(measure.label(mask[0]))

    if len(properties) == 0:
        return (-1, -1)

    # select the centroid of the largest area
    properties = sorted(properties, key=lambda x: x.area, reverse=True)
    p = properties[0].centroid
    p = (int(p[1]), int(p[0]))

    return p
