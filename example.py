import models
import compute
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--yolo", type=str, required=True) # path to pretrained YOLO model
    parser.add_argument("--sam", type=str, required=True) # path to local SAM model

    # the marking object is one-eighth of an A4 paper in the paper
    # be sure to measure the area of your own the marking object in mm^2
    # or you can pass the pix2mm ratio directly
    parser.add_argument("--markImg", type=str, default=None) # path to image with a marking object
    parser.add_argument("--markArea", type=float, default=None) # area of the marking object in mm^2
    parser.add_argument("--pix2mm", type=float, default=None) # pixel to mm ratio

    args = parser.parse_args()

    yolo = models.getYOLO(args.yoloPath)
    sam = models.getSAM(args.samPath)

    if args.markImg is not None:
        assert args.markArea is not None
        import cv2
        markImg = cv2.imread(args.markImg)
        pix2mm = compute.calPix2MM(markImg, args.markArea)
    else:
        if args.pix2mm is None:
            raise ValueError("either markImg and markArea or pix2mm must be provided")
        pix2mm = args.pix2mm

    compute.computeOneLength(args.imgPath, pix2mm, yolo, sam)