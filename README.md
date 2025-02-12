# LenRuler

## Requirements
To try the codes yourself, one should have `PyTorch` installed before installing the following
packages. The `ultralytics` package would have most requiremented pacakges, like `opencv`,
installed.
```
pip install ultralytics
pip install networkx
pip install scikit-image
```

## Usage
Make sure to installed all the required packages before running the codes.
```
python compute.py --img <path to image> --yolo <path to YOLO model> --sam <path to SAM model>\
    --pix2mm <pixel to mm ratio>
```