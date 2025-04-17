# LenRuler: Automated Radicle Length Measurement for Diverse Crops

Welcome to the LenRuler repository! This project introduces a novel, automated method for precisely
measuring radicle length, designed to be applicable across various crop species.

LenRuler leverages the power of the
[Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) to accurately
delineate the seed coat and germination region. To ensure precise segmentation, the approach
iteratively refines SAM's initial predictions by incorporating crucial information: bounding boxes
detected by YOLO and the centroid coordinates of the seeds.

Furthermore, LenRuler utilizes the unique Gaussian distribution characteristics of the seed coat's
color. This allows for the construction of a Gaussian model, enabling effective classification of
segmented components into the seed coat and the developing radicle.

Finally, the radicle length is determined by identifying the endpoint furthest from the seed coat's
skeleton and the nearest intersection point on the radicle.  The actual length measurement is then
computed from the geodesic distance between these two points, providing a robust and accurate
metric.

## Key Features:

- **Universal Application**: Designed for use across diverse crop species.
- **Precise Segmentation**: Employs SAM with iterative refinement for accurate seed coat and radicle
delineation.
- **Gaussian Color Modeling**: Leverages seed coat color properties for effective component
classification.
- **Geodesic Distance Measurement**: Calculates accurate radicle length using geodesic distance.

![Error loading image](resource/demo.gif "The demo has been sped up.")


## Quick Start

### Requirements
As for the algorithm, please make sure to have `PyTorch` installed before installing the following
packages. Most packages needed would be installed along with `ultralytics`. And SAM is imported as a
submodule from git. You could use SAM from `ultralytis`, but there would be much effort on code for
you.
```
conda create -n LenRuler python=3.12
# install pytorch according to your own cuda version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics
pip install networkx
pip install scikit-image

git clone https://github.com/cccccabbage/LenRuler --recursive
cd LenRuler/SAM
pip install -e .
cd ..
```

### Usage
Make sure to install all the required packages before running the codes. The YOLO model should be
```
python example.py --img <path to image> --yolo <path to YOLO model> --sam <path to SAM model> \
    --pix2mm <pixel to mm ratio>
```

Set `--pix2mm` to 1.0 will calcuate the pixel length.