# Local Occupancy Vectors for Semantic Segmentation

This repository contains the code for the study "Local Occupancy Vectors for Semantic Segmentation" by Jeff Rhoades (@rhoadesScholar), accepted by the FAAFO Consortium of Rhoades.

## Hypothesis

I hypothesize that a model that predicts the local occupancy matrix of a pixel in an image can be used to improve the performance of a semantic segmentation model. This is motivated by approaches such as Cellpose, StarDist, Local Shape Descriptors, and Flood-Filling Networks, which use local information to improve the performance of segmentation models. Distance and signed-distance embeddings, Cellpose, StarDist, and Local Shape Descriptors use information about the distances to the boundaries of objects to embed shape information into vectors for each pixel. Distance and signed-distance embeddings simply use the distance to the nearest boundary to embed shape information into a vector for each pixel. Cellpose embeds this information with x-y(-z) flows that are used to predict the boundaries of cells. StarDist uses distance transforms to predict the distances to the boundaries of objects along a specified number of directed rays. The reader is encouraged to read the papers for more information, especially for Local Shape Descripts and Flood-Filling Networks, proper summaries of which are beyond the scope of this work.

The conceptual motivation is that the local occupancy matrix of a pixel can be leveraged in post-processing to improve accuracy and robustness of segmentation by extracting consensus information about local object shapes from each voxel's embeddings. The local occupancy matrix is a binary matrix that represents the occupancy of a pixel's neighbors. The neighbors of a pixel are defined by a kernel, which can be a square, circle, or other shape.

## Findings

TBD

## Setup

To install the required packages, run the following command from the root directory of this repository:

```bash
micromamba env create -n LOV python==3.11 -f requirements.txt -c pytorch -c nvidia -y
micromamba activate LOV
pip install -e .
```

## Data

The data used in the paper is from "Instance segmentation of mitochondria in electron microscopy images with a generalist deep learning model" by Ryan Conrad and Kedar Narayan. See https://volume-em.github.io/empanada.html for more information. The data is not included in this repository, but can be downloaded from:
- https://doi.org/10.6019/EMPIAR-11037 (CEM-MitoLab)
- https://doi.org/10.6019/EMPIAR-11035 (CEM1.5M).
<!-- - https://doi.org/10.6019/EMPIAR-10982 (Seven benchmark datasets of instance segmentation of mitochondria) -->

It is faster to navigate to the websites above, but these datasets can also be downloaded and extracted to the `data` directory by running the following commands:

```bash
python data/download.py
```

This will download the data to the `data` directory, unzip it to the same directory, and randomly split the Mito-Lab data into training, validation, and test sets (80%, 15%, and 5%, respectively).

