# Fourier Space Segmentation

This repository contains the code for the study "Fourier Space Segmentation" by Jeff Rhoades (@rhoadesScholar), accepted by the FAAFO Consortium of Rhoades.

## Goal

This is an observational study to determine the effect of 

## Findings

TBD

Below are thresholded masks of the GT targets and associated the model's output.

![Results](FourierSpaceSegmentation/qualitative_comparison.png)

## Setup

To install the required packages, run the following command from the root directory of this repository:

```bash
micromamba env create -n fss python==3.11 -f requirements.txt -c pytorch -c nvidia -y
micromamba activate fss
pip install -e .
```

## Model

Uses MONAI's implementation of a ViT Autoencoder based on [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

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

