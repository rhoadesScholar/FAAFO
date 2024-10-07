# Knowledge Expansion via Error Distribution Prediction

This repository contains the code for the future paper "Knowledge Expansion via Error Distribution Prediction" accepted at some conference hopefully.

## Setup

To install the required packages, run the following command:

```bash
micromamba create -n KE python=3.10 -y
micromamba env create -n KE -f requirements.txt -c pytorch -c nvidia -y
micromamba activate KE
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

This will download the data to the `data` directory, unzip it to the same directory, and randomly split the Mito-Lab data into training, validation, and test sets.

## Training
Several models are trained in the paper for the experiments. 5 seeds are used for each training run. The same seed is used for all teacher and student pairs.

### Teacher
In general:

The teacher model, a Critic from monai, is trained to predict the Binary cross-entropy (BCE) loss on 80% of CEM-MitoLab, using mean squared error (MSE) loss, validated on 15%, and tested on 5%. It receives the raw image data and label masks as input and predicts the BCE loss. Two different teacher conditions exist (see below):

#### Pretraining
The teacher model is pretrained using random augmentations of the ground truth masks (i.e. RandomErase, elastic deformations, dropout, etc.). The pretrained model is saved to `models/teacher_pretrained_{seed}.pth`.

#### Joint-Training
During joint-training, the pretrained teacher model is further trained to predict the BCE loss of a student model, alongside training on the ground truth masks of the same data. This model is saved to `models/teacher_joint_{seed}.pth`.

### Student
In general:

The student model, a ViT-based autoencoder from monai, is generally trained with BCE loss on 80% of CEM-MitoLab, validated on 15%, and tested on 5%. This model recieves raw image data as input, and outputs a prediction for the label mask. Three different student conditions exist (see below):
1) Baseline
2) Teacher joint-training
3) Knowledge expansion

#### Baseline
This student model is trained from scratch with BCE loss on 80% of CEM-MitoLab, validated on 15%, and tested on 5%. The student model is saved to `models/student_baseline_{seed}.pth`.

#### Teacher Joint-Training
The student model is trained from scratch alongside the teacher model, while using the actual BCE loss from the groundtruth masks. The student model is saved to `models/student_joint_{seed}.pth`.

#### Knowledge Expansion
The joint-trained student model is further trained with the teacher model's BCE loss prediction on the entire CEM1.5 (CEM1500k_unlabelled) dataset. The student model is saved to `models/student_expanded_{seed}.pth`.

# Optimizer and Learning Rate Scheduler
The same optimizer and learning rate scheduler are used for all models. The models are trained for 300 epochs with a batch size of 16. The learning rate is reduced by a factor of 0.1 every 50 epochs.

# Augmentations
...