Efficient Segmentation of Tracheal Structures from MRI Data
===
- Bachelor Thesis in Physics at Georg-August-University Göttingen
- a journal paper (probably in german) will soon be added here 

# Aims
[Prior work](http://ieeexplore.ieee.org/document/6046660/) on Segmentation of Tracheal Structures was able to give sufficent results in terms of DICE Coefficents, but lacking efficency. The goal of this work is to apply a [unet](https://arxiv.org/abs/1505.04597) to the segmentation problem and therby improving runtimes noticeably. 

# Data 
The used data originates from the SHIP-study in germany. Due to privacy rights it can't be published here. 

# Source
This repo is structured the following way. All of the main content is stored in the src dircetory. Which is divided into several modules. In networks.py the used network architectures where specified. In losses.py the dice coefficient as a custom loss function is defined. The files utilities.py and augmentation.py contain general utility-function and utilities for augmentation. The main files for this work are unet.py and scan.py, being the ones where a unet or a SCAN-architecture are trained respectively. 

# Results
In this work, DICE scores of up to 94.4% where achieved, while prior work was only able to achieve 91.8%. The main goal of increasing efficency was also achieved. Segmentation times where less then one second using a Nvidia Titan X GPU, while classical methods take more than a minute. 
