## Introduction 
- Use a deep learning model in pytorch to deal with Salt Area Prediction
## Dataset
- I use the [TGS Salt Segmentation dataset](https://www.kaggle.com/c/tgs-salt-identification-challenge). The dataset was introduced as part of the TGS Salt Identification Challenge on Kaggle
- I use a sub-part of this dataset which comprises 4000 images of size 101×101 pixels, taken from various locations on earth. Here, each pixel corresponds to either salt deposit or sediment. In addition to images, i am also provided with the ground-truth pixel-level segmentation masks of the same dimension as the image
![image](https://github.com/vuniem131104/UNet-in-Salt-Area-Prediction/assets/124224840/afbb71e5-2b3d-4ea4-8913-18e1798e278a)
- The white pixels in the masks represent salt deposits, and the black pixels represent sediment. I aim to correctly predict the pixels that correspond to salt deposits in the images. Thus, I have a binary classification problem where I have to classify each pixel into one of the two classes, Class 1: Salt or Class 2: Not Salt (or, in other words, sediment).
## Model
- I use UNet model and here is its architecture:
  ![image](https://github.com/vuniem131104/UNet-in-Salt-Area-Prediction/assets/124224840/bfd3cea6-9ef8-4091-92ed-096b645abd68)
## Train
- The model is trained for 40 epochs on CPU only (saved in output/unet_tgs_salt.pth) and if you want to train more, you can set up EPOCHS in config.py
