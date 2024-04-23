import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import config
from train import test_loader
from model import UNet
import random
from torchvision import transforms
from PIL import Image

list_images = sorted(os.listdir(config.IMAGE_DATASET_PATH))
images = [os.path.join(config.IMAGE_DATASET_PATH, x) for x in list_images]
list_masks = sorted(os.listdir(config.MASK_DATASET_PATH))
masks = [os.path.join(config.MASK_DATASET_PATH, x) for x in list_masks]
tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
    transforms.ToTensor()
])

def prepare_plot(original, mask, prediction):
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title('Original')
    plt.axis(False)
    axes[1].imshow(mask)
    axes[1].set_title('Ground Truth')
    plt.axis(False)
    axes[2].imshow(prediction)
    axes[2].set_title('Prediction')
    plt.axis(False)
    plt.show()
    
def predict(model, image):
    model.eval()
    with torch.no_grad():
        prediction = model(image)
    return prediction

if __name__ == '__main__':
    for i in range(10):
        image = random.choice(images)
        filename = os.path.split(image)[-1]
        mask = os.path.join(config.MASK_DATASET_PATH, filename)
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf(image)
        mask = cv2.imread(mask, 0)
        mask = tf(mask)
        model = UNet()
        model.load_state_dict(torch.load('./output/unet_tgs_salt.pth', map_location='cpu'))
        prediction = predict(model, image.unsqueeze(dim=0))
        prediction = (prediction >= config.THRESHOLD).type(torch.float32)
        prepare_plot(image.permute(1,2,0).detach().numpy(), mask.permute(1,2,0).detach().numpy(), prediction[0].permute(1,2,0).detach().numpy())