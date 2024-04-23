from dataset import SegmentationDataset
import config
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import UNet
from torch import optim, nn
import torch
import matplotlib.pyplot as plt

list_images = sorted(os.listdir(config.IMAGE_DATASET_PATH))
images = [os.path.join(config.IMAGE_DATASET_PATH, x) for x in list_images]
list_masks = sorted(os.listdir(config.MASK_DATASET_PATH))
masks = [os.path.join(config.MASK_DATASET_PATH, x) for x in list_masks]
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=config.TEST_SPLIT, random_state=42)
tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
    transforms.ToTensor()
])
train_dataset = SegmentationDataset(X_train, y_train, tf)
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=config.PIN_MEMORY)
test_dataset = SegmentationDataset(X_test, y_test, tf)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=config.PIN_MEMORY)
unet = UNet().to(config.DEVICE)
optimizer = optim.Adam(unet.parameters(), lr=config.INIT_LR)
loss_fn = nn.BCEWithLogitsLoss()

if __name__ == '__main__':
    train_losses = []
    test_losses = []
    for epoch in range(config.NUM_EPOCHS):
        total_train_loss, total_test_loss = 0.0, 0.0
        unet.train()
        for (x, y) in train_loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            pred = unet(x)
            train_loss = loss_fn(pred, y)
            total_train_loss += train_loss.item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        mean_train_loss = total_train_loss / len(train_loader)
        train_losses.append(mean_train_loss)
        unet.eval()
        with torch.inference_mode():
            for (x, y) in test_loader:
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                pred = unet(x)
                test_loss = loss_fn(pred, y)
                total_test_loss += test_loss.item()
                
        mean_test_loss = total_test_loss / len(test_loader)
        test_losses.append(mean_test_loss)
        print(f"Epoch: {epoch + 1}, train loss: {mean_train_loss:.4f}, test loss: {mean_test_loss:.4f}")
    
    epochs = list(range(0, config.NUM_EPOCHS))    
    plt.plot(epochs, train_losses, c='r', label='train_loss')
    plt.plot(epochs, test_losses, c='b', label='test_loss')
    plt.legend()
    plt.savefig(config.PLOT_PATH)
    plt.show()
    torch.save(unet.state_dict(), config.MODEL_PATH)