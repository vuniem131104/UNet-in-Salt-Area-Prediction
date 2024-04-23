from torch.utils.data import Dataset
import cv2 

class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):
        super().__init__()
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms
        
    def __len__(self):
        return len(self.imagePaths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.imagePaths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.maskPaths[idx], 0)
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        
        return (image, mask)
    