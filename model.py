import torch 
from torch import nn 
from torchvision import transforms
import config

class Block(nn.Module):
    def __init__(self, inchannels, outchannels):
        super().__init__()
        self.conv1 = nn.Conv2d(inchannels, outchannels, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outchannels, outchannels, 3)
        
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))
    
class Encoder(nn.Module):
    def __init__(self, channels=(3,16,32,64)):
        super().__init__()
        self.encBlocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        blockOutputs = []
        for block in self.encBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        return blockOutputs
    
class Decoder(nn.Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels)-1)])
        self.decBlocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        
    def forward(self, x, encFeatures):
        for i in range(len(self.channels)-1):
            x = self.upconvs[i](x)
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.decBlocks[i](x)
        return x
    
    def crop(self, encFeatures, x):
        _, _, H, W = x.shape
        encFeatures = transforms.CenterCrop([H, W])(encFeatures)
        return encFeatures
    
class UNet(nn.Module):
    def __init__(self, enchannels=(3, 16, 32, 64), dechannels=(64,32,16), 
                 retainDim=True, num_class=config.NUM_CLASSES,
                 outsize=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):
        super().__init__()
        self.encoder = Encoder(enchannels)
        self.decoder = Decoder(dechannels)
        self.head = nn.Conv2d(dechannels[-1], num_class, 1)
        self.retainDim = retainDim
        self.outsize = outsize
        
    def forward(self, x):
        encFeatures = self.encoder(x)
        decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])
        map = self.head(decFeatures)
        
        if self.retainDim:
            map = torch.nn.functional.interpolate(map, self.outsize)
            
        return map