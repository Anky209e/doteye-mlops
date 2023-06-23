import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision
import numpy as np

class DoubleConv(nn.Module):
    """
    Double convolutional block of UNET\n
    """
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()

        # Double Conv block with Batch normalisation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        x = self.conv(x)
        return x


class UNET(nn.Module):
    """
    Custom UNET Model for Biomedical Image Segmentation purpose\n
    Uses Conv blocks from DoubleConv
    """

    def __init__(
            self,in_channels=1,out_channels=1,features=[64,128,256,512] # Features describe image size layers
            ):
        super(UNET,self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        # Vertical or Down layers of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels = feature
        
        # Horizontal of UP part of UNET
        for feature in reversed(features):
            # Upscaling the image
            self.ups.append(
                nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2)
            )
            # Adding two convs
            self.ups.append(
                DoubleConv(feature*2,feature)
            )

        # Bottleneck Layer
        self.bottleneck = DoubleConv(features[-1],features[-1]*2)

        # Last layer for changing color channel
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self,x):
        """
        Forward method defining flow of model
        """

        # skip_connections are left to right connection
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        # reversing for resolution
        skip_connections = skip_connections[::-1]

        # we are choosing step=2 because of doubleconv
        for idx in range(0,len(self.ups),2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # Resizing image if it is different size
            if x.shape != skip_connection.shape:
                x = TF.resize(x,size=skip_connection.shape[2:])
            
            # Concatnating skip connection with org image
            concat_skip = torch.cat((skip_connection,x),dim=1)

            x = self.ups[idx+1](concat_skip)
            

        return self.final_conv(x)


def test_input():
    x = torch.randn((3,1,160,160))
    model = UNET(in_channels=1,out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

