from torch.utils.data import Dataset,dataloader,random_split
from skimage import io,transform
import numpy as np
from torchvision import transforms,utils
from PIL import Image
import os
import matplotlib.pyplot as plt
from utils import image_convert,mask_convert

class MedicalDataset(Dataset):
    """
    Custom Dataset of All nuclie medical Images\n
    image_dir: path of image directory\n
    mask_dir: path of mask directory\n
    transform: data sugmentation (default=None)
    """

    def __init__(self,path,transform=None):
        self.path = path
        self.folders = os.listdir(path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip()
        ])
        

    def __len__(self):
        return len(self.folders)
    
    def __getitem__(self, idx):
        image_folder = os.path.join(self.path,self.folders[idx],'images/')
        mask_folder = os.path.join(self.path,self.folders[idx],'masks/')

        image_path = os.path.join(image_folder,os.listdir(image_folder)[0])

        img = Image.open(image_path).convert("RGB")
        img = img.resize((128,128))
        img = np.array(img,dtype=np.float32)
        mask = self.get_mask(mask_folder,128,128).astype('float32')
        mask = np.array(mask,dtype=np.float32)
        mask[mask != 0] = 1.0
        img = self.transform(img)
        mask = self.transform(mask)

        return (img,mask)

    def get_mask(self,mask_folder,IMG_HEIGHT, IMG_WIDTH):
        """
        Creating a single Mask image by combining all
        """
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1))
        for mask_ in os.listdir(mask_folder):
                mask_ = Image.open(os.path.join(mask_folder,mask_)).convert("L")
                mask_ = mask_.resize((128,128))
                # mask_ = io.imread(os.path.join(mask_folder,mask_))
                # mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
                mask_ = np.expand_dims(mask_,axis=-1)
                mask = np.maximum(mask, mask_)
            
        return mask

# if __name__=="__main__":
#     data = MedicalDataset("stage1_train")

#     image,mask = data[30]


#     tr = transforms.ToPILImage()
#     ig_1 = tr(image)
#     ig_1.show()
#     img = tr(mask)
#     img.show()
            