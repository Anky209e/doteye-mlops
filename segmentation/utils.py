import torch
import torchvision
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Saves Current parameters of Model
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """
    Loads already saved parameters of model
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def mask_convert(mask):
    """
    Convert mask to image
    """
    mask = mask.clone().cpu().detach().numpy()
    mask = mask.transpose((1,2,0))
    mask = mask.clip(0,1)
    mask = np.squeeze(mask)
    return mask

def image_convert(image):
    """
    Convert tensor to image
    """
    image = image.clone().cpu().numpy()
    image = image.transpose((1,2,0))
    image = image.clip(0,1)
    image = np.squeeze(image)
    
    return image

def iou_(y_pred,y):
    """
    calculate IOU
    """
    inputs = y_pred.reshape(-1)
    targets = y.reshape(-1)
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
    smooth = 1    
    iou = (intersection + smooth)/(union + smooth)
    return iou

def iou_batch(y_pred,y):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.clone().cpu().detach().numpy()
    y = y.clone().cpu().detach().numpy() 
    
    for pred, label in zip(y_pred, y):
        ious.append(iou_(pred, label))
    iou = np.nanmean(ious)
    return iou  

def plot_and_save_images(real_image,pred_image,target_image,path):
    fig = plt.figure(figsize=(10,4))

    fig.add_subplot(1, 3, 1)
    plt.imshow(image_convert(real_image))
    plt.axis('on')
    plt.title("Image")

    fig.add_subplot(1, 3, 2)
    plt.imshow(mask_convert(pred_image))
    plt.axis('on')
    plt.title("Predicted Mask")

    fig.add_subplot(1, 3, 3)
    plt.imshow(mask_convert(target_image))
    plt.axis('on')
    plt.title("Actual Mask")

    plt.savefig(path)
    plt.close()



def check_accuracy(loader, model, device="cuda"):
    """
    Accuracy score based on pixel numbers
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    """
    Saves Mask Images
    """
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            # preds = (preds > 0.5).float()
            plot_and_save_images(
                real_image=x[0],
                pred_image=preds[0],
                target_image=y[0],
                path= f"{folder}/pred_{idx}.png"
            )
        # torchvision.utils.save_image(
        #     preds, f"{folder}/pred_{idx}.png"
        # )
        # torchvision.utils.save_image(y, f"{folder}/real_{idx}.png")
        

    model.train()