import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
import torchvision
from tqdm import tqdm
from model import UNET
from dataset import MedicalDataset
from utils import save_checkpoint,load_checkpoint,check_accuracy,save_predictions_as_imgs,iou_,iou_batch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/seg_1')

# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
DATA_PATH = "stage1_train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def add_image_to_tb(loader,model,device="cuda"):

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))

            preds = (preds > 0.3).float()
        pred_grid = torchvision.utils.make_grid(preds)
        real_grid = torchvision.utils.make_grid(y)

        writer.add_image("Predicted",pred_grid)
        writer.add_image("Real",real_grid)


class DiceBCELoss(nn.Module):
    """
    BCE loss with Dice score for segmentation
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
              
        bce_weight = 0.5
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        loss_final = BCE * bce_weight + dice_loss * (1 - bce_weight)
        return loss_final

def train_step(loader,model,optimizer,loss_fn,scaler):
    """
    Function for Updating weights and train one step\n
    loader: Dataloader\n
    model: Network\n
    optmizer: Optimizer function\n
    loss_fn: loss function to perform step\n
    scaler: grad scaler
    """

    loop = tqdm(loader)

    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(device=DEVICE,dtype=torch.float)
        targets = targets.to(device=DEVICE,dtype=torch.float)

        # forward pass
        
        predictions = model(data)
        loss = loss_fn(predictions,targets)
        score = iou_batch(predictions,targets)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        writer.add_scalar("Loss/step",loss.item(),batch_idx)
        writer.add_scalar("Score/step",score,batch_idx)
        # Tqdm loop update
        loop.set_postfix(loss=loss.item(),iou=score)
    return loss,score


def main():
    """
    Main Function for training
    """
    model = UNET(in_channels=3,out_channels=1).to(device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

    data = MedicalDataset(DATA_PATH)
    trainset, valset = random_split(data, [600, 70])

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
        )
    val_loader = DataLoader(
        dataset=valset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
        )
    
    # loading checkpoint and accuracy check function 
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):

        loss,score = train_step(train_loader,model,optimizer,loss_fn,scaler)

        writer.add_scalar("Loss/epoch",loss.item(),epoch)
        writer.add_scalar("Score/epoch",score,epoch)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        add_image_to_tb(val_loader,model,device=DEVICE)
        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images", device=DEVICE
        )

        

if __name__=="__main__":
    main()



