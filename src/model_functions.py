
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from typing import Optional

# --- Train and Test loops using torch datasets and dataloaders ---

def train_torch(model, device, optim, lossfn, loader: DataLoader):
    """Function to train given model for a single epoch with the given dataloader."""

    # Loop through the dataloader and train
    lossi = []
    dice_i = []
    model.train()
    for i, (X, y) in enumerate(loader):
        images = X.to(device)
        masks = y.to(device)

        # Get the prediction
        pred = model(images)['out']

        # Calculate the loss, propagate backwards, update parameters
        optim.zero_grad()
        loss = lossfn(pred, masks.type(torch.long))
        loss.backward()
        optim.step()
        lossi.append(loss.item())

        pred = np.argmax(pred.detach().cpu().numpy(), axis=1).astype(np.float32)
        masks = masks.detach().cpu().numpy().astype(np.float32)

        dice_i.extend(calc_dice(pred, masks))

        print(f"Batch {i}: {loss.item()}")

    print(f"Dice\nMin: {min(dice_i)}, Max: {max(dice_i)}, Stddev: {np.std(dice_i)}, Mean: {np.mean(dice_i)}")

    model.eval()
    return lossi


def model_test_torch(model, device, lossfn, loader: DataLoader):
    """Test the model on the given dataloader and loss function.  """

    dice_i = []
    model.eval()
    with torch.no_grad():
        # Get batch
        for X, y in loader:
            # Move the data to the device(should be GPU)
            images = X.to(device)
            masks = y.to(device)

            # Get the prediction
            pred = model(images)['out']

            # Calculate the loss, propagate backwards, update parameters
            loss = lossfn(pred, masks.type(torch.long))

            pred = np.argmax(pred.detach().cpu().numpy(), axis=1).astype(np.float32)
            masks = masks.detach().cpu().numpy().astype(np.float32)

            dice_i.extend(calc_dice(pred, masks))
    
    print(f"Dice\nMin: {min(dice_i)}, Max: {max(dice_i)}, Stddev: {np.std(dice_i)}, Mean: {np.mean(dice_i)}")
    print(f"Loss: {loss.item()}")

    return loss.item(), min(dice_i), max(dice_i), np.std(dice_i), np.mean(dice_i)

def calc_dice(pred, masks):

    intersection = np.logical_and(pred, masks)
    excl = np.logical_xor(pred, masks)

    dice = (2 * np.sum(intersection, axis=(1, 2))) / (2 * np.sum(intersection, axis=(1, 2)) + np.sum(excl, axis=(1, 2)))
    return dice.tolist()


class ImageSet(Dataset):
    """Dataset for the images and masks """
    
    def __init__(self, file_names: list, img_dir, mask_dir: Optional[str]=None, img_suf='.jpg', mask_suf='.png', img_transform=None, mask_transform=None) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suf = img_suf
        self.mask_suf = mask_suf
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        self.image_file_names = [f"{f.split('.')[0]}{self.img_suf}" for f in file_names]
        if self.mask_dir is not None:
            self.mask_file_names = [f"{f.split('.')[0]}{self.mask_suf}" for f in file_names]
    
    def __len__(self):
        return len(self.image_file_names)
    
    def __getitem__(self, index):
        # return super().__getitem__(index)
        img = self.img_transform(cv2.imread(self.img_dir + self.image_file_names[index]))
        if self.mask_dir is not None:
            # Get the target image
            target = self.mask_transform(cv2.imread(self.mask_dir + self.mask_file_names[index])).max(axis=0).values

            # Create a mask to hold the values
            mask = np.zeros(target.shape, np.float32)
            mask[target > 0.1] = 1

            # Convert mask to tensor and remove trivial axes
            mask = ToTensor()(mask).squeeze()
            return img, mask
        else:
            return img, self.image_file_names[index]


# --- Train and test loops using custom functions --


def get_image_mask_pair(filelist: list, data_path: str, ipt_trans, idx=None):
    """Get the image and mask(target) of a specific index."""

    filelist = sorted(filelist)
    if idx is None:
        idx = np.random.randint(0, len(filelist))

    image = cv2.imread(data_path + f'images/{filelist[idx]}')
    target = cv2.imread(data_path + f'masks/{filelist[idx][:-4]}.png').max(axis=2)
    mask = np.zeros(target.shape, np.float32)
    
    mask[target > 0.1] = 1

    image = ipt_trans(image)
    mask = ToTensor()(mask).squeeze()

    return image, mask, filelist[idx]


def get_batch(batch_size, filelist: list, image_height, image_width, data_path, ipt_trans):
    """Get a batch of images and masks(targets).  """

    images = torch.zeros((batch_size, 3, image_height, image_width))
    masks = torch.zeros((batch_size, image_height, image_width))
    file_names = [''] * 3

    for i in range(batch_size):
        images[i], masks[i], file_names[i] = get_image_mask_pair(filelist, data_path, ipt_trans)

    return images, masks, file_names


def train(model, device, optim, lossfn, batch_size, train_files, image_height, image_width):
    """Train model using custom example fetching functions."""

    model.train()

    # Get batch
    images, masks, _ = get_batch(batch_size, train_files, image_height, image_width)
    images = images.to(device)

    # Get the prediction
    pred = model(images)['out']

    # Calculate the loss, propagate backwards, update parameters
    optim.zero_grad()
    loss = lossfn(pred.to('cpu'), masks.type(torch.long))
    loss.backward()
    optim.step()
    model.eval()

    return loss.item()


def model_test(model, device, lossfn, batch_size, test_files):
    """Test the model using custom example fetching functions.  """

    model.eval()

    with torch.no_grad():
        # Get batch
        images, masks, file_names = get_batch(batch_size=batch_size, filelist=test_files)
        images = images.to(device)

        # Get the prediction
        pred = model(images)['out']

        # Calculate the loss, propagate backwards, update parameters
        loss = lossfn(pred.to('cpu'), masks.type(torch.long))

    return loss.item(), file_names
