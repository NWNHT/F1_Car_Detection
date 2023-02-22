
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from os import listdir, remove, rmdir, mkdir, path


def model_application(model, device, loader: DataLoader, 
                      image_src: str, image_dest: str, mask_dest: str, 
                      skip_masks: bool=False,
                      clear_mask_dest: bool=False, clear_image_dest: bool=False,
                      clean_up_mask_dest: bool=False, clean_up_image_dest: bool=False):

    # Generate the masks
    if not skip_masks:
        generate_masks(model=model,
                       device=device,
                       loader=loader,
                       mask_dest=mask_dest,
                       clear_dest=clear_mask_dest)

    # Generate the output images
    generate_output_images(mask_dest=mask_dest,
                           image_src=image_src,
                           image_dest=image_dest,
                           clear_dest=clear_image_dest)

    # Give option to clean up all mask files and the directory
    if clean_up_mask_dest:
        for f in [f for f in listdir(mask_dest) if f.endswith('.png')]:
            remove(mask_dest + f)
        rmdir(mask_dest)
    if clean_up_image_dest:
        for f in [f for f in listdir(image_dest) if f.endswith('.jpg')]:
            remove(image_dest + f)
        rmdir(image_dest)
    

def generate_masks(model, device, loader: DataLoader,
                   mask_dest: str,
                   clear_dest: bool=False):
    
    # Create mask destination if dne
    if not path.isdir(mask_dest):
        mkdir(mask_dest)
    # If the directory already exists then give ability to clear it
    elif clear_dest:
        for f in [f for f in listdir(mask_dest) if f.endswith('.png')]:
            remove(mask_dest + f)

    # Loop through dataloader creating the predictions and saving to mask directory
    for X, fn in tqdm(loader):
        images = X.to(device)
    
        # Make the predictions
        pred = model(images)['out']
        
        # Process the predictions to get binary masks, this is an abuse of the fact that there are only two classes
        pred_masks = torch.argmax(pred, dim=1).detach().to('cpu').numpy()
        pred_masks[pred_masks > 0] = 255
        
        # Save the binary masks to disk
        for file_name, mask in zip(fn, pred_masks):
            cv2.imwrite(mask_dest + file_name[:-4] + '.png', mask)


def generate_output_images(mask_dest: str,
                           image_src: str,
                           image_dest: str,
                           clear_dest: bool=False):

    # Create image destination if dne
    if not path.isdir(image_dest):
        mkdir(image_dest)
    # If the directory already exists then give ability to clear it
    elif clear_dest:
        for f in [f for f in listdir(image_dest) if f.endswith('.jpg')]:
            remove(image_dest + f)

    # For each mask/image in the save location:
    for img_name in tqdm([f for f in listdir(image_src) if f.endswith('.jpg')]):
        
        # Grab an image and mask
        img = cv2.imread(image_src + img_name)
        mask = np.uint8(cv2.imread(mask_dest + img_name[:-4] + '.png')[:, :, 0])
        
        # Find contour/bounding box of mask
        # mask_thres = cv2.threshold(mask, 0.5, 1, type=cv2.THRESH_BINARY)
        contours = cv2.findContours(mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[0] # List of contours(lists of points)
        rect = cv2.boundingRect(mask) 
        # if the widths and location are reasonable, then use them
        # if 'x' not in vars() or (abs(rect[2] - w) < 200 and abs(rect[3] - h) < 200 and abs(rect[0] - x) < 200 and abs(rect[1] - y) < 200):
        x, y, w, h = rect

        # Draw the contour and bounding box on the image
        img = cv2.drawContours(img, contours, contourIdx=-1, color=(255, 0, 0), thickness=2)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=3)
        
        # Save the image to a new location
        cv2.imwrite(image_dest + img_name, img)

