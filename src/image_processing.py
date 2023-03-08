
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from os import listdir, remove, rmdir, mkdir, path


def model_application(model, device, loader: DataLoader, 
                      image_src: str, image_dest: str, mask_dest: str, 
                      skip_masks: bool=False,
                      orig_dim: tuple=(1920, 1080),
                      clear_mask_dest: bool=False, clear_image_dest: bool=False,
                      clean_up_mask_dest: bool=False, clean_up_image_dest: bool=False):
    """Apply the model to a given sequence or group of images.

    Args:
        model (torch.nn.Module?): Model to apply.
        device (str): Selection of cpu/gpu device to run model on
        loader (DataLoader): Dataset loader.
        image_src (str): Filepath of the source images.
        image_dest (str): Filepath of location for completed images.
        mask_dest (str): Filepath of location for completed masks.
        skip_masks (bool, optional): Option to skip masks if they have already been created. Defaults to False.
        orig_dim (tuple, optional): Original dimensions of the image, used to resize masks to original size.
        clear_mask_dest (bool, optional): Delete all pre-existing masks in specified directory. Defaults to False.
        clear_image_dest (bool, optional): Delete all pre-existing images in specified directory. Defaults to False.
        clean_up_mask_dest (bool, optional): Delete the completed masks in specified directory. Defaults to False.
        clean_up_image_dest (bool, optional): Delete the completed images in specified directory. Defaults to False.
    """

    # Generate the masks
    if not skip_masks:
        generate_masks(model=model,
                       device=device,
                       loader=loader,
                       mask_dest=mask_dest,
                       orig_dim=orig_dim,
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
                   orig_dim: tuple,
                   clear_dest: bool=False):
    """Use given model to generate segmentation predictions and save the results to the specified directory for later processing """
    
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

        # Resize the masks to the original image size
        pred_masks = np.dstack([cv2.resize(pred_masks[i].astype(np.float32), dsize=orig_dim) for i in range(pred_masks.shape[0])]).transpose((2, 0, 1))

        # Save the binary masks to disk
        for file_name, mask in zip(fn, pred_masks):
            cv2.imwrite(mask_dest + file_name[:-4] + '.png', mask)


def generate_output_images(mask_dest: str,
                           image_src: str,
                           image_dest: str,
                           clear_dest: bool=False,
                           row_buffer: int=50,
                           col_buffer: int=50):
    """Generate contours/outlines of the predictions and a bounding box on all predictions """

    # Create image destination if dne
    if not path.isdir(image_dest):
        mkdir(image_dest)
    # If the directory already exists then give ability to clear it
    elif clear_dest:
        for f in [f for f in listdir(image_dest) if f.endswith('.jpg')]:
            remove(image_dest + f)

    x = y = w = h = None
    # For each mask/image in the save location:
    for img_name in tqdm(sorted([f for f in listdir(image_src) if f.endswith('.jpg')])):
        
        # Grab an image and prediction mask
        img = cv2.imread(image_src + img_name)
        mask = np.uint8(cv2.imread(mask_dest + img_name[:-4] + '.png')[:, :, 0])

        # Ignore first frame as there is no existing prediction on the location.  
        # - Then only consider predicted pixels within a certain area around the previous region.
        # - If the first frame has erroneous areas then they will be bounded but will be corrected once there is a frame with only the region of interest
        if x is not None:
            mask_mask = np.zeros_like(mask)
            mask_mask[max(0, y - col_buffer):min(mask.shape[0], y + h + col_buffer), max(0, x - row_buffer):min(mask.shape[1], x + w + row_buffer)] = 1
            mask = np.multiply(mask, mask_mask)
        
        # Find contour/bounding box of prediction mask
        contours = cv2.findContours(mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[0] # List of contours(lists of points)
        rect = cv2.boundingRect(mask) 
        x, y, w, h = rect

        # Draw the contour and bounding box on the image
        img = cv2.drawContours(img, contours, contourIdx=-1, color=(118, 226, 22), thickness=3)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=3)
        
        # Save the image to a new location
        cv2.imwrite(image_dest + img_name, img)
