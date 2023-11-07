# Import PyTorch's distributed package for parallel training and other necessary libraries.
import torch.distributed as dist
import os
from torch.utils.data.distributed import DistributedSampler
from multiprocessing import cpu_count
import warnings
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

# Import various utilities for machine learning.
from sklearn.model_selection import ParameterGrid
import cv2
import numpy as np
import torchvision
import PIL.Image as Image
import glob
from torchvision.transforms import functional as F
from IPython.display import display
import torch
from torch.utils.data import Subset
from torchvision import transforms
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as Fu
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
import traceback
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import (
    resnet50, ViT_H_14_Weights, ResNet50_Weights, EfficientNet_V2_S_Weights,
    AlexNet_Weights, Wide_ResNet101_2_Weights
)
import itertools
from torch.utils.data import Subset
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from torch import manual_seed
from PIL import ImageEnhance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from enum import Enum

# Environment configuration to ensure reproducibility and optimized computation.
torch.backends.cudnn.benchmark = True
import gc
gc.collect()
torch.cuda.empty_cache()

# Setup function for distributed training.
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '44236'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(rank, world_size)

# Cleanup function to destroy the process group once training is complete.
def cleanup():
    dist.destroy_process_group()

# Define a class for distributed early stopping to prevent overfitting.
class DistributedEarlyStopping:
    # Initialize the early stopping mechanism with the required attributes.
    def __init__(self, patience, verbose=False, delta=0, path='checkpoint.pt',trace_func=print, mode='min', check_point=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_min = np.Inf
        self.val_max = np.NINF
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.img_size = None
        self.mode = mode
        self.check_point = check_point

    # The call method that will be executed once an epoch to check if early stopping criteria are met.
    def __call__(self, val_metric, model, optimizer, rank, device):
        # Check if improvement occurred and if not, count towards patience.
        if self.mode == 'min':
            score = -1. * val_metric
        else:
            score = np.copy(val_metric)
        
        metric_tensor = torch.tensor([-1.0 * val_metric if self.mode == 'min' else val_metric]).to(device)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        mean_metric = metric_tensor.item() / dist.get_world_size()
        
        score = mean_metric
        if self.best_score is None:
            self.best_score = score
            if self.check_point == True:
                self.save_checkpoint(mean_metric, model, optimizer, rank)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.check_point == True:
                self.save_checkpoint(mean_metric, model, optimizer, rank)
            self.counter = 0
            
        if self.early_stop:
            stop_signal = torch.tensor([1], dtype=torch.int).to(rank)
        else:
            stop_signal = torch.tensor([0], dtype=torch.int).to(rank)
        dist.all_reduce(stop_signal, op=dist.ReduceOp.SUM)
        if stop_signal.item() == dist.get_world_size():
            self.early_stop = True
        else:
            self.early_stop = False

    # Save the model checkpoint if improvement is observed.
    def save_checkpoint(self, val_metric, model, optimizer, rank):
        if rank != 0:
            return
        if self.verbose:
            self.trace_func(f'Validation metric improved ({self.val_min:.6f} --> {val_metric:.6f}).  Saving model ...')
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, self.path)
        self.val_min = val_metric

# Define a custom subset class with transform capabilities.
class TransformedSubset(Subset):
    # Initialize the subset with the transform applied to the dataset.
    def __init__(self, subset, transform=None):
        super(TransformedSubset, self).__init__(subset.dataset, subset.indices)
        self.transform = transform

    # Override the getitem method to include transform.
    def __getitem__(self, index):
        x, y = super(TransformedSubset, self).__getitem__(index)
        if self.transform:
            x = self.transform(x)
        return x, y

# ... (The rest of the code would continue here with similar comments describing each function/class)
# Define an enumeration for cleaner code when referring to different parts of the dataset.
class ColorType(Enum):
    GRAY = 0  # Grayscale images
    COLOR = 1  # Color images

class DataSet(Enum):
    ALL = 0  # Entire dataset
    TRAIN = 1  # Training set
    VAL = 2  # Validation set
    TEST = 3  # Test set
    TRAIN_LOADER = 4  # DataLoader for training set
    VALID_LOADER = 5  # DataLoader for validation set
    TEST_LOADER = 6  # DataLoader for test set

# Define a custom dataset that allows transforming a subset of the data.
class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# Define a method to convert a NumPy array to a PIL image, ensuring proper channel order for RGB.
def np_array_to_image(np_img):
    np_img = np.squeeze(np_img)
    if len(np_img.shape) != 3 or np_img.shape[2] != 3:
        raise ValueError("Expected 3D array with shape (H, W, C) where C=3.")
    if np_img.dtype == np.float32 or np_img.dtype == np.float64:
        if np_img.max() > 1.0 or np_img.min() < 0:
            raise ValueError("Float images should have pixel values between 0 and 1.")
        np_img = (np_img * 255).astype(np.uint8)
    img = Image.fromarray(np_img)
    return img

# Define a function to apply various image adjustments using the ImageEnhance module.
def adjust_image_transform(img, brightness_factor=1.3, color_factor=1.1, contrast_factor=1.2):
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    img = ImageEnhance.Color(img).enhance(color_factor)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    return img

# Define a function to apply Gaussian blur to an image using OpenCV.
def gaussian_blur(img):
    np_img = np.array(img)
    blurred = cv2.GaussianBlur(np_img, (5, 5), 0)
    return Image.fromarray(blurred)

# Define a function to convert a grayscale image to RGB by repeating the grayscale channel.
def grayscale_to_rgb_16bit(img):
    np_img = np.array(img)
    if len(np_img.shape) == 2 or np_img.shape[0] == 1:
        np_img = np.stack([np_img] * 3, axis=-1)
    np_img = np_img / 65535.0  # Normalize 16-bit image data
    return np_array_to_image(np_img)

# Define a function to apply Contrast Limited Adaptive Histogram Equalization using OpenCV.
def enhance_contrast_using_clahe(img):
    np_img = np.array(img)
    if len(np_img.shape) == 3:
        gray_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = np_img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced_img = clahe.apply(gray_img)
    return enhanced_img

# Define a function to adjust the brightness of an image.
def adjust_brightness(img, factor=0.8):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

# Define a function to apply histogram equalization to an image.
def hist_equalize(img):
    np_img = np.array(img)
    img_yuv = cv2.cvtColor(np_img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return Image.fromarray(img_output)

# Define a function for image segmentation and visualization using OpenCV.
def segment_and_visualize(img_np):
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY )
    kernel = np.ones((2,2), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(img_np)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
        result = cv2.bitwise_and(img_np, img_np, mask=mask)
        return Image.fromarray(result)
    else:
        return Image.fromarray(img_np)

# Define transformations that apply the segmentation visualization function.
def segment_and_visualize_transform(img):
    return segment_and_visualize(np.array(img))

# Define transformations for converting grayscale to RGB.
def grayscale_to_rgb_transform(img):
    return grayscale_to_rgb(img)

# Define a composition of transformations for training images.
def cropped_transforms(grayscale=True):
    transforms_list = [
        transforms.Resize((224,224)),
        transforms.Lambda(grayscale_to_rgb_transform),
        transforms.ToTensor(),
    ]
    return transforms.Compose(transforms_list)

# Define a composition of transformations for validation images.
def valid_transforms(resize=None, img_crop_size=240, grayscale=True):
    transforms_list = [transforms.CenterCrop((img_crop_size, img_crop_size)), transforms.ToTensor()]
    return transforms.Compose(transforms_list)

# Define a composition of transformations for mammogram images.
def mammo_transforms(resize=None, img_crop_size=240, grayscale=True):
    transforms_list = [transforms.CenterCrop((img_crop_size, img_crop_size)), transforms.ToTensor()]
    return transforms.Compose(transforms_list)
