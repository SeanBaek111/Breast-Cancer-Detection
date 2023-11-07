import torch.distributed as dist
#!pip install wandb -q
import os
from torch.utils.data.distributed import DistributedSampler
from multiprocessing import cpu_count
#os.environ["WANDB_NOTEBOOK_NAME"] = "MultiClassifier.ipynb"
import warnings
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from sklearn.model_selection import ParameterGrid
import cv2
import numpy as np
import torchvision
import PIL.Image as Image
import glob
from torchvision.transforms import functional as F
#from torchvision.transforms.v2 import functional as F
from IPython.display import display
#import wandb
from datetime import datetime
import gc
import time
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
from torch.utils.data import DataLoader,Dataset, random_split
from torchvision import datasets, transforms, models
import traceback


import numpy as np
 
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models import resnet50, ViT_H_14_Weights, ResNet50_Weights, EfficientNet_V2_S_Weights, AlexNet_Weights, Wide_ResNet101_2_Weights
import itertools
from torch.utils.data import Subset

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from torch import manual_seed
from PIL import ImageEnhance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from enum import Enum
import os
import shutil
from sklearn.model_selection import train_test_split


torch.backends.cudnn.benchmark = True
import gc
#!del variable_name
gc.collect()
torch.cuda.empty_cache()
 
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '44236'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(rank, world_size)
   # dist.init_process_group("gloo", rank=rank, world_size=world_size)
def cleanup():
    dist.destroy_process_group()



import torch.distributed as dist

class DistributedEarlyStopping:
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

    def __call__(self, val_metric, model, optimizer, rank, device):
        # ... (생략) 여기에 기존의 __call__ 메서드의 일부 코드를 그대로 사용합니다.
        if self.mode == 'min':
            score = -1. * val_metric
        else:
            score = np.copy(val_metric)
        
        # 현재 프로세스의 metric 값을 모든 프로세스와 공유합니다.
        metric_tensor = torch.tensor([-1.0 * val_metric if self.mode == 'min' else val_metric]).to(device)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        mean_metric = metric_tensor.item() / dist.get_world_size()
        
        # 이제 mean_metric 값을 이용하여 early stopping을 수행합니다.
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
        
         # 모든 프로세스에서 stop_signal을 공유합니다.
        dist.all_reduce(stop_signal, op=dist.ReduceOp.SUM)
        
        # 모든 프로세스가 early stopping을 원하면 학습을 종료합니다.
        if stop_signal.item() == dist.get_world_size():
            self.early_stop = True
        else:
            self.early_stop = False
        
            
    def save_checkpoint(self, val_metric, model, optimizer, rank):
        if rank != 0:
            return
        '''Saves model when metric improves.'''
        if self.verbose:
            if self.mode == 'min':
                self.trace_func(f'Validation metric decreased ({self.val_min:.6f} --> {val_metric:.6f}).  Saving model ...')
            else:
                self.trace_func(f'Validation metric increased ({self.val_max:.6f} --> {val_metric:.6f}).  Saving model ...')
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, self.path)
        
        if self.mode == 'min':
            self.val_min = val_metric
        else:
            self.val_max = val_metric


class TransformedSubset(Subset):
    def __init__(self, subset, transform=None):
        super(TransformedSubset, self).__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, index):
        x, y = super(TransformedSubset, self).__getitem__(index)
        if self.transform:
            x = self.transform(x)
        return x, y

    
import numpy as np
import torch
from torch.utils.data import Subset

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt',check_point=True, trace_func=print, mode='min'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the quantity
                        monitored has stopped decreasing; in 'max' mode it will stop when the
                        quantity monitored has stopped increasing.
        """
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

    def __call__(self, val_metric, model, optimizer):
        if self.mode == 'min':
            score = -1. * val_metric
        else:
            score = np.copy(val_metric)
        
        if self.best_score is None:
            self.best_score = score
            if self.check_point == True:
                self.save_checkpoint(val_metric, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.check_point == True:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.check_point == True:
                self.save_checkpoint(val_metric, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_metric, model, optimizer):
        '''Saves model when metric improves.'''
        if self.verbose:
            if self.mode == 'min':
                self.trace_func(f'Validation metric decreased ({self.val_min:.6f} --> {val_metric:.6f}).  Saving model ...')
            else:
                self.trace_func(f'Validation metric increased ({self.val_max:.6f} --> {val_metric:.6f}).  Saving model ...')
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, self.path)
        if self.mode == 'min':
            self.val_min = val_metric
        else:
            self.val_max = val_metric


class TransformedSubset(Subset):
    def __init__(self, subset, transform=None):
        super(TransformedSubset, self).__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, index):
        x, y = super(TransformedSubset, self).__getitem__(index)
        if self.transform:
            x = self.transform(x)
        return x, y


class CenterCropPercentage:
    def __init__(self, percentage):
        assert 0 < percentage <= 1, "Percentage should be between 0 and 1"
        self.percentage = percentage

    def __call__(self, img):
     #   print("CenterCropPercentage is called!")
        width, height = img.size
        new_width, new_height = width * self.percentage, height * self.percentage
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        cropped_img = img.crop((left, top, right, bottom))
       # print("Cropped Image Size:", cropped_img.size)
        return cropped_img

    def __repr__(self):
        return self.__class__.__name__ + '(percentage={0})'.format(self.percentage)




class ColorType(Enum):
    GRAY = 0
    COLOR = 1

class DataSet(Enum):
    ALL   = 0
    TRAIN = 1
    VAL   = 2
    TEST  = 3
    TRAIN_LOADER = 4
    VALID_LOADER = 5
    TEST_LOADER = 6
   


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


 

def np_array_to_image(np_img):
    # Squeeze unnecessary singleton dimensions
    np_img = np.squeeze(np_img)

    # Ensure the NumPy array is 3D: (H, W, C)
    if len(np_img.shape) != 3:
        raise ValueError("Array needs to be 3D, got shape {}".format(np_img.shape))

    # Ensure channel order is correct
    if np_img.shape[2] != 3:
        raise ValueError("Last dimension should be 3 (for R, G, B), got {}".format(np_img.shape[2]))
 
    # Convert float images
    if np_img.dtype == np.float32 or np_img.dtype == np.float64:
        # Ensure the values are between 0 and 1
        if np_img.max() > 1.0 or np_img.min() < 0:
            raise ValueError("Float images should have pixel values between 0 and 1.")
        # Convert to uint8 [0, 255]
        np_img = (np_img * 255).astype(np.uint8)

    # Convert to PIL Image
    img = Image.fromarray(np_img)

    return img

def adjust_image_transform(img, brightness_factor=1.3, color_factor=1.1, contrast_factor=1.2):
    """Adjust brightness, color (saturation), and contrast of an image using torchvision transform.
    
    Args:
        img (PIL.Image): Input image.
        brightness_factor (float): Factor to adjust brightness. Default is 1.4.
        color_factor (float): Factor to adjust color (saturation). Default is 1.1.
        contrast_factor (float): Factor to adjust contrast. Default is 1.5.

    Returns:
        PIL.Image: Adjusted image.
    """
   # img = ImageEnhance.Brightness(img).enhance(brightness_factor)
   # img = ImageEnhance.Color(img).enhance(color_factor)
   # img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    return img

def gaussian_blur(img):
    np_img = np.array(img)
    blurred = cv2.GaussianBlur(np_img, (5, 5), 0)
    return Image.fromarray(blurred)

def grayscale_to_rgb_16bit(img):
    np_img = np.array(img)
 #   print("grayscale_to_rgb")
    # Check if the image is grayscale
    if len(np_img.shape) == 2 or np_img.shape[0] == 1:
        np_img = np.stack([np_img] * 3, axis=-1)
    elif np_img.shape[0] == 3:  # Move the channel dimension to the end
        np_img = np.transpose(np_img, (1, 2, 0))

 #   print("gray 22222")
    # Normalize here, if not done in __getitem__
    np_img = np_img / 65535.0  # assuming the original dtype was uint16
   # print("gray 3333")
    return np_array_to_image(np_img)

def grayscale_to_rgb(img):
    np_img = np.array(img)
    if len(np_img.shape) == 2:  # Check if the image is grayscale
        np_img = np.stack([np_img]*3, axis=-1)
    return Image.fromarray(np_img)
  
def enhance_contrast_using_clahe(img):
    np_img = np.array(img)
    
    # Check if the image is already grayscale
    if len(np_img.shape) == 2 or np_img.shape[2] == 1:
        gray_img = np_img
    else:
        gray_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced_img = clahe.apply(gray_img)
    
    # If you want to return a 3-channel image instead of 1-channel
    # return cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)
    
    return enhanced_img

def adjust_brightness(img, factor=0.8):
    """Adjust brightness of an image.
    
    Args:
        img (PIL.Image): Input image.
        factor (float): A value between 0 and 1 that determines the brightness.
                        1 means no change, less than 1 dims the image.

    Returns:
        PIL.Image: Image with adjusted brightness.
    """
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def hist_equalize(img):
    np_img = np.array(img)
    img_yuv = cv2.cvtColor(np_img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return Image.fromarray(img_output)
 
def segment_and_visualize(img_np):
    # Convert to grayscale if it's a color image
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY )
    kernel = np.ones((2,2), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Check if any contours were found
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(img_np)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
        result = cv2.bitwise_and(img_np, img_np, mask=mask)
        return Image.fromarray(result)
    else:
        print("Warning: No contours found in image")
        # Handle the case appropriately: you might return the original image,
        # or you might return a black image of the same shape, etc.
        return Image.fromarray(img_np)  # or whatever is appropriate in context

    
def segment_and_visualize_transform(img):
    return segment_and_visualize(np.array(img))

def grayscale_to_rgb_transform(img):
    return grayscale_to_rgb(img)
   
def cropped_transforms(grayscale=True):
    print("cropped_transforms called")
    transforms_list = [
        transforms.Resize((224,224)),
         
       # transforms.Lambda(gaussian_blur),
     #   transforms.Lambda(segment_and_visualize_transform),
       # transforms.Lambda(enhance_contrast_using_clahe),
        transforms.Lambda(grayscale_to_rgb_transform),
        transforms.ToTensor()
    ]
    # if grayscale:
    #     mean = [0.456]
    #     std = [0.224]
    # else:
    #     mean = [0.485, 0.456, 0.406]
    #     std = [0.229, 0.224, 0.225]
    # transforms_list.append(transforms.Normalize(mean, std))
    return transforms.Compose(transforms_list)

def valid_transforms(resize=None, img_crop_size = 240, grayscale=True):
     #
    transforms_list = []
    if resize is not None:
        transforms_list.append(transforms.Resize((resize, resize)))
        img_crop_size = int(resize * 0.86)
        
  #  print("img_crop_size",img_crop_size)
    
   
    transforms_list.extend([
       transforms.CenterCrop((img_crop_size, img_crop_size)),
       # transforms.RandomHorizontalFlip(),
       # transforms.RandomVerticalFlip(),
        transforms.Lambda(segment_and_visualize_transform),
        transforms.Lambda(enhance_contrast_using_clahe),
    #    transforms.Lambda(grayscale_to_rgb_transform),
        transforms.ToTensor()
    ])
    
    
    if grayscale:
        mean = [0.456]
        std = [0.224]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    transforms_list.append(transforms.Normalize(mean, std))
    return transforms.Compose(transforms_list)

def mammo_transforms(resize=None, img_crop_size=240, grayscale=True):
    #
    transforms_list = []
    if resize is not None:
        transforms_list.append(transforms.Resize((resize, resize)))
        img_crop_size = int(resize * 0.86)
    
    
    print("img_crop_size",img_crop_size)
   
    transforms_list.extend([
        transforms.CenterCrop((img_crop_size, img_crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(segment_and_visualize_transform),
        transforms.Lambda(enhance_contrast_using_clahe),
    #    transforms.Lambda(grayscale_to_rgb_transform),
        transforms.ToTensor()
    ])
    
    if grayscale:
        mean = [0.456]
        std = [0.224]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    transforms_list.append(transforms.Normalize(mean, std))
    return transforms.Compose(transforms_list)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = Fu.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        smooth = 1e-5
        y_pred = torch.sigmoid(y_pred)
        
        # Ensure y_true is one-hot encoded
        if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
            y_true = F.one_hot(y_true.squeeze().long(), num_classes=y_pred.shape[1]).float()
        elif len(y_true.shape) == 4:  # Handling potential 4D tensor
            y_true = y_true.permute(0, 3, 1, 2).float()

        tp = (y_true * y_pred).sum()
        fn = (y_true * (1-y_pred)).sum()
        fp = ((1-y_true) * y_pred).sum()
        tversky_index = tp / (tp + self.alpha*fn + self.beta*fp + smooth)
        return 1 - tversky_index
    
# class DiceLoss(nn.Module):
#     def forward(self, y_pred, y_true):
#         smooth = 1e-5
         
            

#         # If y_true is not already one-hot, make it one-hot
#         if len(y_true.shape) == 1 or y_true.shape[1] == 1:  # If it’s not one-hot
#             y_true_ohe = F.one_hot(y_true.to(torch.int64), num_classes=y_pred.size(1))
#         else:  # If it’s already one-hot
#             y_true_ohe = y_true
        
#         y_true_ohe = y_true_ohe.float()
#         y_pred = torch.sigmoid(y_pred)

        
       
#         # Calculating the loss
#         intersection = (y_pred * y_true_ohe).sum()
#         dice_coef = (2. * intersection + smooth) / (y_pred.sum() + y_true_ohe.sum() + smooth)
#         return 1 - dice_coef
class DiceLoss(nn.Module):
    def forward(self, y_pred, y_true):
        smooth = 1e-5
         
        # If y_true is not already one-hot, make it one-hot
        if len(y_true.shape) == 1 or y_true.shape[1] == 1:  # If it’s not one-hot
            y_true_ohe = torch.nn.functional.one_hot(y_true.to(torch.int64), num_classes=y_pred.size(1))
        else:  # If it’s already one-hot
            y_true_ohe = y_true
        
        y_true_ohe = y_true_ohe.float()
        y_pred = torch.sigmoid(y_pred)

        # Calculating the loss
        intersection = (y_pred * y_true_ohe).sum()
        dice_coef = (2. * intersection + smooth) / (y_pred.sum() + y_true_ohe.sum() + smooth)
        return 1 - dice_coef

        
class CostSensitiveLoss(nn.Module):
    def __init__(self, fn_cost=2.0, fp_cost=1.0):
        super(CostSensitiveLoss, self).__init__()
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost

    def forward(self, outputs, targets):
        # 확률을 구하기 위한 softmax
        probs = nn.Softmax(dim=1)(outputs)
        positive_probs = probs[:, 1]

        # False Negative 비용
        fn_loss = -torch.mean(targets * torch.log(positive_probs + 1e-10) * self.fn_cost)
        
        # False Positive 비용
        fp_loss = -torch.mean((1 - targets) * torch.log(1 - positive_probs + 1e-10) * self.fp_cost)

        return fn_loss + fp_loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha, gamma, fn_cost, fp_cost, lambda_factor):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.cost_sensitive_loss = CostSensitiveLoss(fn_cost=fn_cost, fp_cost=fp_cost)
        self.lambda_factor = lambda_factor

    def forward(self, inputs, targets):
        return self.lambda_factor * self.focal_loss(inputs, targets) + \
               (1 - self.lambda_factor) * self.cost_sensitive_loss(inputs, targets)

def adjust_threshold(output, threshold=0.5):
    """
    Adjust the threshold for classification.

    Parameters:
        output (torch.Tensor): The raw output from the model.
        threshold (float): The threshold for classification.

    Returns:
        torch.Tensor: The adjusted output.
    """
    probas = Fu.softmax(output, dim=1)  # Compute probabilities using softmax
    predictions = (probas[:, 1] > threshold).float()  # Binary classification based on adjusted threshold
    return predictions

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

def normalize_16bit_to_float(img_np):
    # Ensure it is a NumPy array and 16-bit
    assert isinstance(img_np, np.ndarray), "Input should be a NumPy array"
    assert img_np.dtype == np.uint16, "Expected a 16-bit image"
    
    # Convert to float and normalize to [0, 1]
    img_np = img_np.astype(np.float32) / 65535.0  # 65535 = 2^16 - 1
    
    return img_np

class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform
        self.class_labels = os.listdir(img_dir)
        
        self.img_paths = []
        self.labels = []
        for label, class_folder in enumerate(['cancer', 'not_cancer']):
            class_folder_path = os.path.join(img_dir, class_folder)
            for img_name in os.listdir(class_folder_path):
                self.img_paths.append(os.path.join(class_folder_path, img_name))
                self.labels.append(label)
        
        # Add a targets attribute to store labels
        self.targets = self.labels

 
        print(f'Number of image paths: {len(self.img_paths)}')
        print(f'Number of labels: {len(self.labels)}')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        # Load image
        img = Image.open(img_path)

     #   print("11111111")
        # Apply transformations
        if self.transform:
            img = self.transform(img)
      #  print("333")
        # Convert to numpy array and ensure dtype is float
        img_np = np.array(img).astype(np.float32)
     #   print("444")
        # Normalize image to [0, 1] range
        img_np = img_np / 65535.0  # assuming the original dtype was uint16
     #   print("555")
        # Ensure the numpy array is 3D (H, W, C)
        if len(img_np.shape) == 2:  # Check if the image is grayscale
            img_np = np.stack([img_np]*3, axis=-1)
    #    print("666")
        # Ensure channel order is correct
        if img_np.shape[2] != 3:
            raise ValueError("Last dimension should be 3 (for R, G, B), got {}".format(img_np.shape[2]))
     #   print("777")
        # Convert to torch.Tensor
        img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1)))  # Convert (H, W, C) to (C, H, W)
      #  print("888")
        return img_tensor, label






# # Example usage:
# transform = transforms.Compose([
#     transforms.ToTensor()
#     # Add other transformations if needed
#     # ...
# ])

# dataset = CustomDataset(img_dir='/path/to/images', transform=transform)


class MultiClassifier():
    
    def __init__(self, model_name, dataset_path, class_labels, loss_fn="focal",
                 batch_size=4,lr=1e-4, weight_decay=1e-4, num_workers=0, patience=3, epoch=20,  accumulation_steps=4,pretrain=True, check_point=True, rank=0, world_size=2):
        self.class_labels = class_labels
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.model = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.patience = patience
        self.epoch = epoch
        self.img_crop_size = None
        self.img_size = None
        self.dataset = None
        self.pretrain = pretrain
        self.accumulation_steps = accumulation_steps
        self.loss_fn = loss_fn
        self.check_point = check_point
        self.device = None
        
        self.ddp_model = None
        self.rank = rank
        self.world_size = world_size
        
        self.model_name = model_name
        self.model_configs = {
            "faster_rcnn": {
                "model_func": torchvision.models.detection.fasterrcnn_resnet50_fpn,
                "last_layer": "roi_heads",
                "dropout": 0.5
            },
            "resnet50": {
                "model_func": torchvision.models.resnet50,
                "last_layer": "fc",
                "dropout": 0.5
            },
            "resnet101": {
                "model_func": torchvision.models.resnet101,
                "last_layer": "fc",
                "dropout": 0.5
            },
            "resnet152": {
                "model_func": torchvision.models.resnet152,
                "last_layer": "fc",
                "dropout": 0.5
            },
            "resnext101_32x8d": {
                "model_func": torchvision.models.resnext101_32x8d,
                "last_layer": "fc",
                "dropout": 0.5
            },
             "wide_resnet101_2": {
                "model_func": torchvision.models.wide_resnet101_2,
                "last_layer": "fc",
                "dropout": 0.5
            },
            
            "efficientnet_b5": {
                "model_func": torchvision.models.efficientnet_b7,
                "last_layer": "classifier",
                "in_features_idx": 1,
                "dropout": 0.5
            }, "efficientnet_b7": {
                "model_func": torchvision.models.efficientnet_b7,
                "last_layer": "classifier",
                "in_features_idx": 1,
                "dropout": 0.5
            },  "efficientnet_v2_s": {
                "model_func": torchvision.models.efficientnet_v2_s,
                "last_layer": "classifier",
                "in_features_idx": 1,
                "dropout": 0.5
            },
             "efficientnet_v2_l": {
                "model_func": torchvision.models.efficientnet_v2_l,
                "last_layer": "classifier",
                "in_features_idx": 1,
                "dropout": 0.5
            },
            "alexnet": {
                "model_func": torchvision.models.alexnet,
                "last_layer": "classifier",
                "in_features_idx": 6,
                "dropout": 0.5
            },
            #Swin_S_Weights.IMAGENET1K_V1
            "swin_s": {
                "model_func": torchvision.models.swin_s,
                "last_layer": "head",
             #   "in_features_idx": 1,#(head): Linear(in_features=768, out_features=1000, bias=True)
                "dropout": 0.5
            },
            "swin_t": {
                "model_func": torchvision.models.swin_t,
                "last_layer": "head",
             #   "in_features_idx": 1,#(head): Li`near(in_features=768, out_features=1000, bias=True)
                "dropout": 0.5
            },
            "swin_v2_b": {
                "model_func": torchvision.models.swin_v2_b,
                "last_layer": "head",
             #   "in_features_idx": 1,#(head): Li`near(in_features=768, out_features=1000, bias=True)
                "dropout": 0.7
            },
            "vit_b_32": {
                "model_func": torchvision.models.vit_b_32,
                "last_layer": "heads",
                "in_features_idx": 0,#(head): Li`near(in_features=768, out_features=1000, bias=True)
                "dropout": 0.5
            },
            "vit_h_14": {
                "model_func": torchvision.models.vit_h_14,
                "last_layer": "heads",
                "in_features_idx": 0,#(head): Li`near(in_features=768, out_features=1000, bias=True)
                "dropout": 0.7
            },
            "maxvit_t": {
                "model_func": torchvision.models.maxvit_t,
                "last_layer": "classifier",
                "in_features_idx": 5,#(head): Li`near(in_features=768, out_features=1000, bias=True)
                "dropout": 0.5
            },
            
            
            #Swin_T_Weights.IMAGENET1K_V1
        }
        self.init_model()
        
    def setup_ddp(self, model, rank, world_size):
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.model.to(self.device)
        self.ddp_model = DDP(model, device_ids=[rank])
        self.world_size = world_size
    
    def init_ddp(self, rank, world_size):
        self.model.to(self.device)
     #   self.model = DDP(self.model, device_ids=[self.device], output_device=self.device)
     #   self.rank = rank
        self.world_size = world_size
        
    def init_model(self):
        model_config = self.model_configs.get(self.model_name)
        if not model_config:
            raise ValueError(f"Model {self.model_name} is not supported.")

        cached_weights_path = f"{self.model_name}_weights.pth"

        # If rank is 0 and cached weights do not exist, download and cache them
        if self.rank == 0:
            print("Downloading model weights...")
            # Change this line to use weights instead of pretrained
            self.model = model_config["model_func"](weights=True)
            torch.save(self.model.state_dict(), cached_weights_path)

        # Ensure all processes wait until rank 0 has downloaded and saved the weights
        torch.distributed.barrier()

        # For all other ranks, load the model with pretrained=False and then load the cached weights
        if self.rank != 0 or self.model is None:
            print(f"Rank {self.rank} loading cached model weights...")
            # Ensure model is initialized before loading state_dict
            self.model = model_config["model_func"](weights=False)
            self.model.load_state_dict(torch.load(cached_weights_path))

        # Additional model setup code...


        
    #    print(self.model)
        last_layer = getattr(self.model, model_config["last_layer"])
        if "in_features_idx" in model_config and model_config["in_features_idx"] is not None:
            num_ftrs = last_layer[model_config["in_features_idx"]].in_features
        elif self.model_name != "faster_rcnn":
             num_ftrs = last_layer.in_features
        
       
        # 예: efficientnet_v2_s의 경우
        # last_layer = self.model.classifier
        # num_ftrs = last_layer[1].in_features
        
        if self.model_name == "alexnet":
            setattr(self.model, model_config["last_layer"], nn.Sequential(
                nn.Linear(9216, 4096),  # Adjusted the input size here
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),  # Added another layer to reduce further if desired
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, len(self.class_labels))
            ))
        elif self.model_name == "maxvit_t":
            setattr(self.model, model_config["last_layer"], nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),  # Adjusted the input size here
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.LayerNorm((512,), eps=1e-05, elementwise_affine=True),
                nn.Linear(in_features=512, out_features=512, bias=True),  # Added another layer to reduce further if desired
                nn.Tanh(),
                nn.Dropout(),
                nn.Linear(512, len(self.class_labels), bias=False)
            ))
        elif self.model_name == "faster_rcnn":
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            num_classes = len(self.class_labels)
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.class_labels))
        
        else:
            # setattr(self.model, model_config["last_layer"], nn.Sequential(
            #   #  nn.Dropout(p=model_config["dropout"]),
            #     nn.Linear(num_ftrs, len(self.class_labels))
            # ))
            setattr(self.model, model_config["last_layer"],
                    nn.Sequential(
                # nn.Linear(num_ftrs, 256),
                # nn.BatchNorm1d(256),
                # nn.ReLU(),
                # nn.Linear(256, 128),
                # nn.BatchNorm1d(128),
                # nn.ReLU(),
                # nn.Dropout(p=0.5),
                # nn.Linear(128, len(self.class_labels))
                nn.Linear(num_ftrs, 256),
                nn.SyncBatchNorm(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.SyncBatchNorm(128),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, len(self.class_labels))
            ))
      # print(self.model)
        if 'resnet' in self.model_name:
            conv1_weight = self.model.conv1.weight.mean(dim=1, keepdim=True)
            # 첫 번째 합성곱 층 수정
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.conv1.weight.data.copy_(conv1_weight)
        elif 'efficientnet_v2_l' in self.model_name:
            conv1_weight = self.model.features[0][0].weight.mean(dim=1, keepdim=True)
            self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.features[0][0].weight.data.copy_(conv1_weight)
        elif 'efficientnet_v2_s' in self.model_name:
            conv1_weight = self.model.features[0][0].weight.mean(dim=1, keepdim=True)
            self.model.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.features[0][0].weight.data.copy_(conv1_weight)
        elif 'maxvit' in self.model_name:
            conv1_weight = self.model.stem[0][0].weight.mean(dim=1, keepdim=True)
            self.model.stem[0][0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.stem[0][0].weight.data.copy_(conv1_weight)
        elif 'swin' in self.model_name:
            conv1_weight = self.model.features[0][0].weight.mean(dim=1, keepdim=True)
            self.model.features[0][0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
            self.model.features[0][0].weight.data.copy_(conv1_weight)


   #     print(self.model)
    

        self.model = self.model.to(self.device)
    def init_model_org(self):
        
        
        model_config = self.model_configs.get(self.model_name)
        if not model_config:
            raise ValueError(f"Model {self.model_name} is not supported.")
        
        self.model = model_config["model_func"](weights=self.pretrain)
    #    print(self.model)
        last_layer = getattr(self.model, model_config["last_layer"])
        if "in_features_idx" in model_config and model_config["in_features_idx"] is not None:
            num_ftrs = last_layer[model_config["in_features_idx"]].in_features
        elif self.model_name != "faster_rcnn":
             num_ftrs = last_layer.in_features
        
       
        # 예: efficientnet_v2_s의 경우
        # last_layer = self.model.classifier
        # num_ftrs = last_layer[1].in_features
        
        if self.model_name == "alexnet":
            setattr(self.model, model_config["last_layer"], nn.Sequential(
                nn.Linear(9216, 4096),  # Adjusted the input size here
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),  # Added another layer to reduce further if desired
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, len(self.class_labels))
            ))
        elif self.model_name == "maxvit_t":
            setattr(self.model, model_config["last_layer"], nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),  # Adjusted the input size here
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.LayerNorm((512,), eps=1e-05, elementwise_affine=True),
                nn.Linear(in_features=512, out_features=512, bias=True),  # Added another layer to reduce further if desired
                nn.Tanh(),
                nn.Dropout(),
                nn.Linear(512, len(self.class_labels), bias=False)
            ))
        elif self.model_name == "faster_rcnn":
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            num_classes = len(self.class_labels)
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.class_labels))
        
        else:
            # setattr(self.model, model_config["last_layer"], nn.Sequential(
            #   #  nn.Dropout(p=model_config["dropout"]),
            #     nn.Linear(num_ftrs, len(self.class_labels))
            # ))
            setattr(self.model, model_config["last_layer"],
                    nn.Sequential(
                nn.Linear(num_ftrs, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, len(self.class_labels))
            ))
      # print(self.model)
        if 'resnet' in self.model_name:
            conv1_weight = self.model.conv1.weight.mean(dim=1, keepdim=True)

            # 첫 번째 합성곱 층 수정
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.conv1.weight.data.copy_(conv1_weight)
        elif 'efficientnet_v2_l' in self.model_name:
            conv1_weight = self.model.features[0][0].weight.mean(dim=1, keepdim=True)
            self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.features[0][0].weight.data.copy_(conv1_weight)
        elif 'efficientnet_v2_s' in self.model_name:
            conv1_weight = self.model.features[0][0].weight.mean(dim=1, keepdim=True)
            self.model.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.features[0][0].weight.data.copy_(conv1_weight)


   #     print(self.model)
    

        self.model = self.model.to(self.device)
        
 

    from sklearn.metrics import (
        roc_auc_score, roc_curve, accuracy_score,
        classification_report, confusion_matrix
    )

    def calculate_metrics(self, true_labels, predicted_labels, predicted_probs, classes,
                          calc_roc_auc=True, calc_roc_curve=True, calc_accuracy=True,
                          calc_report=True, calc_conf_matrix=True):
        """
        Calculate various classification metrics.

        Parameters:
        - true_labels: Actual labels of the samples
        - predicted_labels: Predicted labels of the samples
        - predicted_probs: Predicted probabilities of the positive class
        - classes: Names of the classes
        - calc_roc_auc: Whether to calculate the ROC AUC score (default: True)
        - calc_roc_curve: Whether to calculate the ROC curve (default: True)
        - calc_accuracy: Whether to calculate the accuracy (default: True)
        - calc_report: Whether to generate a classification report (default: True)
        - calc_conf_matrix: Whether to generate a confusion matrix (default: True)

        Returns:
        - A tuple containing calculated metrics in the following order:
          (roc_index, fpr, tpr, overall_accuracy, report, cm)
          Depending on the flags provided, some elements may be None.
        """
        roc_index = roc_auc_score(true_labels, predicted_probs) if calc_roc_auc else None
        fpr, tpr, _ = roc_curve(true_labels, predicted_probs) if calc_roc_curve else (None, None, None)
        overall_accuracy = accuracy_score(true_labels, predicted_labels) if calc_accuracy else None
        report = classification_report(true_labels, predicted_labels, target_names=classes) if calc_report else None
        cm = confusion_matrix(true_labels, predicted_labels) if calc_conf_matrix else None

        return roc_index, fpr, tpr, overall_accuracy, report, cm
    
#     def calculate_metrics(self, true_labels, predicted_labels, predicted_probs, classes):
#         roc_index = roc_auc_score(true_labels, predicted_probs)
#         fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
#         overall_accuracy = accuracy_score(true_labels, predicted_labels)
#         report = classification_report(true_labels, predicted_labels, target_names=classes)
#         cm = confusion_matrix(true_labels, predicted_labels)
#         return roc_index, fpr, tpr, overall_accuracy, report, cm

    def plot_roc_curve(self, fpr, tpr, roc_index):
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_index)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    
    def report_metrics(self, true_labels, predicted_labels, predicted_probs, classes):
        roc_index, fpr, tpr, overall_accuracy, report, cm = self.calculate_metrics(true_labels, predicted_labels, predicted_probs, classes)
        
        print("########################################")
        print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
        print(report)
        self.plot_confusion_matrix(cm, classes)
        self.plot_roc_curve(fpr, tpr, roc_index)
        # Assuming you have a method plot_confusion_matrix
      
  
    def init_optimiser(self):
        optimizer = optim.AdamW(self.ddp_model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                betas=(0.9, 0.999), eps=1e-8)
        # Use ReduceLROnPlateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)
        return optimizer, scheduler

   

    def getDataSet(self, dataset_type):
        dataset = None
        if dataset_type == DataSet.ALL:
            dataset = self.dataset
        elif dataset_type == DataSet.TRAIN:
            dataset = self.train_dataset
        elif dataset_type == DataSet.VAL:
            dataset = self.valid_dataset
        elif dataset_type == DataSet.TEST:
            dataset = self.test_dataset
        elif dataset_type == DataSet.TRAIN_LOADER:
            dataset = self.train_loader
        elif dataset_type == DataSet.VAL_LOADER:
            dataset = self.valid_loader
        elif dataset_type == DataSet.TEST_LOADER:
            dataset = self.test_loader
        return dataset
         

    def normalize_image(self, image_array):
        min_val = np.min(image_array)
        max_val = np.max(image_array)
        print("min max", min_val, max_val)
        if min_val == max_val:
            print("Warning: Constant image encountered and skipped normalization")
            normalized_image_array = np.zeros_like(image_array, dtype=np.uint8)
        else:
            normalized_image_array = (image_array - min_val) / (max_val - min_val)
            # Check for NaN or Inf values
            if np.any(np.isnan(normalized_image_array)) or np.any(np.isinf(normalized_image_array)):
                print("Warning: NaN or Inf values found in normalized image")
                normalized_image_array[np.isnan(normalized_image_array)] = 0
                normalized_image_array[np.isinf(normalized_image_array)] = 0

        # Ensure type conversion is valid
        normalized_image_array = np.clip(normalized_image_array * 255, 0, 255).astype(np.uint8)
        return normalized_image_array
 

    def show_images(self, dataset_type, num_images=5, color_type=ColorType.GRAY):
        dataset = self.getDataSet(dataset_type)

        # Check the number of rows required for the given number of images
        num_rows = int(np.ceil(num_images / 5))
        fig, axes = plt.subplots(num_rows, min(5, num_images), figsize=(15, 5 * num_rows))

        # If the dataset is a DataLoader
        if isinstance(dataset, DataLoader):
            images, labels = next(iter(dataset))
            for i in range(min(num_images, len(images))):
                ax = axes[i // 5, i % 5] if num_rows > 1 else axes[i % 5]
                image = images[i]
                label = labels[i]
                if color_type == ColorType.GRAY:
                    image_gray = image[0].numpy()
                    ax.imshow(image_gray, cmap='gray')
                else:
                    ax.imshow(image.permute(1, 2, 0).numpy())
                ax.set_title(f"{self.class_labels[label]}")

        # If the dataset is an ImageFolder or similar
        else:
            for i in range(min(num_images, len(dataset))):
                ax = axes[i // 5, i % 5] if num_rows > 1 else axes[i % 5]
                image, label = dataset[i]
                if isinstance(image, torch.Tensor):
                    if color_type == ColorType.GRAY:
                        image_gray = image[0].numpy()
                        ax.imshow(image_gray, cmap='gray')
                    else:
                        ax.imshow(image.permute(1, 2, 0).numpy())
                else:  # Assumes PIL Image
                    if color_type == ColorType.GRAY:
                        image_gray = np.array(image.convert('L'))
                        ax.imshow(image_gray, cmap='gray')
                    else:
                        ax.imshow(np.array(image))
                ax.set_title(f"{self.class_labels[label]}")

        # Save the figure
        plt.savefig(f"{dataset_type}_images.png", bbox_inches='tight')
        plt.close(fig)  # Close the figure window

            
    def print_class_distribution(self, dataset, dataset_name="---"):
      #  print("############################################")
        print("------------ " + dataset_name + " ------------")
        labels = [label for _, label in dataset]
        class_counts = {label: labels.count(label) for label in set(labels)}
        nCnt = 0
        nSum = 0
        for class_label, count in class_counts.items():
            print(f"{self.class_labels[nCnt]}: {count} images")
            nCnt+=1
            nSum += count
        print("total:", nSum)
       # print("############################################")
                  
     
    def load_data(self):
        print("load_data started")
        start_time = time.time()  # Capture the starting time
        # Load the dataset without any transforms
        self.dataset = dataset = datasets.ImageFolder(root=self.dataset_path)
        self.img_size = dataset[0][0].size[0]
        
        self.img_width, self.img_height = dataset[0][0].size
       
        self.img_crop_size = int(self.img_size * 0.86)
        
       
        
        if "cropped" in self.dataset_path:
            train_transforms = cropped_transforms()
            valid_test_transforms = cropped_transforms()
        else:
            if self.model_name == "vit_h_14":
                train_transforms = mammo_transforms(518, self.img_crop_size)
                valid_test_transforms = valid_transforms(518, self.img_crop_size)
            elif self.model_name == "maxvit_t":
                train_transforms = mammo_transforms(224, self.img_crop_size)
                valid_test_transforms = valid_transforms(224, self.img_crop_size)
            else:
                train_transforms = mammo_transforms(img_crop_size=self.img_crop_size)
                valid_test_transforms = valid_transforms(img_crop_size=self.img_crop_size)
        
        manual_seed(42)

        
        # Apply transformations by creating new ImageFolder datasets with the splitted data
        train_valid_dataset = datasets.ImageFolder(root=os.path.join(self.dataset_path, 'train'), transform=train_transforms)
        test_dataset = datasets.ImageFolder(root=os.path.join(self.dataset_path, 'test'), transform=valid_test_transforms)

        # Get class distribution and find the minimum class size for under-sampling
        _, class_counts_train = np.unique(train_valid_dataset.targets, return_counts=True)
        min_class_count_train = np.min(class_counts_train)

        # Function to get the indices for each class
        def get_class_indices(class_idx, targets):
            return [i for i, target in enumerate(targets) if target == class_idx]

        # Under-sampling: Get random samples from each class equal to the minimum class size
        indices_train = []
        for class_idx in range(len(class_counts_train)):
            class_indices = get_class_indices(class_idx, train_valid_dataset.targets)
            under_sample_indices = np.random.choice(class_indices, min_class_count_train, replace=False)
            indices_train.extend(under_sample_indices)

        # Shuffle the indices to mix the classes
        np.random.shuffle(indices_train)

        # Creating new datasets with under-sampled indices
        train_valid_dataset = Subset(train_valid_dataset, indices_train)

        # Now you can split your train_valid_dataset into train and valid datasets
        train_size = int(0.7 * len(train_valid_dataset))
        valid_size = len(train_valid_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(train_valid_dataset, [train_size, valid_size])

        # Print class distributions after under-sampling
        # if self.rank == 0:
        #     self.print_class_distribution(train_dataset, "train_dataset")
        #     self.print_class_distribution(valid_dataset, "valid_dataset")
        #     self.print_class_distribution(test_dataset, "test_dataset")

        # Assign to class variables
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
         
        self.train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank)
        self.valid_sampler = DistributedSampler(valid_dataset, num_replicas=self.world_size, rank=self.rank)
        self.test_sampler = DistributedSampler(test_dataset, num_replicas=self.world_size, rank=self.rank)

       
         
            # Create DistributedSamplers and assign to DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Must be False when using DistributedSampler
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.train_sampler,
         #   drop_last=True
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.valid_sampler,
           # drop_last=True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.test_sampler,
           # drop_last=True
    
        )
        
       
        print("Loading data finished", f"Elapsed time: {time.time() - start_time:.2f} seconds")
#         self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, sampler=train_sampler)
#         self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, sampler=valid_sampler)
#         self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, sampler=test_sampler)
    
    def is_valid_file(self, path):
        return not '.ipynb_checkpoints' in path
    
    def load_model(self, model_save_path=None):
        #model_save_path="early_stop_checkpoint.pt"
                
        if self.model_name == "":
            self.init_model()
        if model_save_path is not None:
            if os.path.exists(model_save_path):
                checkpoint = torch.load(model_save_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
     
                 
                print(f"Loaded model weights from {model_save_path}")
                #self.load_data()
            else:
                print("Please place \"best_model_weights.pth\" in the same folder with this file.")
                # self.load_data()
                # #self.load_data_merge_unlabelled()
                 
       # else:
       #     print("Train...")
            
       ## self.model.to(rank)
       # self.model = DDP(self.model, device_ids=[rank])
     
    def profile_transform(self, transform, dataset, num_samples=100):
        transform_times = []
        for i in range(num_samples):
            start_time = time.time()
            _ = transform(dataset[i][0])
            transform_times.append(time.time() - start_time)
        return transform_times

    def train(self):
        #self.load_data()
        #self.load_data_stratified_split()
        
     
        self.train_model(self.epoch, self.check_point)

    def test_image(self, file_name):
             # Load the image
        image = Image.open(file_name)
        
        # Apply transformations (assuming you have a transform for validation/testing)
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.2643, 0.1983, 0.2102])
        ])
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Move tensor to device
        image_tensor = image_tensor.to(device)

        # Set model to evaluation mode and get prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
        prediction = output.argmax(dim=1).item()

        # Convert the prediction to the corresponding class label
        label = self.class_labels[prediction]

        return label
    
#     def calc_class_weights(self):
#          #### Calculate class counts and weights
#         class_counts = {i: 0 for i in range(len(self.class_labels))}
#         for _, labels in self.train_loader:
#             for label in labels:
#                 class_counts[label.item()] += 1
#         total_samples = sum(class_counts.values())
#         weights = [total_samples / class_counts[i] for i in range(len(self.class_labels))]
#         class_weights = torch.FloatTensor(weights).to(self.device)
#         return class_weights
    def calc_class_weights(self):
        try:
            class_counts = {i: 0 for i in range(len(self.class_labels))}
            for _, labels in self.train_loader:
                for label in labels:
                    label_item = label.item()
                    if not isinstance(label_item, int):
                        print(f"Warning: Non-integer label: {label_item}")
                    class_counts[label_item] += 1
            total_samples = sum(class_counts.values())
            weights = [total_samples / class_counts[i] for i in range(len(self.class_labels))]
            class_weights = torch.FloatTensor(weights).to(self.device)
            return class_weights
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

    
    def plot_training_graphs(self, train_losses, val_losses, train_accuracies, val_accuracies, precisions, recalls, f1_scores, aucs):
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.legend()
        plt.title('Loss')

        plt.subplot(2, 2, 2)
        plt.plot(epochs, train_accuracies, label='Training Accuracy')
        plt.plot(epochs, val_accuracies, label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy')

        plt.subplot(2, 2, 3)
        plt.plot(epochs, precisions, label='Precision')
        plt.plot(epochs, recalls, label='Recall')
        plt.plot(epochs, f1_scores, label='F1 Score')
        plt.legend()
        plt.title('Precision, Recall and F1 Score')

        plt.subplot(2, 2, 4)
        plt.plot(epochs, aucs, label='AUC')
        plt.legend()
        plt.title('AUC')

        plt.tight_layout()
        plt.show()

    def plot_training_graphs_s(self, train_losses, val_losses, train_accuracies, val_accuracies):
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(12, 8))

        # Plot Loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.legend()
        plt.title('Loss')

        # Plot Accuracy
        plt.subplot(2, 1, 2)
        plt.plot(epochs, train_accuracies, label='Training Accuracy')
        plt.plot(epochs, val_accuracies, label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy')

        plt.tight_layout()
        plt.show()

    def to_one_hot(self, tensor, n_classes):
        n = tensor.size(0)  # Get the batch size
        one_hot = torch.zeros(n, n_classes).to(tensor.device)  # Initialize one-hot tensor
        one_hot.scatter_(1, tensor.unsqueeze(1), 1)  # Perform one-hot encoding
        return one_hot


    import time  # 시간 측정을 위한 모듈

    def train_epoch(self, model, data_loader, criterion, optimizer, device, accumulation_steps,scaler):
        start_time = time.time()  # Forward pass 시작 시간 측정
        
        self.model.train()
        
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        optimizer.zero_grad()  
       
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            
            with torch.cuda.amp.autocast():
                outputs = self.ddp_model(inputs)
                loss = criterion(outputs, self.to_one_hot(labels, len(self.class_labels))) / accumulation_steps
            
            scaler.scale(loss).backward()
            

            if (i + 1) % accumulation_steps == 0:
                start_time = time.time()  # Optimizer step 시작 시간 측정
                scaler.step(optimizer)  
                scaler.update()  
                optimizer.zero_grad()
                

            running_loss += loss.item() * accumulation_steps  
            probas_train = torch.softmax(outputs, dim=1)
            _, predicted_train = torch.max(probas_train, 1)

            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        train_loss = running_loss / len(self.train_loader)
        train_accuracy = correct_train / total_train

        return train_loss, train_accuracy

            
    def validate_epoch(self, model, data_loader, criterion, device):
       # print(self.rank,"validate_epoch")
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        true_labels = []
        pred_labels = [] 
        with torch.no_grad():
            for data in self.valid_loader:
                images, labels = data[0].to(device), data[1].to(device)
                with torch.cuda.amp.autocast():
                    outputs = self.ddp_model(images)
                    
                    loss = criterion(outputs, labels)
#                     outputs = self.model(images)
#                     loss = criterion(outputs, labels)
                   

                val_loss += loss.item()

                probas = Fu.softmax(outputs, dim=1)  # Compute probabilities
                threshold = 0.5  # Adjust threshold as needed for validation
                predicted = adjust_threshold(probas, threshold=threshold)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(predicted.cpu().numpy())
     
        return total, correct, val_loss, true_labels, pred_labels

    def calculate_metrics_for_training(self, total, correct, val_loss, true_labels, pred_labels):
     #   print(self.rank,"calculate_metrics_for_training")
        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        
        f1 = f1_score(true_labels, pred_labels)
        auc = roc_auc_score(true_labels, pred_labels)

        mean_val_loss = val_loss / len(self.valid_loader)
        val_accuracy = correct / total
        return mean_val_loss, val_accuracy, precision, recall, f1, auc

    
        
    def train_model(self, num_epochs=20, check_point=True):
        print("check_point", check_point)
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        aucs = []
        destination_rank = 0
        
        self.device = torch.device(f"cuda:{self.rank}")
        self.model.to(self.device)
        print(self.rank,"self.device", self.device)
        #### Calculate class counts and weights
        class_weights = self.calc_class_weights()

        # Initialize loss function and optimizer
        # Initialize loss function with class weights
        
        #'loss_fn': ['dice', 'focal', 'crossentropy', 'tversky']
        if self.loss_fn == 'crossentropy':
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif self.loss_fn == 'dice':
            criterion = DiceLoss()
        elif self.loss_fn == 'focal':
            criterion = FocalLoss()
        elif self.loss_fn == 'tversky':
            criterion = TverskyLoss()
        #criterion = FocalLoss()
        #  criterion = CostSensitiveLoss(fn_cost=1.7, fp_cost=1.1)
      #  criterion = nn.CrossEntropyLoss()
       # criterion = TverskyLoss()
        #criterion = nn.HingeEmbeddingLoss()
       # criterion = DiceLoss()
        # Initialize loss function with hyperparameters
    
        optimizer, scheduler = self.init_optimiser()
    #    print(self.rank,"optimizer")
        # Initialize early stopping
        early_stopping = DistributedEarlyStopping(patience=self.patience, verbose=True, path=self.model_name+'_'+self.dataset_path.replace('/','_')+'_best.pt', check_point=check_point)
    #    print(self.rank,"early_stopping")
        scaler = torch.cuda.amp.GradScaler()
        accumulation_steps = self.accumulation_steps # Number of mini-batches to accumulate over
        
      #  print(self.rank,"accumulation_steps 1111")
        for epoch in range(num_epochs):
            self.train_sampler.set_epoch(epoch)
            train_loss, train_accuracy = self.train_epoch(self.model, self.train_loader, criterion, optimizer, self.device, accumulation_steps,scaler)
            total, correct, val_loss, true_labels, pred_labels = self.validate_epoch(self.model, self.valid_loader, criterion, self.device)
            mean_val_loss, val_accuracy, precision, recall, f1, auc = self.calculate_metrics_for_training(total, correct, val_loss, true_labels, pred_labels)
           # print(self.rank,"accumulation_steps 22222")
            # Sum up the metrics across all ranks
            
            train_loss_tensor = torch.tensor(train_loss).to(self.device)
            train_accuracy_tensor = torch.tensor(train_accuracy).to(self.device)
            mean_val_loss_tensor = torch.tensor(mean_val_loss).to(self.device)
            val_accuracy_tensor = torch.tensor(val_accuracy).to(self.device)
            precision_tensor = torch.tensor(precision).to(self.device)
            recall_tensor = torch.tensor(recall).to(self.device)
            f1_tensor = torch.tensor(f1).to(self.device)
            auc_tensor = torch.tensor(auc).to(self.device)

            # Reduce all tensors
            dist.reduce(train_loss_tensor, dst=destination_rank, op=dist.ReduceOp.SUM)
            dist.reduce(train_accuracy_tensor, dst=destination_rank, op=dist.ReduceOp.SUM)
            dist.reduce(mean_val_loss_tensor, dst=destination_rank, op=dist.ReduceOp.SUM)
            dist.reduce(val_accuracy_tensor, dst=destination_rank, op=dist.ReduceOp.SUM)
            dist.reduce(precision_tensor, dst=destination_rank, op=dist.ReduceOp.SUM)
            dist.reduce(recall_tensor, dst=destination_rank, op=dist.ReduceOp.SUM)
            dist.reduce(f1_tensor, dst=destination_rank, op=dist.ReduceOp.SUM)
            dist.reduce(auc_tensor, dst=destination_rank, op=dist.ReduceOp.SUM)

            # If needed, convert them back to Python floats
            train_loss = train_loss_tensor.item()
            train_accuracy = train_accuracy_tensor.item()
            mean_val_loss = mean_val_loss_tensor.item()
            val_accuracy = val_accuracy_tensor.item()
            precision = precision_tensor.item()
            recall = recall_tensor.item()
            f1 = f1_tensor.item()
            auc = auc_tensor.item()

           
            if torch.distributed.get_rank() == destination_rank:
                 
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Loss: {train_loss/self.world_size:.3f}, "
                      f"Acc: {train_accuracy/self.world_size:.3f}, "
                      f"Val Loss: {mean_val_loss/self.world_size:.3f}, "
                      f"Val Accuracy: {val_accuracy/self.world_size:.3f}, "
                      f"Precision: {precision/self.world_size:.3f}, "
                      f"Recall: {recall/self.world_size:.3f}, "
                      f"F1: {f1/self.world_size:.3f}, "
                      f"AUC: {auc/self.world_size:.3f}")

#             print(f"Epoch {epoch+1} /{num_epochs}, Loss: {train_loss:.3f}, Acc: {train_accuracy:.3f}, Val Loss: {mean_val_loss:.3f}, Val Accuracy: {val_accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
            
            # Save metrics
            train_losses.append(train_loss)
            val_losses.append(mean_val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            aucs.append(auc)
            
            #scheduler.step(recall)
            scheduler.step(mean_val_loss)
            early_stopping(mean_val_loss, self.model, optimizer, self.rank, self.device)
         
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        # Load best model weights
      
        if check_point == True:
            self.load_checkpoint(self.model_name+'_'+self.dataset_path.replace('/','_')+'_best.pt')
       
      #  print('Finished Training and Validation')
        return mean_val_loss
       # return train_losses, val_losses, train_accuracies, val_accuracies
  
#     def load_checkpoint(self, filename):
#         # Only rank 0 loads the model
#         print(self.rank, "load_checkpoint")
#         if self.rank == 0:
#             checkpoint = torch.load(filename)
#             self.model.load_state_dict(checkpoint['model_state_dict'])

#         # Broadcast model state to all the other processes
#         for param_tensor in self.model.state_dict():
#             dist.broadcast(self.model.state_dict()[param_tensor], src=0)
#         print(self.rank, "load_checkpoint end")

    def self_val_model_matrix_org(self, mode='val'):
        print('Testing ', mode, 'data...')

        classes = ['cancer', 'not_cancer']
        data_loader = self.valid_loader if mode == 'val' else self.test_loader

        true_labels = []
        predicted_labels = []

        correct_predictions = 0
        total_predictions = 0

        print("########################################")
        self.model.eval()

        with torch.no_grad():  # Disable gradient computation during validation/testing
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.ddp_model(inputs)
                _, preds = torch.max(outputs, 1)

                total_predictions += labels.size(0)
                correct_predictions += (preds == labels).sum().item()

                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(preds.cpu().numpy())
                

        print("########################################")
        average_accuracy = (correct_predictions / total_predictions) * 100
        print(f'{mode.capitalize()} Accuracy: {average_accuracy:.2f}%')

        overall_accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

        report = classification_report(true_labels, predicted_labels, target_names=classes)
        print(report)

        cm = confusion_matrix(true_labels, predicted_labels)
        self.plot_confusion_matrix(cm, classes, mode)
        
    def load_checkpoint(self, filename):
        print(self.rank, "load_checkpoint")

        # Only rank 0 loads the model
        if self.rank == 0:
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Ensure all processes wait until rank 0 has loaded the checkpoint
        dist.barrier()

        # Broadcast model state to all the other processes
        for param_tensor in self.model.state_dict():
            # Explicitly move each parameter tensor to the device that is being used by the current process
            tensor_to_send = self.model.state_dict()[param_tensor].to(self.device) if self.rank == 0 else torch.zeros_like(self.model.state_dict()[param_tensor]).to(self.device)
            dist.broadcast(tensor_to_send, src=0)
            if self.rank != 0:
                self.model.state_dict()[param_tensor].copy_(tensor_to_send)

    def self_val_model_matrix(self, mode='val'):
        print('Testing ', mode, 'data...')
        classes = ['cancer', 'not_cancer']
        data_loader = self.valid_loader if mode == 'val' else self.test_loader

        # 1. Local predictions and metric calculations
        local_true_labels = []
        local_predicted_labels = []
        
        correct_predictions = 0
        total_predictions = 0

        print("########################################")
        self.model.eval()

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                total_predictions += labels.size(0)
                correct_predictions += (preds == labels).sum().item()
                
                local_true_labels.extend(labels.cpu().numpy())
                local_predicted_labels.extend(preds.cpu().numpy())
        
        print("########################################")
        local_average_accuracy = (correct_predictions / total_predictions) * 100
        local_average_accuracy_tensor = torch.tensor(local_average_accuracy).to(self.device)
        dist.reduce(local_average_accuracy_tensor, dst=0, op=dist.ReduceOp.SUM)
        
        # 2. Gather all predictions from all processes
        global_true_labels = [np.array([])] * dist.get_world_size()
        global_predicted_labels = [np.array([])] * dist.get_world_size()
        dist.all_gather_object(global_true_labels, np.array(local_true_labels))
        dist.all_gather_object(global_predicted_labels, np.array(local_predicted_labels))
        
        # 3. Calculate and print global metrics (only on rank=0 process to avoid duplicate outputs)
        if self.rank == 0:
            global_true_labels = np.concatenate(global_true_labels)
            global_predicted_labels = np.concatenate(global_predicted_labels)
            
            global_accuracy = local_average_accuracy_tensor.item() / dist.get_world_size()
            print(f"Global accuracy: {global_accuracy:.2f}%")
            
            report = classification_report(global_true_labels, global_predicted_labels, target_names=classes)
            print(report)

            cm = confusion_matrix(global_true_labels, global_predicted_labels)
            self.plot_confusion_matrix(cm, classes, mode)


    def plot_confusion_matrix(self, cm, classes, mode, title='Confusion Matrix', cmap=plt.cm.Blues):
       

        plt.figure(figsize=(4, 4))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        #plt.show()
        plt.savefig(mode+"_plot_confusion_matrix.png")

    




# dataset_path = 'datasets/CBIS-DDSM/train_256/'
# class_labels = ['cancer', 'not_cancer']
# cf = MultiClassifier("swin_s", dataset_path, class_labels)
 
# print(f'Testing performance ')
# print('########################################################################')

# print('Loading model...')
# cf.load_model()
 
def model_parallel(rank, world_size, model_name, dataset_path, class_labels, loss_fn, training_params):
    batch_size = training_params["batch_size"]
    lr = training_params["lr"]
    weight_decay = training_params["weight_decay"]
    num_workers = training_params["num_workers"]
    patience = training_params["patience"]
    epoch = training_params["epoch"]
    accumulation_steps = training_params["accumulation_steps"]
    pretrain = training_params["pretrain"]
    checkpoint = training_params["checkpoint"]

    
    try:
        # Initialize Distributed Data Parallel (DDP)
      #  dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=rank, world_size=world_size)

     #   dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        
        print(f"Running DDP with model parallel example on rank {rank}.")
        setup(rank, world_size)

        # Initialize model
        cf = MultiClassifier(model_name, dataset_path, class_labels, loss_fn, batch_size
                                         ,lr ,weight_decay, num_workers, patience, epoch, accumulation_steps,
                                         pretrain, checkpoint, rank, world_size)
        cf.load_data()
        
        if rank == 0:
            cf.show_images(DataSet.ALL)
            cf.show_images(DataSet.TRAIN)
            cf.show_images(DataSet.TEST)
        
        cf.model = cf.model.to(cf.device)
        cf.setup_ddp(cf.model, rank, world_size)

        cf.train()
    
        print("train end")
       
        cf.self_val_model_matrix('test')
        cf.self_val_model_matrix('val')
         
        torch.distributed.barrier()
        cleanup()
        print("cleanup end")
    except Exception as e:
        print(f"Process {rank} encountered exception: {e}")
        traceback.print_exc()
    finally:
        if dist.is_initialized():  # Check if process group is still active
            dist.destroy_process_group()

if __name__ == "__main__":
    torch.cuda.empty_cache()  # GPU 캐시 메모리 정리
    cpu_count = cpu_count()  # Gets the number of CPU cores in the system
    print("cpu_count",cpu_count)
    
     
    num_workers = cpu_count // 2

    world_size = 2
    training_params = {
        "batch_size": 8,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "num_workers": num_workers,
        "patience": 5,
        "epoch": 30,
        "accumulation_steps": 8,
        "pretrain": True,
        "checkpoint": True
    }
    print(training_params)
    base_data_path=''
    #base_data_path='datasets/CBIS-DDSM/'
    #cf = MultiClassifier("efficientnet_v2_s", dataset_path, class_labels, "dice", 16, 2e-4, 1e-4, 4,7, 110, 4, True, True)
    #dataset_path = 'full_256'
    #dataset_path = 'cropped_256'
    #dataset_path = 'Full_Roi_Overlay_256_20'
    #dataset_path = 'Full_Roi_Overlay_256_20_test'
    dataset_path = base_data_path + 'full_2048'
    class_labels = ['cancer', 'not_cancer']
    #cf = MultiClassifier("efficientnet_v2_s", dataset_path, class_labels, 16, 2e-4, 1e-4,4,7, 30, True)
    mp.spawn(model_parallel, args=(world_size, "efficientnet_v2_s", dataset_path, class_labels, "dice", training_params), nprocs=world_size, join=True)

   # mp.spawn(model_parallel, args=(world_size, ...), nprocs=world_size, join=True)
 #
  
