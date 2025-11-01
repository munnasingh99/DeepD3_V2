import numpy as np
import flammkuchen as fl
import cv2
import albumentations as A
import random
import torch
from torch.utils.data import Dataset

class DataGeneratorDataset(Dataset):
    def __init__(self, fn, samples_per_epoch=50000, size=(1, 128, 128), 
                 target_resolution=None, augment=True, shuffle=True, 
                 seed=42, normalize=[0, 1], min_content=50):
        """Data Generator for PyTorch training.

        Args:
            fn (str): The path to the training data file
            samples_per_epoch (int, optional): Samples used in each epoch. Defaults to 50000.
            size (tuple, optional): Shape of a single sample. Defaults to (1, 128, 128).
            target_resolution (float, optional): Target resolution in microns. Defaults to None.
            augment (bool, optional): Enables augmenting the data. Defaults to True.
            shuffle (bool, optional): Enabled shuffling the data. Defaults to True.
            seed (int, optional): Creates pseudorandom numbers for shuffling. Defaults to 42.
            normalize (list, optional): Values range when normalizing data. Defaults to [-1, 1].
            min_content (float, optional): Minimum content in image. Defaults to 0.
        """
        # Save settings
        self.fn = fn
        self.augment = augment
        self.shuffle = shuffle
        self.aug = self._get_augmenter()
        self.seed = seed
        self.normalize = normalize
        self.samples_per_epoch = samples_per_epoch
        self.size = size
        self.target_resolution = target_resolution
        self.min_content = min_content

        # Load data
        self.d = fl.load(self.fn)
        self.data = self.d['data']
        self.meta = self.d['meta']

        # Seed randomness
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def __len__(self):
        """Returns the number of samples per epoch"""
        return self.samples_per_epoch

    def __getitem__(self, idx):
        """Get a single sample
        
        Args:
            idx (int): Index of the sample (not used as we generate random samples)
        
        Returns:
            tuple: Contains image tensor and tuple of (dendrite, spine) label tensors
        """
        eps = 1e-5

        # Get a valid sample
        image, dendrite, spines = self.getSample()

        # Augmenting the data
        if self.augment:
            augmented = self.aug(
                image=image, 
                mask1=dendrite.astype(np.uint8), 
                mask2=spines.astype(np.uint8)
            )
            image = augmented['image']
            dendrite = augmented['mask1']
            spines = augmented['mask2']

        # Min/max scaling
        image = (image.astype(np.float32) - image.min()) / (image.max() - image.min() + eps)
        # Shifting and scaling
        image = image * (self.normalize[1]-self.normalize[0]) + self.normalize[0]

        # Convert to torch tensors and add channel dimension if needed
        image = torch.from_numpy(image).float()
        if image.dim() == 2:
            image = image.unsqueeze(0)  # Add channel dimension

        dendrite = torch.from_numpy(dendrite.astype(np.float32)).float() / (dendrite.max() + eps)
        if dendrite.dim() == 2:
            dendrite = dendrite.unsqueeze(0)

        spines = torch.from_numpy(spines.astype(np.float32)).float() / (spines.max() + eps)
        if spines.dim() == 2:
            spines = spines.unsqueeze(0)

        return image, (dendrite, spines)

    def _get_augmenter(self):
        """Defines used augmentations"""
        aug = A.Compose([
            A.RandomBrightnessContrast(p=0.25),    
            A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Blur(p=0.2),
            A.GaussNoise(p=0.5)], p=1,
            additional_targets={
                'mask1': 'mask',
                'mask2': 'mask'
            })
        return aug

    def getSample(self, squeeze=True):
        """Get a sample from the provided data

        Args:
            squeeze (bool, optional): if plane is 2D, skip 3D. Defaults to True.

        Returns:
            list(np.ndarray, np.ndarray, np.ndarray): stack image with respective labels
        """
        while True:
            r = self._getSample(squeeze)
            
            # If sample was successfully generated
            # and we don't care about the content
            if r is not None and self.min_content == 0:
                return r

            # If sample was successfully generated
            # and we do care about the content
            elif r is not None:
                # In either or both annotation should be at least `min_content` pixels
                # that are being labelled.
                if (r[1]).sum() > self.min_content or (r[2]).sum() > self.min_content:
                    return r
                else:
                    continue
            else:
                continue

    def _getSample(self, squeeze=True):
        """Retrieves a sample

        Args:
            squeeze (bool, optional): Squeezes return shape. Defaults to True.

        Returns:
            tuple: Tuple of stack (X), dendrite (Y0) and spines (Y1)
        """
        # Adjust for 2 images
        if len(self.size) == 2:
            size = (1,) + self.size
        else:
            size = self.size
        
        # sample random stack
        r_stack = np.random.choice(len(self.meta))
        
        target_h = size[1]
        target_w = size[2]

        if self.target_resolution is None:
            # Keep everything as is
            scaling = 1
            h = target_h
            w = target_w
        else:
            # Computing scaling factor
            scaling = self.target_resolution / self.meta.iloc[r_stack].Resolution_XY
            
            # Compute the height and width and random offsets
            h = round(scaling * target_h)
            w = round(scaling * target_w)
        
        # Correct for stack dimensions
        if self.meta.iloc[r_stack].Width-w == 0:
            x = 0
        elif self.meta.iloc[r_stack].Width-w < 0:
            return
        else:
            x = np.random.choice(self.meta.iloc[r_stack].Width-w)

        # Correct for stack dimensions            
        if self.meta.iloc[r_stack].Height-h == 0:
            y = 0
        elif self.meta.iloc[r_stack].Height-h < 0:
            return
        else:
            y = np.random.choice(self.meta.iloc[r_stack].Height-h)
            
        ## Select random plane + range
        r_plane = np.random.choice(self.meta.iloc[r_stack].Depth-size[0]+1)
        
        z_begin = r_plane
        z_end   = r_plane+size[0]
        
        # Scale if necessary to the correct dimensions
        tmp_stack = self.data['stacks'][f'x{r_stack}'][z_begin:z_end, y:y+h, x:x+w]
        tmp_dendrites = self.data['dendrites'][f'x{r_stack}'][z_begin:z_end, y:y+h, x:x+w]
        tmp_spines = self.data['spines'][f'x{r_stack}'][z_begin:z_end, y:y+h, x:x+w]

        # Data needs to be rescaled
        if scaling != 1:
            return_stack = []
            return_dendrites = []
            return_spines = []
        
            # Do this for each plane
            # and ensure that OpenCV is happy
            for i in range(tmp_stack.shape[0]):
                return_stack.append(cv2.resize(tmp_stack[i], (target_h, target_w)))
                return_dendrites.append(cv2.resize(tmp_dendrites[i].astype(np.uint8), 
                                                 (target_h, target_w)).astype(bool))
                return_spines.append(cv2.resize(tmp_spines[i].astype(np.uint8), 
                                              (target_h, target_w)).astype(bool))
                
            return_stack = np.asarray(return_stack)
            return_dendrites = np.asarray(return_dendrites)
            return_spines = np.asarray(return_spines)
            
        else:
            return_stack = tmp_stack
            return_dendrites = tmp_dendrites
            return_spines = tmp_spines
                
        if squeeze:
            # Return sample
            return return_stack.squeeze(), return_dendrites.squeeze(), return_spines.squeeze()
        else:
            return return_stack, return_dendrites, return_spines