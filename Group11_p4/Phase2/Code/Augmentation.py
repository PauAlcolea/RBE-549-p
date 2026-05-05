import torch
import random


class VisualOdometryAugmentation:
    """
    Safe augmentations for visual odometry that preserve geometric relationships.
    Only applies photometric transformations that don't invalidate pose labels.
    """

    def __init__(
        self,
        brightness_range=(0.7, 1.3),
        contrast_range=(0.7, 1.3),
        saturation_range=(0.7, 1.3),
        hue_range=(-0.1, 0.1),
        gaussian_noise_std=0.02,
        apply_prob=0.8,
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.gaussian_noise_std = gaussian_noise_std
        self.apply_prob = apply_prob

    def __call__(self, image):
        """
        Apply augmentations to a single image tensor.
        
        Args:
            image: (C, H, W) tensor in [0, 1] range
            
        Returns:
            Augmented image tensor
        """
        if random.random() > self.apply_prob:
            return image

        # Clone to avoid in-place modifications
        img = image.clone()

        # Color jitter (applied in RGB space)
        if random.random() > 0.5:
            # Brightness
            brightness_factor = random.uniform(*self.brightness_range)
            img = img * brightness_factor

        if random.random() > 0.5:
            # Contrast (around mean)
            contrast_factor = random.uniform(*self.contrast_range)
            mean = img.mean(dim=[1, 2], keepdim=True)
            img = (img - mean) * contrast_factor + mean

        # Gaussian noise
        if random.random() > 0.5:
            noise = torch.randn_like(img) * self.gaussian_noise_std
            img = img + noise

        # Clamp to valid range
        img = torch.clamp(img, 0.0, 1.0)

        return img


class PairAugmentation:
    """
    Augmentation wrapper for image pairs.
    Applies the SAME augmentation to both images to maintain temporal consistency.
    """

    def __init__(self, base_augmentation):
        self.aug = base_augmentation

    def __call__(self, image_t, image_tp1):
        """
        Apply same random augmentation to both images in a pair.
        
        Args:
            image_t: (C, H, W) tensor
            image_tp1: (C, H, W) tensor
            
        Returns:
            Tuple of augmented (image_t, image_tp1)
        """
        # Sample augmentation parameters once
        if random.random() > self.aug.apply_prob:
            return image_t, image_tp1

        # Apply same transformation to both
        seed = random.randint(0, 2**32 - 1)
        
        random.seed(seed)
        torch.manual_seed(seed)
        aug_t = self._apply_aug(image_t)
        
        random.seed(seed)
        torch.manual_seed(seed)
        aug_tp1 = self._apply_aug(image_tp1)

        return aug_t, aug_tp1

    def _apply_aug(self, image):
        img = image.clone()

        # Sample parameters
        brightness_factor = random.uniform(*self.aug.brightness_range)
        contrast_factor = random.uniform(*self.aug.contrast_range)
        noise_std = self.aug.gaussian_noise_std

        # Brightness
        if random.random() > 0.5:
            img = img * brightness_factor

        # Contrast
        if random.random() > 0.5:
            mean = img.mean(dim=[1, 2], keepdim=True)
            img = (img - mean) * contrast_factor + mean

        # Gaussian noise
        if random.random() > 0.5:
            noise = torch.randn_like(img) * noise_std
            img = img + noise

        return torch.clamp(img, 0.0, 1.0)