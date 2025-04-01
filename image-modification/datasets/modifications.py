import torch
import numpy as np


def add_gaussian_noise(img: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """
    Add Gaussian noise to the image.
    """
    noise = torch.randn_like(img) * std
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0, 1)


def add_random_mask(
    img: torch.Tensor,
    mask_size: tuple[int, int] = (64, 64),
    num_patches: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random mask and apply it to the image.
    """
    _, h, w = img.shape
    mask = torch.ones((1, h, w), dtype=img.dtype, device=img.device) 

    for _ in range(num_patches):
        # Randomly select the top-left corner of each mask
        x = np.random.randint(0, w - mask_size[0])
        y = np.random.randint(0, h - mask_size[1])

        # Set mask region to zero (black)
        mask[:, y:y+mask_size[1], x:x+mask_size[0]] = 0

    # Set mask region to zero (black)
    mask[:, y:y+mask_size[1], x:x+mask_size[0]] = 0

    # Apply mask to the image
    masked_img = img * mask

    return masked_img, mask
