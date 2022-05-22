import albumentations as A
import numpy as np


class Albument:
    def __init__(self, augment: A.Compose) -> None:
        """Class for performing albumentations augmentations.
        Args:
            - :augment: (Compose) a list of albumentations transforms
        """
        self.augment = augment
    def __call__(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Perform tranformations on given image
        Args:
            - :img: (np.ndarray) image to transform
        Returns:
            - np.ndarray: transformed image
        """
        aug = self.augment(image=img, mask=mask)
        return aug["image"], aug["mask"]
