from __future__ import annotations

from dataclasses import dataclass
from PIL import Image

import glob

import logging
from typing import Callable, Final, Iterator, Sequence

import torch
import torchvision.transforms as T # type: ignore
from torchvision.transforms.functional import InterpolationMode # type: ignore

imagenet_mean: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
imagenet_std:  Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)

logger = logging.getLogger(__name__)

@dataclass
class ImageLoader:
    """
    ImageLoader is a utility class for loading for deep learning workflows.
    Attributes:
        chunk_size (int): The size (in pixels) of rescaling.
        
    Methods:
        transform(image: Image.Image) -> torch.Tensor:
            Abstract static method for transforming an image to a tensor. Should be implemented via build_transform().
        __post_init__():
            Initializes the image transformation pipeline after object creation.
        convert_to_rgb(image: Image.Image) -> Image.Image:
            Converts the input image to RGB mode if it is not already in RGB.
        build_transform() -> Callable[[Image.Image], torch.Tensor]:
            Constructs and returns a composed image transformation pipeline including resizing, normalization, and conversion to tensor.
        load_image_tensor(image_file: str) -> torch.Tensor:
            Loads an image from a file.
        load_image_tensors(filename_pattern: str | Sequence[str]) -> ImageIterator:
            Loads image tensors from files matching the given filename pattern(s) and returns an iterator over the tensors.
    Inner Classes:
        ImageIterator:
            An iterator class for sequentially loading image tensors from a list of files.
    Usage:
        Instantiate ImageLoader and use load_image_tensor or load_image_tensors to process images for model input.
    Example:
        loader = ImageLoader(chunk_size=448)
        image_tensor = loader.load_image_tensor("path/to/image.jpg")
    """
    
    chunk_size: int = 448
    
    @staticmethod
    def transform(image: Image.Image, /) -> torch.Tensor:
        raise NotImplementedError("This method should be built by build_transform()")
    
    def __post_init__(self):
        self.transform = staticmethod(self.build_transform())
    
    @staticmethod
    def convert_to_rgb(image: Image.Image) -> Image.Image:
        """Convert image to RGB mode if not already."""
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    def build_transform(self) -> Callable[[Image.Image], torch.Tensor]:
        return T.Compose([
            T.Lambda(self.convert_to_rgb),
            T.Resize((self.chunk_size, self.chunk_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

    def load_image_tensor(
        self, image_file: str
    ) -> torch.Tensor:
        """
        Loads an image from the specified file path.
        Args:
            image_file (str): Path to the image file to be loaded.
        Returns:
            torch.Tensor: A tensor containing the transformed image, the shape is [1, 3, chunk_size, chunk_size].
        Logs:
            Information about the image loading process.
        """
        logger.debug(f"Loading image from {image_file}")
        image = Image.open(image_file).convert("RGB")
        return self.transform(image).unsqueeze(0)
    
    @dataclass
    class ImageIterator(Iterator[torch.Tensor]):
        image_loader: ImageLoader
        files: Sequence[str]
        index: int = 0
        
        def __iter__(self):
            return self
        
        def __len__(self):
            return len(self.files)
        
        def __next__(self) -> torch.Tensor:
            if self.index >= len(self.files):
                raise StopIteration
            image_file = self.files[self.index]
            self.index += 1
            return self.image_loader.load_image_tensor(image_file)
        
    def load_image_tensors(
        self, filename_pattern: str | Sequence[str],
    ) -> ImageIterator:
        """
        Loads image tensors from files matching the given filename pattern(s).
        Args:
            filename_pattern (str | Sequence[str]): A filename pattern or a list of patterns to match image files.
                Patterns can include wildcards as supported by `glob`.
        Returns:
            ImageIterator: An iterator over the loaded image tensors.
        Raises:
            FileNotFoundError: If no files are found matching the provided pattern(s).
        Logs:
            Info-level message indicating the number of files found and the patterns used.
        """
        if isinstance(filename_pattern, str):
            filename_pattern = [filename_pattern]
        elif len(filename_pattern) == 0:
            raise FileNotFoundError("No files found matching the pattern.")
        files: list[str] = []
        for pattern in filename_pattern:
            files.extend(glob.glob(pattern))
        if len(files) == 0:
            raise FileNotFoundError("No files found matching the pattern.")

        logger.debug(f"Found {len(files)} files matching the pattern: {filename_pattern}")
        return self.ImageIterator(self, files)