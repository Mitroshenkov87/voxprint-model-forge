from typing import List, Optional
import numpy as np

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import is_numpy_array

from mivolo.data.misc import prepare_classification_images


class MiVOLOImageProcessor(BaseImageProcessor):
    
    model_input_names = ["pixel_values"]

    def __init__(
        self, 
        input_size: int = 384, 
        mean: List[float] = [0.485, 0.456, 0.406], 
        std: List[float] = [0.229, 0.224, 0.225], 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.mean = mean
        self.std = std
        self.input_size = input_size

    def preprocess(
        self,
        images: List[Optional[np.ndarray]],
    ):
        # Transformations expect numpy arrays or None.
        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type List[numpy.ndarray]."
            )
        
        input = prepare_classification_images(
            images, self.input_size, self.mean, self.std
        )

        data = {"pixel_values": input}
        return BatchFeature(data=data, tensor_type="pt")


def valid_images(imgs):
    # If we have an list of images, make sure every image is valid
    if isinstance(imgs, (list, tuple)):
        for img in imgs:
            if img is None:
                continue
            if not is_numpy_array(img):
                return False
    else:
        return False
    return True
    