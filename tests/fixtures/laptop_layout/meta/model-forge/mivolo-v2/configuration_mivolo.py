"""A HuggingFace-style model configuration."""
from typing import Any, Dict
from transformers import PretrainedConfig


class MiVOLOConfig(PretrainedConfig):
    model_type = 'mivolo'

    def __init__(
        self,
        model_name: str = "mivolo_d1_384",
        num_classes: int = 3, 
        in_chans: int = 6, 
        input_size: int = 384,
        use_test_size: bool = True,
        with_persons_model: bool = True,
        disable_faces: bool = False,
        use_persons: bool = True,
        only_age: bool = False,
        num_classes_gender: int = 2,
        min_age: int = 0,
        max_age: int = 100,
        avg_age: float = 61.0,
        gender_id2label: Dict = {"0": "male", "1": "female"},
        **kwargs: Any
    ):

        self.model_name = model_name
        self.num_classes = num_classes
        self.in_chans = in_chans

        self.input_size = input_size
        self.use_test_size = use_test_size
        self.num_classes_gender = num_classes_gender

        self.with_persons_model = with_persons_model
        self.disable_faces = disable_faces
        self.use_persons = use_persons
        self.only_age = only_age

        self.min_age = min_age
        self.max_age = max_age
        self.avg_age = avg_age

        self.gender_id2label = {int(key): value for key, value in gender_id2label.items()}
        self.use_face_crops = not self.disable_faces or not self.with_persons_model

        super().__init__(**kwargs)
