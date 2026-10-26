from transformers.modeling_outputs import ModelOutput
from transformers import PreTrainedModel
from transformers.utils import logging

from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch.nn as nn
import torch

from mivolo.model.create_timm_model import create_model
from .configuration_mivolo import MiVOLOConfig


logger = logging.get_logger(__name__)


@dataclass
class MiVOLOBaseModelOutput(ModelOutput):
    head_outputs: Optional[torch.FloatTensor] = None


@dataclass
class MiVOLOOutput(ModelOutput):
    head_outputs: Optional[torch.FloatTensor] = None
    age_output: Optional[torch.FloatTensor] = None
    raw_age_output: Optional[torch.FloatTensor] = None
    gender_probs: Optional[torch.FloatTensor] = None
    gender_class_idx: Optional[torch.FloatTensor] = None
    raw_gender_output: Optional[torch.FloatTensor] = None


class MiVOLOPreTrainedModel(PreTrainedModel):
    config_class = MiVOLOConfig
    base_model_prefix = 'model'
    _no_split_modules = ['network', 'patch_embed']


    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class MiVOLOModel(MiVOLOPreTrainedModel):
    def __init__(self, config: MiVOLOConfig):
        super().__init__(config)
        self.config = config

        self.model = create_model(
            model_name=config.model_name,
            num_classes=config.num_classes,
            in_chans=config.in_chans,
            pretrained=False,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        faces_input: Optional[torch.Tensor] = None, 
        body_input: Optional[torch.Tensor] = None,
        concat_input: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MiVOLOBaseModelOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if concat_input is None:
            if faces_input is None and body_input is None:
                raise ValueError("You have to specify faces_input or body_input.")

            if self.config.with_persons_model:
                model_input = torch.cat((faces_input, body_input), dim=1)
            else:
                model_input = faces_input

                if faces_input is None:
                    raise ValueError("You have to specify faces_input.")
        else:
            model_input = concat_input
            if concat_input is None:
                raise ValueError("Model input must be not None.")
            
        head_outputs = self.model(model_input)

        if not return_dict:
            return (head_outputs,)

        return MiVOLOBaseModelOutput(
            head_outputs=head_outputs,
        )



class MiVOLOForImageClassification(MiVOLOPreTrainedModel):
    def __init__(self, config: MiVOLOConfig) -> None:
        super().__init__(config)

        self.mivolo = MiVOLOModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        faces_input: Optional[torch.Tensor] = None, 
        body_input: Optional[torch.Tensor] = None,
        concat_input: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, MiVOLOOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mivolo(
            faces_input,
            body_input,
            concat_input=concat_input,
            return_dict=return_dict,
        )
        
        output = outputs[0]
        
        if self.config.only_age:
            raw_age_output = output
            gender_probs, gender_class_idx, raw_gender_output = None, None, None
        else:
            raw_age_output = output[:, self.config.num_classes_gender:]
            raw_gender_output = output[:, :self.config.num_classes_gender]
            gender_output = raw_gender_output.softmax(-1)
            gender_probs, gender_class_idx = gender_output.topk(1)
    

        age_output = raw_age_output * (self.config.max_age - self.config.min_age) + self.config.avg_age
        if not return_dict:
            return (output,) + (age_output, raw_age_output, gender_probs, gender_class_idx, raw_gender_output)

        return MiVOLOOutput(
            head_outputs=output,
            age_output=age_output,
            raw_age_output=raw_age_output,
            gender_probs=gender_probs,
            gender_class_idx=gender_class_idx,
            raw_gender_output=raw_gender_output,
        )
