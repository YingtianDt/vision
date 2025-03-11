import torch
import numpy as np
from torchvision import transforms
from transformers import VivitModel, VivitImageProcessor

from brainscore_vision.model_helpers.activations.temporal.model.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file


def get_model(identifier="ViViT"):
    inferencer_kwargs = {
        "fps": 12.5,  # common YouTube frame rate
        "num_frames": 32,
        "layer_activation_format": 
        {
            "embeddings": "THWC",
            **{f"encoder.layer.{i}": "THWC" for i in range(12)},
        },
    }
    
    def process_activation(layer, layer_name, inputs, output):
        # (torch.nn.Module, str, torch.Tensor, torch.Tensor) -> torch.Tensor
        if layer_name.startswith("embeddings") or layer_name.startswith("encoder.layer"):
            if layer_name.startswith("encoder.layer"):
                output = output[0]
            output = output[:, 1:]  # remove the [CLS] token
            output # [batch_size, num_frames*weight*height, num_features]
            output = output.reshape(output.shape[0], -1, 14, 14, output.shape[-1])
        return output

    process_output = process_activation

    name = "google/vivit-b-16x2-kinetics400"
    model = VivitModel.from_pretrained(name, attn_implementation="sdpa", torch_dtype=torch.float16)
    processor = VivitImageProcessor.from_pretrained(name)
    def process_input(input):
        arr = input.to_numpy()
        ret = processor.preprocess([a for a in arr], return_tensors="pt")
        ret = ret['pixel_values'][0].to(torch.float16)
        return ret

    wrapper = PytorchWrapper(identifier, model, process_input, 
                             process_output=process_output,
                             **inferencer_kwargs)
    
    return wrapper