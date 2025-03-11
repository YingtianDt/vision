import yaml
import os

import torch

import ijepa.models.vision_transformer as models
from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.activations.temporal.core.inferencer import Inferencer
from brainscore_vision.model_helpers.activations.temporal.inputs import Image
from brainscore_vision.model_helpers.activations.temporal.utils import download_weight_file


LARGE_MODEL_LAYER_STEP = 4

normalization=((0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))

def get_transform_image(size):
    from torchvision import transforms

    transform_img = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.Normalize(*normalization)
        ])

    def transform_image(image):
        image = image.to_numpy() / 255.
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)
        return transform_img(image)
    return transform_image


def get_model(identifier):
    
    if identifier == 'IJEPA-ViT-H14-ImageNet1K':
        model_cls = models.vit_huge
        patch_size = 14
        resolution = 224
        num_layers = 32
        weight_path = download_weight_file('https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar', folder='ijepa')
    else:
        raise ValueError(f"Unknown model identifier: {identifier}")

    model = model_cls(patch_size=patch_size)
    state = torch.load(weight_path, map_location='cpu')["encoder"]
    new_state = {}
    for k in list(state.keys()):
        if k.startswith("module."):
            new_state[k[7:]] = state[k]
    model.load_state_dict(new_state)
    image_transform = get_transform_image(resolution)
    
    layer_activation_format = {
        "patch_embed": "CHW",
        **{f"blocks.{i}": "CHW" for i in range(0, num_layers, LARGE_MODEL_LAYER_STEP)},
    }

    return PytorchWrapper(identifier, model, image_transform, inferencer_cls=Inferencer, layer_activation_format=layer_activation_format, stimulus_type=Image)