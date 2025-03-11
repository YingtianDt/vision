import torch
import os
import numpy as np
from torchvision import transforms
from dorsalnet import dorsalnet

from brainscore_vision.model_helpers.activations.temporal.model.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file


img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

def transform_video(video):
    frames = video.to_numpy() / 255.
    frames = torch.Tensor(frames)
    frames = frames.permute(0, 3, 1, 2)
    frames = img_transform(frames)
    return frames.permute(1, 0, 2, 3)


def get_model(identifier="DorsalNet"):
    inferencer_kwargs = {
        "fps": 30,  # from paper
        "layer_activation_format": 
        {
            "conv1": "CTHW",
            "s1": "CTHW",
            **{f"res{i}": "CTHW" for i in range(4)},
        },
    }

    process_output = None
    model_pth = load_weight_file(
            bucket="brainscore-vision",
            relative_path='temporal_model_dorsalnet/airsim_dorsalnet_batch2_model.ckpt-3174400-2021-02-12 02-03-29.666899.pt',
            version_id="waDm9AZImNwuaXlvi5x48_j2I3J7CxLg",
            sha1="c1328afb918d25ee3ceabb5b19629dd7973f5ffb"
        )

    # Instantiate the model
    model = dorsalnet.DorsalNet(symmetric=False, nfeats=32)

    # Load the model weights
    st = torch.load(model_pth, map_location=torch.device('cpu'))
    st = {k.replace("subnet.", ""): v for k, v in st.items() if "fully_connected" not in k}
    model.load_state_dict(st)

    wrapper = PytorchWrapper(identifier, model, transform_video, 
                             process_output=process_output,
                             **inferencer_kwargs)
    
    return wrapper