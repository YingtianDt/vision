import h5py
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import video as vid

from brainscore_vision.model_helpers.activations.temporal.model.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file


class MotionNet(nn.Module):
    """Modified and copied from: https://github.com/patrickmineault/your-head-is-there-to-move-you-around
    
    An PyTorch implementation of Rideaux & Welchman (2020).
    
    https://www.jneurosci.org/content/40/12/2538

    I saved their Tensorflow checkpoint as a pickle file to remove any 
    dependency on TF1.

    Note that I re-interpret their final readout layer as a convolutional layer
    by repeating the fully connected action all over space.
    """
    def __init__(self, ckpt_path):
        super().__init__()

        # load from h5py
        with h5py.File(ckpt_path, 'r') as f:
            results = {k: np.array(v) for k, v in f.items()}

        self.conv1 = nn.Conv3d(1, 
                              128, 
                              (6, 6, 6),
                              (1, 1, 1),
                              padding=(3, 3, 3)
                             )

        self.conv1.weight.data = torch.tensor(
            results['wconv:0'].transpose((3, 2, 1, 0))
        ).unsqueeze(1)

        self.conv1.bias.data = torch.tensor(results['bconv:0'])

        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(128,
                               64,
                               (1, 27, 27),
                               (1, 9, 9),
                               padding=(0, 17, 17))

        self.conv2.weight.data = torch.tensor(
            results['wout:0'].reshape(27, 27, 128, 64).transpose((3, 2, 1, 0))
        ).unsqueeze(2)

        self.conv2.bias.data = torch.tensor(results['bout:0'])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = X.mean(axis=1, keepdims=True)
        X = self.conv1(X)
        X = X[:, :, :-1, :, :]
        X = self.relu(X)
        X = self.conv2(X)
        X = self.softmax(X)

        return X


img_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

def transform_video(video):
    frames = video.to_numpy() / 255.
    frames = torch.Tensor(frames)
    frames = frames.permute(0, 3, 1, 2)
    frames = img_transform(frames)
    return frames.permute(1, 0, 2, 3)  # CTHW

def get_model(identifier="MotionNet"):
    assert identifier == "MotionNet"
    inferencer_kwargs = {
        "fps": 25,
        "layer_activation_format": 
        {
            "relu": "CTHW",
            "conv2": "CTHW",
        },
    }

    process_output = None

    model_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="temporal_model_motion_net/motion_net.h5", 
            version_id="bIn4_N3JFXNt6WAapPw4ttpRS5jsNbpM",
            sha1="97202eb361714d030452301229dc55db9639ad01"
        )
    model = MotionNet(model_path)

    wrapper = PytorchWrapper(identifier, model, transform_video, 
                             process_output=process_output,
                             **inferencer_kwargs)
    
    return wrapper