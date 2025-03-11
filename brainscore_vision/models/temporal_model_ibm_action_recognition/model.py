from timm import create_model
from ibm_action_recognition import build_model
import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.activations.temporal.utils import download_weight_file
from torchvision import transforms


input_mean = [0.485, 0.456, 0.406] # IMAGENET_DEFAULT_MEAN
input_std = [0.229, 0.224, 0.225] # IMAGENET_DEFAULT_STD
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=input_mean, std=input_std),
])

def transform_video(video):
    import torch
    frames = torch.Tensor(video.to_numpy() / 255.0).permute(0, 3, 1, 2)
    frames = transform_img(frames)
    return frames.permute(1, 0, 2, 3)


def get_model(identifier):
    kwargs = {
        "num_classes": 174,
        "dropout": 0.5,
        "without_t_stride": False,
    }

    if identifier == "I3D-R50-smthsmthv2":
        backbone_net = "i3d_resnet"
        kwargs["depth"] = 50
        num_layers = 4
        weight_pth = "https://github.com/IBM/action-recognition-pytorch/releases/download/weights-v0.1/SSV2-I3D-ResNet-50-f32.pth.tar"
    elif identifier == "I3D-R101-smthsmthv2":
        backbone_net = "i3d_resnet"
        kwargs["depth"] = 101
        num_layers = 4
        weight_pth = "https://github.com/IBM/action-recognition-pytorch/releases/download/weights-v0.1/SSV2-I3D-ResNet-101-f32.pth.tar"
    elif identifier == "TAM-R50-smthsmthv2":
        backbone_net = "resnet"
        kwargs["depth"] = 50
        kwargs["temporal_module_name"] = "TAM"
        weight_pth = "https://github.com/IBM/action-recognition-pytorch/releases/download/weights-v0.1/SSV2-TAM-ResNet-50-f32.pth.tar"
    elif identifier == "TAM-R101-smthsmthv2":
        backbone_net = "resnet"
        kwargs["depth"] = 101
        kwargs["temporal_module_name"] = "TAM"
        weight_pth = "https://github.com/IBM/action-recognition-pytorch/releases/download/weights-v0.1/SSV2-TAM-ResNet-101-f32.pth.tar"

    weight_pth = download_weight_file(weight_pth)
    
    net = build_model(backbone_net, **kwargs)
    state_dict = th.load(weight_pth, map_location="cpu")["state_dict"]
    net.load_state_dict(state_dict)

    if backbone_net == "i3d_resnet":
        inferencer_kwargs = {
            "fps": 25,
            "layer_activation_format": {
                "conv1": "CTHW",
                **{f"layer{i}" : "CTHW" for i in range(1, 1+num_layers)},
            },
        }
    elif backbone_net == "resnet":
        raise NotImplementedError("ResNet model is not supported yet.")

    wrapper = PytorchWrapper(identifier, net, transform_video, 
                                **inferencer_kwargs)
    return wrapper
