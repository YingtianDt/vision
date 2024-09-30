import yaml
import os

import torch
from torchvision import transforms

import video_chat
import video_chat2

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file


LARGE_MODEL_LAYER_STEP = 2

VIDEOCHAT_HOME = os.path.dirname(os.path.abspath(video_chat.__file__))
VIDEOCHAT2_HOME = os.path.dirname(os.path.abspath(video_chat2.__file__))
NUM_FRAMES = 8
IMAGE_SIZE = 224
FPS = 2

input_mean = [0.48145466, 0.4578275, 0.40821073]
input_std = [0.26862954, 0.26130258, 0.27577711]

def get_transform_video(transform_img):
    def transform_video(video):
        frames = video.to_numpy() / 255.
        frames = torch.Tensor(frames)
        frames = frames.permute(0, 3, 1, 2)
        frames = transform_img(frames)
        return frames.permute(1, 0, 2, 3)
    return transform_video

img_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize(
        mean=input_mean,
        std=input_std
    )
])

video_transform = get_transform_video(img_transform)


class VideoChatWrapper(PytorchWrapper):
    def __init__(self, version, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version = version

    def forward(self, inputs):
        import torch
        tensor = torch.stack(inputs)
        tensor = tensor.to(self._device)
        if self.version == 2:
            return self._model.vision_encoder(tensor, use_image=False)
        else:
            return self._model.visual_encoder(tensor)


def get_model(identifier, fps=FPS):
    
    if identifier == 'VideoChat2-7B':
        cfg_path = os.path.join(VIDEOCHAT2_HOME, "scripts/config_7b_stage3.py")

        from video_chat2.utils.config import Config
        from video_chat2.models.videochat2_it import VideoChat2_it

        weight_path = load_weight_file(
            bucket="brainscore-vision", 
            relative_path="temporal_model_ask_anything/videochat2_7b_stage3.pth", 
            version_id="sBJPIy7r_6pTLJrw.Znt6Y7iJ7OQpmfu",
            sha1="a7f85bcf28c89548a33e02ac56efa007fb680637"
        )

        num_frames = 8
        cfg = Config.from_file(cfg_path)
        cfg.model.vision_encoder["num_frames"] = num_frames
        cfg.model.vit_blip_model_path = None
        cfg.model.videochat2_model_path = weight_path
        model = VideoChat2_it(cfg.model)

        patch_size = cfg.model.vision_encoder.patch_size
        num_patches = IMAGE_SIZE // patch_size
        num_vit_layers = 23
        num_qformer_layers = 12

        def process_output(layer, layer_name, inputs, output):
            B = output.shape[0]
            C = output.shape[-1]
            return output.reshape(B, -1, num_patches, num_patches, C)

        layer_activation_format = {
            'vision_encoder.encoder.patch_embed': 'THWC',
            **{f'vision_encoder.encoder.blocks.{i}': 'THWC' for i in range(0, num_vit_layers, LARGE_MODEL_LAYER_STEP)},
            # **{f'qformer.bert.encoder.layer.{i}': 'THWC' for i in range(0, num_qformer_layers, LARGE_MODEL_LAYER_STEP)},  # qformer is part of language modeling
        }
        version = 2


    elif identifier in ["VideoChat-7B", "VideoChat-13B"]:

        if identifier == "VideoChat-7B":
            cfg_path = os.path.join(VIDEOCHAT_HOME, "configs/config_7b.json")
            weight_path = load_weight_file(
                bucket="brainscore-vision", 
                relative_path="temporal_model_ask_anything/videochat_7b_stage1.pth", 
                version_id="Kdhmt2fYI1KSbFbkvLU.cdvXgqyGY2n8",
                sha1="4c6bab988cdc4341b7d3e23d8d1496144ae35946"
            )
            num_gmhra_layers = 8

        elif identifier == "VideoChat-13B":
            cfg_path = os.path.join(VIDEOCHAT_HOME, "configs/config.json")
            weight_path = load_weight_file(
                bucket="brainscore-vision", 
                relative_path="temporal_model_ask_anything/videochat_13b_stage1.pth", 
                version_id="qCmeZuvS8ssTSNszx_AlyfSSY1oX_9AR",
                sha1="a2663211eafe4f884a205f65f34c133ff1903225"
            )
            num_gmhra_layers = 8

        from video_chat.utils.config import Config
        from video_chat.models.videochat import VideoChat

        vit_weight_path = load_weight_file(
                bucket="brainscore-vision", 
                relative_path="temporal_model_ask_anything/eva_vit_g.pth", 
                version_id="EFHoZMQNiiz2MW.sXrFNwrr8y92gus1o",
                sha1="fde241b52f4616df880f8a7fb8094b287b69ea1c"
            )
        cfg = Config.from_file(cfg_path)
        cfg.model.videochat_model_path = weight_path
        model = VideoChat(cfg.model)
        ckpt = torch.load(vit_weight_path, map_location="cpu")
        ckpt['patch_embed.proj.weight'] = ckpt['patch_embed.proj.weight'][:, :, None, ...]
        msg = model.visual_encoder.load_state_dict(ckpt, strict=False)
        print(msg)

        patch_size = 14
        num_vit_layers = 39
        num_patches = IMAGE_SIZE // patch_size
        num_frames = None
        T = None

        def process_output(layer, layer_name, inputs, output):
            global T
            if layer_name == "visual_encoder.patch_embed":
                T = output.shape[2]

            if layer_name.startswith("visual_encoder.blocks"):
                BT = output.shape[0]
                C = output.shape[-1]
                output = output[:, 1:]  # remove cls token
                output = output.reshape(BT//T, T, num_patches, num_patches, C)

            if layer_name.startswith("visual_encoder.gmhra"):
                output = output[0]  # [1, B, C] -> [B, C]

            return output

        layer_activation_format = {
            'visual_encoder.patch_embed': 'CTHW',
            **{f'visual_encoder.blocks.{i}': 'THWC' for i in range(0, num_vit_layers, LARGE_MODEL_LAYER_STEP*2)},
            **{f'visual_encoder.gmhra.{i}': 'C' for i in range(0, num_gmhra_layers, LARGE_MODEL_LAYER_STEP)},
        }

        version = 1

    if fps != FPS:
        identifier = f"{identifier}-fps{fps}"

    return VideoChatWrapper(version, identifier, model, video_transform, process_output=process_output,
                fps=fps, num_frames=num_frames, layer_activation_format=layer_activation_format)