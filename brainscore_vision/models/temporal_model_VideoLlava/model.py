import torch
import numpy as np
from torchvision import transforms
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

from brainscore_vision.model_helpers.activations.temporal.model.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file



class VideoLlavaWrapper(PytorchWrapper):
    def forward(self, inputs):
        inputs = torch.stack(inputs).to(self._device)
        result = self._model.get_video_features(inputs, vision_feature_layer=self._model.config.vision_feature_layer)
        return result


def get_model(identifier="VideoLlava"):
    T = 8
    inferencer_kwargs = {
        "fps": 2,
        "num_frames": T, # the model's video encoder is just frame-by-frame CLIP image encoder
        "layer_activation_format": 
        {
            "video_tower.vision_model.embeddings": "THWC",
            **{f"video_tower.vision_model.encoder.layers.{i}": "THWC" for i in range(24)},
        },
    }
    
    def process_activation(layer, layer_name, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        output = output[:, 1:]  # remove the [CLS] token
        output # [batch_size*T, weight*height, num_features]
        output = output.reshape(output.shape[0]//T, T, 16, 16, output.shape[-1])
        return output

    process_output = process_activation

    name = "LanguageBind/Video-LLaVA-7B-hf"
    model = VideoLlavaForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16)
    processor = VideoLlavaProcessor.from_pretrained(name)
    def process_input(input):
        arr = input.to_numpy()
        ret = processor(text='', videos=arr, return_tensors="pt")
        return ret['pixel_values_videos'][0]

    wrapper = VideoLlavaWrapper(identifier, model, process_input, 
                             process_output=process_output,
                             **inferencer_kwargs)

    return wrapper