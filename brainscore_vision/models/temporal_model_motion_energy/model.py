from collections import OrderedDict

import numpy as np
import moten
from PIL import Image
from functools import partial

from brainscore_vision.model_helpers.activations.temporal.model.base import ActivationWrapper


def processing(video, frame_size, fps, pyramid):
    frames = video.set_size(frame_size).set_fps(fps).to_numpy()
    luminance = moten.io.imagearray2luminance(frames, size=frame_size)
    return pyramid.project_stimulus(luminance)


class MotionEnergyModel(ActivationWrapper):
    def __init__(self, frame_size=(224, 224), fps=25):
        self.frame_size = frame_size
        self.fps = fps
        self.pyramid = moten.get_default_pyramid(vhsize=frame_size, fps=fps)
        self.process = partial(processing, frame_size=frame_size, fps=fps, pyramid=self.pyramid)
        super().__init__(
            identifier='motion-energy', preprocessing=self.process, 
            fps=fps, layer_activation_format={'motion-energy': 'TC'}
        )

    def get_activations(self, inputs, layers):
        assert layers == ['motion-energy']
        motion_energy = np.array(inputs)
        return OrderedDict([('motion-energy', motion_energy)])

