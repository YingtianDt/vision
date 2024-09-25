import numpy as np
import math
from collections import OrderedDict
from tqdm import tqdm

from .base import TemporalContextInferencerBase
from brainscore_vision.model_helpers.activations.temporal.utils import stack_with_nan_padding, get_mem
from brainio.assemblies import NeuroidAssembly


class BlockInferencer(TemporalContextInferencerBase):
    """Inferencer that divides the original video into smaller blocks and does inference on the blocks separately.
    Finally, the activations are joint along the temporal dimension for the final activations. 

    Specifically, suppose the video lasts for 1000ms and the block size is 400ms.
    Then, the video is segmented into [0~400ms], [400~800ms], [800~1200ms] (1000~1200ms padded).
    The activations for each segment will be stacked together.

    The block size is determined by the temporal parameters (num_frames & duration) and temporal_context_strategy.
    If num_frames or duration is given, the model's temporal context will be set to match the two.
    """
    
    def __call__(self, paths, layers, mmap_path=None):
        _, context = self._compute_temporal_context()

        if np.isinf(context):
            return super().__call__(paths, layers, mmap_path)

        EPS = 1e-6
        interval = 1000 / self.fps - EPS
        stimuli = self.load_stimuli(paths)
        longest_stimulus = stimuli[np.argmax(np.array([stimulus.duration for stimulus in stimuli]))]
        num_time_bins = int(int(math.ceil(longest_stimulus.duration / context)) * context / interval)
        num_frames_per_block = int(context / interval)
        num_stimuli = len(paths)
        time_bin_coords = self._get_time_bin_coords(num_time_bins, self.fps)
        stimulus_paths = paths

        t_offsets = []
        stimulus_index = []
        for s, stimulus in enumerate(stimuli):
            duration = stimulus.duration
            videos = []
            # for each stimulus, divide it into block clips with the specified context
            for time_start in np.arange(0, duration, context):
                time_end = time_start + context
                clip = stimulus.set_window(time_start, time_end, padding=self.out_of_bound_strategy)
                t_offsets.append(int(time_start / interval))
                videos.append(clip)
                stimulus_index.append(s)
            self._executor.add_stimuli(videos)

        data = None
        for temporal_layer_activations, indicies in self._executor.execute_batch(layers):
            for temporal_layer_activation, i in zip(temporal_layer_activations, indicies):
                s = stimulus_index[i]
                # determine the time bin correspondence for each layer
                for t, layer_activation in self._disect_time(temporal_layer_activation, num_frames_per_block):
                    if data is None:
                        num_feats, neuroid_coords = self._get_neuroid_coords(layer_activation, self._remove_T(self.layer_activation_format))
                        data = get_mem(mmap_path, shape=(num_stimuli, num_time_bins, num_feats), dtype=self.dtype)
                    flatten_activation = self._flatten_activations(layer_activation)
                    t = t_offsets[i] + t
                    data[s, t, :] = flatten_activation

        model_assembly = NeuroidAssembly(
            data.load(), 
            dims=["stimulus_path", "time_bin", "neuroid"],
            coords={
                "stimulus_path": stimulus_paths, 
                **neuroid_coords,
                **time_bin_coords,
            }, 
        )

        return model_assembly