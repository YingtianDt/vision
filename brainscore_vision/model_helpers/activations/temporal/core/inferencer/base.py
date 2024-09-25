import functools
import logging
from collections import OrderedDict
from typing import Callable, Hashable, List, Dict, Any, Union
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
import gc

from brainio.assemblies import NeuroidAssembly, walk_coords
from brainscore_vision.model_helpers.utils import fullname

from brainscore_vision.model_helpers.activations.temporal.core.executor import BatchExecutor
from brainscore_vision.model_helpers.activations.temporal.utils import stack_with_nan_padding, batch_2d_resize, get_mem
from brainscore_vision.model_helpers.activations.temporal.inputs import Stimulus


channel_name_mapping = {
    "T": "channel_temporal",
    "C": "channel",
    "H": "channel_y",
    "W": "channel_x",
    "K": "channel_token"
}


class Inferencer:
    """Inferencer for batch processing of stimuli and packaging the activations.
    
    Parameters
    ----------
    get_activations : function 
        function that takes a list of processed stimuli and a list of layers, and returns a dictionary of activations.
    preprocessing : function
        function that takes a stimulus and returns a processed stimulus.
    stimulus_type: Stimulus
        the type of the stimulus.
    layer_activation_format: dict
        a dictionary that specifies the dimensions of the activations of each layer.
        For example, {"temp_conv": "TCHW", "spatial_conv": "CHW",  "fc": "C"}.
    visual_degrees: float
        the visual degrees of the stimuli.
    max_spatial_size: int/float
        the maximum spatial size of the activations. If the spatial size of the activations is larger than this value,
        the activations will be downsampled to this size. This is used to avoid the large memory consumption by the first layers of some model.
        If float, resize the image based on this factor.
    dtype: np.dtype
        data type of the activations.
    batch_size: int
        number of stimuli to process in each batch.
    batch_grouper: function
        function that takes a stimulus and return the property based on which the stimuli can be grouped.
    batch_padding: bool
        whether to pad the each batch with the last stimulus to make it the same size as the specified batch size.
        Otherwise, some batches will have size < batch_size because of the lacking of samples in that group.
    max_workers: int
        the maximum number of workers to use for parallel processing.

        
    APIs
    ----
    __call__(paths, layers, mmap_path=None)
        process the stimuli and return a holistic assembly that compile activations of all specified layers.
        The returned assembly is a NeuroidAssembly with the dimensions [stimulus_path, neuroid].
        All dimensions of the activations will be stacked together to form the "neuroid" dimension.
        If mmap_path is specified, the activations will be saved to the mmap file.

    Examples
    --------
    >>> inferencer = Inferencer(get_activations, preprocessing, Video, batch_size=64, batch_grouper=lambda s: s.duration)
    >>> model_assembly = inferencer(stimulus_paths, layers)
    """

    def __init__(
            self, 
            get_activations : Callable[[List[Any]], Dict[str, np.array]], 
            preprocessing : Callable[[List[Stimulus]], Any],
            layer_activation_format : dict,
            stimulus_type : Stimulus,
            visual_degrees : float = 8.,
            max_spatial_size : Union[int, float] = None,
            dtype : np.dtype = np.float16,
            batch_size : int = 64,
            batch_grouper : Callable[[Stimulus], Hashable] = None,
            batch_padding : bool = False,
            max_workers : int = None,
            *args,
            **kwargs
        ):

        self.stimulus_type = stimulus_type
        self.layer_activation_format = layer_activation_format
        if isinstance(max_spatial_size, float):
            assert max_spatial_size < 1, "a proporational max_spatial_size should be < 1."
        self.max_spatial_size = max_spatial_size
        self.visual_degrees = visual_degrees
        self.dtype = dtype
        self._executor = BatchExecutor(get_activations, preprocessing, batch_size, batch_padding, batch_grouper, max_workers)
        self._stimulus_set_hooks = {}
        self._batch_activations_hooks = {}
        self._logger = logging.getLogger(fullname(self))

        # register hooks
        self._executor.register_after_hook(self._make_spatial_downsample_hook(max_spatial_size))
        self._executor.register_after_hook(self._make_dtype_hook(dtype))

    @property
    # identifier for the inferencer: including all the features that may affect the activations
    def identifier(self) -> str:
        to_add = [
            f".dtype={self.dtype.__name__}",
            f".vdeg={self.visual_degrees}",
        ]
        if self.max_spatial_size is not None:
            to_add.append(f".max_s={self.max_spatial_size}")
        to_add = "".join(to_add)
        return f"{self.__class__.__name__}{to_add}"
    
    def set_visual_degrees(self, visual_degrees: float):
        self.visual_degrees = visual_degrees
        print("Visual degrees not supported yet. Bypassing...")

    # given the paths of the stimuli and the layers, return the model activations as a NeuroidAssembly
    def __call__(self, paths: List[Union[str, Path]], layers: List[str], mmap_path: str = None) -> NeuroidAssembly:
        stimuli = self.load_stimuli(paths)
        num_stimuli = len(paths)
        stimulus_paths = paths

        self._executor.add_stimuli(stimuli)
        data = None
        
        for layer_activations, indicies in self._executor.execute_batch(layers):
            for layer_activation, i in zip(layer_activations, indicies):
                if data is None:
                    num_feats, neuroid_coords = self._get_neuroid_coords(layer_activation, self.layer_activation_format)
                    data = get_mem(mmap_path, shape=(num_stimuli, num_feats), dtype=self.dtype)
                flatten_activation = self._flatten_activations(layer_activation)
                data[i, :] = flatten_activation

        model_assembly = NeuroidAssembly(
            data, 
            dims=["stimulus_path", "neuroid"],
            coords={
                "stimulus_path": stimulus_paths, 
                **neuroid_coords
            }, 
        )

        return model_assembly

    def load_stimuli(self, paths : List[Union[str, Path]]) -> List[Stimulus]:
        ret = []
        for p in tqdm(paths, desc="Loading stimuli"):
            ret.append(self.load_stimulus(p))
        return ret

    def load_stimulus(self, path : Union[str, Path]) -> Stimulus:
        return self.stimulus_type.from_path(path)
  
    def _flatten_activations(self, layer_activation):
        arrs = [arr.flatten() for arr in layer_activation.values()]
        return np.concatenate(arrs)

    def _get_neuroid_coords(self, layer_activation, layer_specs):
        feat_sizes = [arr.size for arr in layer_activation.values()]
        feat_shapes = [arr.shape for arr in layer_activation.values()]
        layers = list(layer_activation.keys())
        num_feats = sum(feat_sizes)
        count_neuroids = 0
        neuroid_coords = {}
        neuroid_coords['neuroid_id'] = []
        neuroid_coords['neuroid_num'] = []
        neuroid_coords['layer'] = []

        def _grid(*ns):
            from itertools import product
            ns = [range(n) for n in ns]
            return product(*ns)

        def _expand_spec(spec, shape):
            assert len(spec) == len(shape)
            return [".".join([f"{dim}{i}" for dim, i in zip(spec, grid)]) for grid in _grid(*shape)]

        # neuroid_id, neuroid_num, layer
        for layer, size, shape in zip(layers, feat_sizes, feat_shapes):
            layer_spec = layer_specs[layer]
            layer_spec = [(dim, s) for dim, s in zip(layer_spec, shape)]
            neuroid_ids = [f"{layer}.{nid}" for nid in _expand_spec(*list(zip(*layer_spec)))]
            neuroid_nums = list(range(count_neuroids, count_neuroids+size))
            count_neuroids += size
            layers = [layer] * size

            neuroid_coords['neuroid_id'].extend(neuroid_ids)
            neuroid_coords['neuroid_num'].extend(neuroid_nums)
            neuroid_coords['layer'].extend(layers)

        for key, val in neuroid_coords.items():
            neuroid_coords[key] = ('neuroid', val)

        return num_feats, neuroid_coords
  
    # turn the activations into the specified dtype
    def _make_dtype_hook(self, dtype):
        return lambda val, layer, stimulus: val.astype(dtype)
    
    # downsample the activations with the largest spatial size (among width and height) to the specified size
    def _make_spatial_downsample_hook(self, max_spatial_size, mode="pool"):
        def hook(val, layer, stimulus):
            if max_spatial_size is None:
                return val
            
            dims = self.layer_activation_format[layer]

            # require both H and W dimensions to do spatial downsampling
            if "H" not in dims or "W" not in dims:
                return val

            H_dim, W_dim = dims.index("H"), dims.index("W")
            val = val.swapaxes(H_dim, 0).swapaxes(W_dim, 1)
            shape = val.shape[2:]
            h, w = val.shape[:2]
            val = val.reshape(h, w, -1)
            new_size = _compute_new_size(w, h, self.max_spatial_size)
            new_val = batch_2d_resize(val[None,:], new_size, mode=mode)[0]
            new_val = new_val.reshape(*new_size, *shape)
            new_val = new_val.swapaxes(0, H_dim).swapaxes(1, W_dim)
            return new_val.astype(self.dtype)
        return hook


def _compute_new_size(w, h, max_spatial_size):
    if isinstance(max_spatial_size, int):
        if h > w:
            new_h = max_spatial_size
            new_w = int(w * new_h / h)
        else:
            new_w = max_spatial_size
            new_h = int(h * new_w / w)
    else:
        new_h = int(h * max_spatial_size)
        new_w = int(w * max_spatial_size)
    
    new_h = max(1, new_h)
    new_w = max(1, new_w)

    return new_h, new_w
