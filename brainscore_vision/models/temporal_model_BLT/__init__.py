from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.temporal.utils import get_specified_layers
from brainscore_vision.model_interface import BrainModel
from . import model


def commit_model(identifier):
    activations_model=model.get_model(identifier)
    layers=model.LAYERS
    return ModelCommitment(identifier=identifier, activations_model=activations_model, layers=layers)


model_registry["blt_temporal"] = lambda: commit_model("blt_temporal")
