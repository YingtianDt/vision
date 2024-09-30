from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.temporal.utils import get_specified_layers
from brainscore_vision.model_interface import BrainModel
from . import model


def commit_model(identifier, fps=2):
    activations_model=model.get_model(identifier, fps)
    layers=get_specified_layers(activations_model)
    return ModelCommitment(identifier=activations_model.identifier, activations_model=activations_model, layers=layers)


model_registry["VideoChat-7B"] = lambda: commit_model("VideoChat-7B")
model_registry["VideoChat-13B"] = lambda: commit_model("VideoChat-13B")
model_registry["VideoChat2-7B"] = lambda: commit_model("VideoChat2-7B")


model_registry["VideoChat-7B-fps16"] = lambda: commit_model("VideoChat-7B", 16)
model_registry["VideoChat-13B-fps16"] = lambda: commit_model("VideoChat-13B", 16)
model_registry["VideoChat2-7B-fps16"] = lambda: commit_model("VideoChat2-7B", 16)
