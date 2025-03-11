from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import MotionEnergyModel

model_registry['motion-energy'] = lambda: ModelCommitment(
    identifier='motion-energy',
    activations_model=MotionEnergyModel(),
    layers=['motion-energy'])
