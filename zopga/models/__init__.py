"""Model registry."""

from .alexnet import alexnet, alexnet_half
from .lenet import lenet5, lenet5_half
from .resnet import resnet18, resnet34

_REGISTRY = {
    "alexnet": alexnet,
    "alexnet_half": alexnet_half,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "lenet5": lenet5,
    "lenet5_half": lenet5_half,
}


def get_model(name, num_classes=10):
    """Build a model by registry name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](num_classes=num_classes)


def available_models():
    return sorted(_REGISTRY)
