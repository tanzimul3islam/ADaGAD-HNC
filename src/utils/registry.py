from typing import Any


class Registry:
    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, type[Any]] = {}

    def register(self, name: str) -> Any:
        def decorator(cls: type[Any]) -> type[Any]:
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[Any]:
        if name not in self._registry:
            raise KeyError(f"{name} not found in {self._name} registry.")
        return self._registry[name]

    def list(self) -> list:
        return list(self._registry.keys())


ENCODER_REGISTRY = Registry("encoder")
READOUT_REGISTRY = Registry("readout")
SAMPLER_REGISTRY = Registry("sampler")
LOSS_REGISTRY = Registry("loss")
MODEL_REGISTRY = Registry("model")
