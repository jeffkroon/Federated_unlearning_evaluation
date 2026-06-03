"""Registry for model factories (plug-and-play experiment types)."""

from typing import Dict, Callable, Any, Optional


class ModelRegistry:
    """Maps experiment_type to a factory callable(**kwargs) -> model."""

    _factories: Dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, experiment_type: str, factory: Callable[..., Any]) -> None:
        cls._factories[experiment_type] = factory

    @classmethod
    def get_factory(cls, experiment_type: str) -> Optional[Callable[..., Any]]:
        return cls._factories.get(experiment_type)

    @classmethod
    def create_model(cls, experiment_type: str, **kwargs) -> Any:
        factory = cls.get_factory(experiment_type)
        if factory is None:
            raise ValueError(
                f"Unknown experiment type: {experiment_type}. "
                f"Registered types: {list(cls._factories.keys())}. "
                "Register via ModelRegistry.register(experiment_type, factory)."
            )
        return factory(**kwargs)

    @classmethod
    def list_registered(cls) -> list:
        return list(cls._factories.keys())
