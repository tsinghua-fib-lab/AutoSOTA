from typing import Any, Callable, Dict, Type, TypeVar

T = TypeVar("T")

REGISTRIES = {}


def setup(registry_name: str):
	"""Setup a registry.

	Args:
		registry_name (str): Registry name for grouping classes.

	Returns:
		Tuple of the two functions:
		  - register: Register a class as the given name.
		  - get_cls: Return the registered class of the given name.
	"""
	REGISTRY = {}
	REGISTRIES[registry_name] = REGISTRY

	def register(name: str) -> Callable[[Type[T]], Type[T]]:
		"""Register a class as the given name.

		Args:
			name (str): The name of a class.
		"""

		def _register(cls: Type[T]):
			if name in REGISTRY:
				raise ValueError(
					f"{name} already registered as {REGISTRY[name].__name__}. ({cls.__name__})"
				)
			REGISTRY[name] = cls
			return cls

		return _register

	def get_cls(name: str):
		if name not in REGISTRY:
			raise NotImplementedError(
				f"`{name}` is not registered in `{registry_name}`."
			)
		return REGISTRY[name]

	return register, get_cls


def get_registry(registry_name: str) -> Dict[str, Type[Any]]:
	"""Get registry of the given name.

	Args:
		registry_name (str): Registry name to be returned.

	Returns:
		Dict[str, Type[Any]]: Class mapper from registered name to its corresponding class.
	"""
	return REGISTRIES[registry_name]
