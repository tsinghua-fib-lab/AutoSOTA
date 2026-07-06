import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


def _slugify(model_name: str) -> str:
    return model_name.replace("/", "-").replace(":", "-")


@dataclass
class ModelSpec:
    name: str
    roles: List[str]
    temperature: float
    reasoning: Optional[str] = None
    display_name: Optional[str] = None

    @property
    def slug(self) -> str:
        return _slugify(self.name)

    @property
    def pretty(self) -> str:
        return self.display_name or self.name


class ModelRegistry:
    def __init__(self, config_path: Path):
        with config_path.open("r") as f:
            raw = json.load(f)
        self.default_temperature = raw.get("default_temperature", 0.2)
        self.default_reasoning = raw.get("default_reasoning")
        self.models: Dict[str, ModelSpec] = {}
        self.slug_index: Dict[str, ModelSpec] = {}
        seen_slugs: Dict[str, str] = {}
        for entry in raw.get("models", []):
            spec = ModelSpec(
                name=entry["name"],
                roles=entry.get("roles", []),
                temperature=entry.get("temperature", self.default_temperature),
                reasoning=entry.get("reasoning", self.default_reasoning),
                display_name=entry.get("display_name"),
            )
            # Check for slug uniqueness
            if spec.slug in seen_slugs:
                raise ValueError(f"Duplicate slug '{spec.slug}' for models '{spec.name}' and '{seen_slugs[spec.slug]}'")
            seen_slugs[spec.slug] = spec.name
            self.models[spec.name] = spec
            self.slug_index[spec.slug] = spec

    def by_role(self, role: str) -> List[ModelSpec]:
        results = [spec for spec in self.models.values() if role in spec.roles]
        if not results:
            logger.warning(f"No models found with role '{role}'")
        return results

    def pick(self, names: Optional[Iterable[str]]) -> List[ModelSpec]:
        if not names:
            return list(self.models.values())
        selected = []
        for name in names:
            if name not in self.models:
                raise ValueError(f"Model '{name}' not found in registry")
            selected.append(self.models[name])
        return selected

    def display_name_for_slug(self, slug: str) -> str:
        spec = self.slug_index.get(slug)
        if spec:
            return spec.pretty
        return slug

    def display_name_for_name(self, name: str) -> str:
        spec = self.models.get(name)
        if spec:
            return spec.pretty
        return name

    def resolve_model_name(self, name_or_slug: str) -> str:
        """
        Resolve a model name or slug to the canonical model name.
        Returns the input if not found in registry.
        """
        if not name_or_slug:
            return name_or_slug
        if name_or_slug in self.models:
            return name_or_slug
        spec = self.slug_index.get(name_or_slug)
        if spec:
            return spec.name
        return name_or_slug

    def candidate_model_names(self, name_or_slug: str) -> List[str]:
        """
        Get all possible name variations for a model (name, slug, display_name).
        Used for matching models in different contexts.
        """
        if not name_or_slug:
            return []
        names = {name_or_slug}
        if name_or_slug in self.models:
            spec = self.models[name_or_slug]
            names.update({spec.slug, spec.pretty})
        if name_or_slug in self.slug_index:
            spec = self.slug_index[name_or_slug]
            names.update({spec.name, spec.pretty})
        return [n for n in names if n]


def load_registry(path: str = "configs/models.json") -> ModelRegistry:
    return ModelRegistry(Path(path))


__all__ = ["ModelRegistry", "ModelSpec", "load_registry", "_slugify"]
