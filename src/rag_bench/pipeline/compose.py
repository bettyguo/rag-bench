"""YAML pipeline composer.

Loads a pipeline YAML and instantiates the Pipeline. Dispatches each stage
on the `type:` field via the component registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rag_bench.pipeline.base import (
    Chunker,
    Component,
    ComponentConfig,
    Generator,
    Reranker,
    Retriever,
    Stage,
    get_component_cls,
)

# Importing the component modules forces their @register calls. Without this
# the registry stays empty when compose.py is imported standalone.
from rag_bench.pipeline.components import chunkers as _c  # noqa: F401
from rag_bench.pipeline.components import generators as _g  # noqa: F401
from rag_bench.pipeline.components import rerankers as _rr  # noqa: F401
from rag_bench.pipeline.components import retrievers as _r  # noqa: F401
from rag_bench.pipeline.pipeline import Pipeline


class PipelineSpec(BaseModel):
    """Top-level pipeline YAML schema."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Human-readable name; appears on the leaderboard.")
    chunker: dict[str, Any]
    retriever: dict[str, Any]
    reranker: dict[str, Any] = Field(default_factory=lambda: {"type": "identity", "top_k": 10})
    generator: dict[str, Any]
    retriever_top_k: int = Field(50, ge=1, description="Top-k from the retriever stage.")
    reranker_top_k: int = Field(10, ge=1, description="Top-k from the reranker stage; what the generator sees.")


def _instantiate(stage: Stage, raw: dict[str, Any]) -> Component:
    type_name = raw.get("type")
    if not type_name:
        raise ValueError(f"{stage} config missing required 'type' field: {raw!r}")
    cls = get_component_cls(stage, type_name)
    cfg_cls = _config_class_for(cls)
    cfg = cfg_cls(**raw)
    return cls(cfg)


def _config_class_for(component_cls: type[Component]) -> type[ComponentConfig]:
    """Discover the ComponentConfig subclass attached to a component.

    Convention: a component's __init__ first parameter is annotated with its config class.
    Resolves string annotations via typing.get_type_hints (so `from __future__ import
    annotations` modules work).
    """
    import inspect
    import typing

    sig = inspect.signature(component_cls.__init__)
    params = list(sig.parameters.values())
    if len(params) < 2:
        raise TypeError(f"{component_cls.__name__}.__init__ must accept a config argument")
    cfg_name = params[1].name
    hints = typing.get_type_hints(component_cls.__init__)
    cfg_type = hints.get(cfg_name)
    if cfg_type is None or not isinstance(cfg_type, type):
        raise TypeError(
            f"{component_cls.__name__}.__init__ config parameter must have a class annotation"
        )
    if not issubclass(cfg_type, ComponentConfig):
        raise TypeError(
            f"{component_cls.__name__} config annotation must subclass ComponentConfig"
        )
    return cfg_type


def compose_from_dict(spec: dict[str, Any]) -> Pipeline:
    """Build a Pipeline from a Python dict (typically yaml.safe_load output)."""
    if "pipeline" in spec:
        spec = spec["pipeline"]
    parsed = PipelineSpec(**spec)
    chunker = _instantiate("chunker", parsed.chunker)
    retriever = _instantiate("retriever", parsed.retriever)
    reranker = _instantiate("reranker", parsed.reranker)
    generator = _instantiate("generator", parsed.generator)
    assert isinstance(chunker, Chunker)
    assert isinstance(retriever, Retriever)
    assert isinstance(reranker, Reranker)
    assert isinstance(generator, Generator)
    return Pipeline(
        name=parsed.name,
        chunker=chunker,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        retriever_top_k=parsed.retriever_top_k,
        reranker_top_k=parsed.reranker_top_k,
    )


def compose_from_yaml(yaml_text: str) -> Pipeline:
    """Build a Pipeline from a YAML string."""
    data = yaml.safe_load(yaml_text)
    if data is None:
        raise ValueError("Pipeline YAML is empty.")
    return compose_from_dict(data)


def compose_from_path(path: str | Path) -> Pipeline:
    """Build a Pipeline from a YAML file path."""
    text = Path(path).read_text(encoding="utf-8")
    return compose_from_yaml(text)
