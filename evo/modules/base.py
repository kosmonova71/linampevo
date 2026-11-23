from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModuleUniform:
    name: str
    value: float
    kind: str = "float"

    @property
    def glsl_decl(self) -> str:
        gl_kind = "int" if self.kind == "int" else "float"
        return f"uniform {gl_kind} {self.name};"


@dataclass
class ModuleConfig:
    category: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    raymarch_steps: int | None = None
    fractal_layers: int | None = None


@dataclass
class ModuleSections:
    header: str = ""
    footer: str = ""
    call: str = ""
    uniforms: list[ModuleUniform] = field(default_factory=list)

    def merge(self, other: ModuleSections) -> None:
        if other.header:
            self.header += "\n" + other.header if self.header else other.header
        if other.footer:
            self.footer += "\n" + other.footer if self.footer else other.footer
        if other.call:
            self.call += "\n" + other.call if self.call else other.call
        self.uniforms.extend(other.uniforms)


class ShaderModule(ABC):
    config: ModuleConfig
    sections: ModuleSections

    def __init__(self, config: ModuleConfig) -> None:
        self.config = config

    @abstractmethod
    def sections(self) -> ModuleSections: ...

    _TWO_PI = 6.28318

    @staticmethod
    def _glsl_float(value: float, lo: float | None = None, hi: float | None = None) -> str:
        v = float(value)
        if not __import__("math").isfinite(v):
            v = 0.0
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        s = repr(v)
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s

    def _uniform(
        self,
        key: str,
        default: float,
        kind: str = "float",
        uniforms: list[ModuleUniform] | None = None,
    ) -> ModuleUniform:
        value = float(self.config.params.get(key, default))
        name = f"u_{self.config.category}_{key}"
        u = ModuleUniform(name=name, value=value, kind=kind)
        if uniforms is not None:
            uniforms.append(u)
        return u

    def _define(self, name: str, value: Any) -> str:
        return f"#define {name} {value}\n"

    def _call_snippet(self, fmt: str, **kwargs) -> str:
        return fmt.format(prefix=self.config.name, **kwargs)

    def mutate(self, strength: float = 0.1) -> None:
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.config.name}>"
