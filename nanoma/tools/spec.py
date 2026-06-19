"""Structured tool definitions — schemas are *generated*, not hand-written JSON.

Instead of maintaining nested OpenAI function-schema dicts by hand, declare a tool
once with typed argument specs:

    Tool(
        "create_file",
        "Create a new file with content.",
        tool_create_file,
        arg("path", str, "File path (relative to workspace)"),
        arg("content", str, "File content to write"),
        arg("overwrite", bool, "Overwrite an existing file", default=False),
    )

`Tool.schema` lazily produces the JSON the LLM sees. Arrays and nested objects are
declared structurally too:

    arg("replacements", list, "List of edits", items=obj(
        arg("old_string", str), arg("new_string", str),
    ))

`Tool` is also dict-compatible (`tool["handler"]`, `tool["schema"]`, `tool.get("is_meta")`)
so the runtime dispatch code needs no changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

_MISSING = object()

# Python type -> JSON Schema type
_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(t: Any) -> str:
    if isinstance(t, str):
        return t  # already a JSON type name
    if t in _JSON_TYPES:
        return _JSON_TYPES[t]
    raise ValueError(f"Unsupported arg type: {t!r}. Use str/int/float/bool/list/dict or a JSON type name.")


@dataclass
class Arg:
    """One tool parameter. `default` set ⇒ optional; else `required` controls it."""
    name: str
    type: Any = str
    description: str = ""
    default: Any = _MISSING
    enum: list | None = None
    items: Any = None          # for arrays: a python type, a JSON type name, or an obj()/dict schema
    required: bool = True

    @property
    def is_required(self) -> bool:
        return self.required and self.default is _MISSING

    def to_schema(self) -> dict[str, Any]:
        # Raw-schema escape hatch: arg(..., type={...}) passes a dict straight through.
        if isinstance(self.type, dict):
            prop = dict(self.type)
        else:
            prop = {"type": _json_type(self.type)}
        if self.description:
            prop["description"] = self.description
        if self.enum is not None:
            prop["enum"] = self.enum
        if prop.get("type") == "array":
            prop["items"] = _items_schema(self.items)
        if self.default is not _MISSING:
            prop["default"] = self.default
        return prop


def _items_schema(items: Any) -> dict[str, Any]:
    if items is None:
        return {"type": "string"}            # sensible default for string arrays
    if isinstance(items, dict):
        return items                          # an obj(...) result or raw schema
    return {"type": _json_type(items)}        # a python type / JSON type name


def arg(name: str, type: Any = str, description: str = "", *,
        default: Any = _MISSING, enum: list | None = None,
        items: Any = None, required: bool = True) -> Arg:
    """Declare a tool parameter. See `Arg`."""
    return Arg(name=name, type=type, description=description,
               default=default, enum=enum, items=items, required=required)


def obj(*args: Arg) -> dict[str, Any]:
    """Build an object JSON-schema from `arg(...)` specs (for nested array items)."""
    props = {a.name: a.to_schema() for a in args}
    required = [a.name for a in args if a.is_required]
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def build_function_schema(name: str, description: str, args: list[Arg]) -> dict[str, Any]:
    """Assemble an OpenAI-style function schema from arg specs."""
    props = {a.name: a.to_schema() for a in args}
    parameters: dict[str, Any] = {"type": "object", "properties": props}
    required = [a.name for a in args if a.is_required]
    if required:
        parameters["required"] = required
    return {"type": "function", "function": {
        "name": name, "description": description, "parameters": parameters,
    }}


@dataclass
class Tool:
    """A tool: name, description, handler, and structured argument specs.

    Dict-compatible for the runtime: ``tool["handler"]``, ``tool["schema"]``,
    ``tool.get("is_meta")``.
    """
    name: str
    description: str
    handler: Callable[..., Any]
    args: list[Arg] = field(default_factory=list)
    is_meta: bool = False

    @property
    def schema(self) -> dict[str, Any]:
        return build_function_schema(self.name, self.description, self.args)

    # ── dict compatibility ──────────────────────────────────────────────
    def __getitem__(self, key: str) -> Any:
        return {
            "name": self.name,
            "handler": self.handler,
            "schema": self.schema,
            "is_meta": self.is_meta,
        }[key]

    def __contains__(self, key: str) -> bool:
        return key in ("name", "handler", "schema", "is_meta")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def registry(*tools: Tool) -> dict[str, Tool]:
    """Build a {name: Tool} registry, rejecting duplicate names."""
    out: dict[str, Tool] = {}
    for t in tools:
        if t.name in out:
            raise ValueError(f"Duplicate tool name: {t.name}")
        out[t.name] = t
    return out
