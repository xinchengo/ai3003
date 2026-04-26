import copy
import json
from typing import Any, Dict


INHERIT_KEY = "inherit"
INHERITABLE_SECTIONS = ("preprocess", "model", "train")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == INHERIT_KEY:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_parents(parents: Any, section_name: str, name: str) -> list[str]:
    if isinstance(parents, str):
        return [parents]
    if isinstance(parents, list) and all(isinstance(parent, str) for parent in parents):
        return parents
    raise ValueError(
        f"{section_name}.{name}.{INHERIT_KEY} must be a string or a list of strings"
    )


def resolve_config_section(
    section: Dict[str, Dict[str, Any]],
    section_name: str,
) -> Dict[str, Dict[str, Any]]:
    resolved = {}
    resolving = []

    def resolve_entry(name: str) -> Dict[str, Any]:
        if name in resolved:
            return copy.deepcopy(resolved[name])
        if name not in section:
            raise ValueError(f"{section_name} config {name} not found")

        entry = section[name]
        if not isinstance(entry, dict):
            raise ValueError(f"{section_name} config {name} must be an object")
        if name in resolving:
            cycle = " -> ".join(resolving + [name])
            raise ValueError(f"Circular config inheritance detected: {cycle}")

        resolving.append(name)
        merged = {}
        if INHERIT_KEY in entry:
            for parent_name in _normalize_parents(entry[INHERIT_KEY], section_name, name):
                merged = _deep_merge(merged, resolve_entry(parent_name))
        merged = _deep_merge(merged, entry)
        resolving.pop()

        resolved[name] = merged
        return copy.deepcopy(merged)

    for name in section:
        resolve_entry(name)
    return resolved


def resolve_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved_config = copy.deepcopy(config)
    for section_name in INHERITABLE_SECTIONS:
        if section_name in resolved_config:
            resolved_config[section_name] = resolve_config_section(
                resolved_config[section_name],
                section_name,
            )
    return resolved_config


def load_config(json_file: str = "config.json") -> Dict[str, Any]:
    with open(json_file, "r") as f:
        return resolve_config_dict(json.load(f))
