import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class AoiConfig:
    name: str
    description: str
    bbox: dict
    crs: str
    resolution_m: int
    sentinel1: dict
    dem: dict


@dataclass
class PipelineConfig:
    version: str
    aoi_config: str
    paths: dict
    surface_response_classes: dict
    construction_risk: dict


def _resolve_env_vars(obj):
    if isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(i) for i in obj]
    return obj


def load_aoi_config(path: Path) -> AoiConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AoiConfig(
        name=raw["name"],
        description=raw.get("description", ""),
        bbox=raw["bbox"],
        crs=raw["crs"],
        resolution_m=raw["resolution_m"],
        sentinel1=raw["sentinel1"],
        dem=raw["dem"],
    )


def load_pipeline_config(path: Path) -> PipelineConfig:
    with open(path) as f:
        raw = _resolve_env_vars(yaml.safe_load(f))
    return PipelineConfig(
        version=raw["version"],
        aoi_config=raw["aoi_config"],
        paths=raw["paths"],
        surface_response_classes=raw["surface_response_classes"],
        construction_risk=raw["construction_risk"],
    )


def load_config_dict(path: Path) -> dict:
    """Load any YAML config as a plain dict with env-var resolution."""
    with open(path) as f:
        return _resolve_env_vars(yaml.safe_load(f))
