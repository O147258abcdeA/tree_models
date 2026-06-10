"""配置加载与版本管理。

- load_config: 读取 yaml
- merge_config: 深度合并（override 优先）
- config_hash: 配置内容 hash，用于 feature_version / label_version 绑定
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_config(path: str | Path) -> dict:
    """读取 yaml 配置文件。path 可以是绝对路径或 config/ 下的文件名。"""
    p = Path(path)
    if not p.exists():
        p = CONFIG_DIR / path
    if not p.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def merge_config(base: dict, override: dict | None) -> dict:
    """深度合并两个配置 dict，override 中的键优先。"""
    if override is None:
        return dict(base)
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = v
    return out


def config_hash(cfg: dict, length: int = 8) -> str:
    """配置内容的稳定 hash，用于自动版本号。配置变更 -> hash 变更 -> 新版本。"""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:length]


def get_version(cfg: dict, key: str, prefix: str) -> str:
    """读取配置中的显式版本号；若缺失则用配置 hash 自动生成（禁止无版本入库）。"""
    v: Any = cfg.get(key)
    if v:
        return str(v)
    return f"{prefix}{config_hash(cfg)}"
