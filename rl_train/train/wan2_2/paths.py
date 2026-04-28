"""Wan2.2 RL 用到的仓库内路径（可移植；可用环境变量覆盖）。"""

from __future__ import annotations

import os
from pathlib import Path

_WAN22_DIR = Path(__file__).resolve().parent


def repo_root() -> Path:
    """rl_train_new 仓库根：wan2_2 -> train -> rl_train -> 根。"""
    return _WAN22_DIR.parent.parent.parent


def videox_fun_root() -> Path:
    """VideoX-Fun 根目录（内含 `videox_fun` 包）。默认：仓库内 bundled 副本。

    覆盖：设置环境变量 ``VIDEOX_FUN_ROOT`` 指向其它 VideoX-Fun 克隆。
    """
    override = os.environ.get("VIDEOX_FUN_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "eval" / "infer" / "wan2.2" / "VideoX-Fun"


def default_wan22_config_path() -> str:
    """默认 OmegaConf yaml（相对 bundled VideoX-Fun）。

    覆盖：环境变量 ``WAN22_CONFIG_PATH``。
    """
    override = os.environ.get("WAN22_CONFIG_PATH", "").strip()
    if override:
        return str(Path(override).expanduser().resolve())
    cfg = videox_fun_root() / "config" / "wan2.2" / "wan_civitai_5b.yaml"
    return str(cfg.resolve())
