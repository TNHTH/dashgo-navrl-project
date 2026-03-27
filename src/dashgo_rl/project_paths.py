from __future__ import annotations

from pathlib import Path


def resolve_project_root(start: str | Path | None = None) -> Path:
    """从任意仓库内路径回溯到当前实验仓库根目录。"""
    path = Path(start).resolve() if start is not None else Path(__file__).resolve()
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists() and (candidate / "README.md").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = resolve_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
APPS_ROOT = PROJECT_ROOT / "apps"
CONFIGS_ROOT = PROJECT_ROOT / "configs"
TOOLS_ROOT = PROJECT_ROOT / "tools"
DOCS_ROOT = PROJECT_ROOT / "docs"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
LOGS_ROOT = ARTIFACTS_ROOT / "logs"
CHECKPOINTS_ROOT = ARTIFACTS_ROOT / "checkpoints"
EVAL_ROOT = ARTIFACTS_ROOT / "eval"
TMP_ROOT = PROJECT_ROOT / ".tmp"

DASHGO_URDF_PATH = CONFIGS_ROOT / "robot" / "dashgo.urdf"
EAI_PARAMS_YAML = CONFIGS_ROOT / "robot" / "dashgo_driver.yaml"
TRAIN_CONFIG_ROOT = CONFIGS_ROOT / "train"


def ensure_project_sys_path() -> Path:
    """供脚本入口在运行前注入项目与 src 路径。"""
    import sys

    for candidate in (PROJECT_ROOT, SRC_ROOT):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    return PROJECT_ROOT
