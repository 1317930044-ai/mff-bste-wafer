import csv
import hashlib
import json
import os
import platform
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "default.yaml"
ARTIFACTS = ROOT / "artifacts"
RUNS = ARTIFACTS / "runs"
CACHE = ARTIFACTS / "cache"
SAMPLES = ARTIFACTS / "samples"
ASSETS = ARTIFACTS / "reports"
CLASSES = ("Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full", "None")
BRANCHES = ("spatial", "geometry", "radon", "texture", "projection")
SOURCE_URL = "https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map"
SYNTHETIC = "合成测试数据 / SYNTHETIC TEST DATA"
FEATURE_VERSION = "1.0"
BASE = ("decision_tree", "random_forest", "extra_trees", "bagging", "hist_gradient", "ada_boost", "gradient_boost")
OPTIONAL = ("balanced_rf", "xgboost", "lightgbm", "catboost")
STRATEGIES = ("none", "class_weight", "oversample", "smote")
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def labels(task):
    return CLASSES if task == "all9" else CLASSES[:-1]


def load_config(path):
    return read_json(Path(path))  # JSON syntax is a valid subset of YAML 1.2.


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def run_path(name):
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError("运行 ID 只允许字母、数字、点、下划线和连字符，并以字母或数字开头。")
    return RUNS / name


def environment():
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "Pillow", "joblib", "threadpoolctl",
                "streamlit", "imbalanced-learn", "psutil", "xgboost", "lightgbm", "catboost")
    versions = {}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "未安装"
    cpu = os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or platform.machine()
    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return {"python": sys.version.split()[0], "platform": platform.platform(), "cpu": cpu,
            "logical_cpus": os.cpu_count(), "packages": versions}


class Progress:
    def __init__(self, path=None, quiet=False):
        self.path = path
        self.quiet = quiet
        self.start = time.perf_counter()

    def __call__(self, stage, message, done=None, total=None, **extra):
        elapsed = round(time.perf_counter() - self.start, 3)
        row = dict(time=utc_now(), elapsed_s=elapsed, stage=stage, message=message,
                   done=done, total=total, **extra)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if not self.quiet:
            count = f" {done}/{total}" if total else ""
            print(f"[{elapsed:8.2f}s] {stage}{count} | {message}", flush=True)
