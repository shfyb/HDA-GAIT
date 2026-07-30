#!/usr/bin/env python3
"""Offline release checks: configuration, imports, secrets, and large files."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024
PRIVATE_USER = "su" + "hui"
FORBIDDEN_PATH_FRAGMENTS = (
    f"/home/{PRIVATE_USER}",
    f"/data_4/{PRIVATE_USER}",
    "migration_bundle",
    "CombinedDataset_Pretrain_merged",
)
SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile("aws_" + "secret_access_key", re.IGNORECASE),
    re.compile(r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY"),
)


def fail(message: str) -> None:
    raise AssertionError(message)


def check_files() -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_FILE_BYTES:
            fail(f"Unexpected file larger than 10 MiB: {relative}")
        if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".pkl", ".pickle"}:
            fail(f"Unexpected model/data artifact: {relative}")
        if path.suffix.lower() in {
            ".py", ".yaml", ".yml", ".json", ".md", ".txt", ".cff", ""
        }:
            # This checker necessarily contains the detection signatures.
            if relative == Path("tests/check_release.py"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for fragment in FORBIDDEN_PATH_FRAGMENTS:
                if fragment in text:
                    fail(f"Private path fragment {fragment!r} in {relative}")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    fail(f"Possible secret in {relative}: {pattern.pattern}")


def check_config() -> None:
    config_path = ROOT / "configs/gaitssb/pretrain_v4_6V100.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["model_cfg"]["model"] == "GaitSSB_Pretrain_MD_V4"
    assert config["model_cfg"]["backbone_cfg"]["type"] == "ResNet9_domain"
    assert config["trainer_cfg"]["sampler"]["batch_size"] == [18, 1]
    assert config["trainer_cfg"]["sampler"]["independent_views"] is True
    weights = {
        item["log_prefix"]: item["loss_term_weight"]
        for item in config["loss_cfg"]
    }
    assert weights == {
        "supcon_all": 1.0,
        "mid_mmd_loss": 0.1,
        "adv_domain_high": 0.05,
    }


def check_imports() -> None:
    sys.path.insert(0, str(ROOT / "opengait"))
    from modeling.backbones import ResNet9_domain  # noqa: F401
    from modeling.losses import (  # noqa: F401
        AdvLoss,
        MultiDomainMMDLoss,
        SupConLoss_Lp,
    )
    from modeling.models import GaitSSB_Pretrain_MD_V4  # noqa: F401


if __name__ == "__main__":
    check_files()
    check_config()
    check_imports()
    print("Release checks passed.")
