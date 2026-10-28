"""VLMOD 数据集加载。

训练集/测试集 JSON 结构完全不同：
- 训练集（MonoMulti3D/train）：顶层为 list，[0] 是 3 条表达式记录（含 label_3 匹配GT），
  [1] 是相机标定 calib/denorm。
- 测试集（MonoMulti3D/test）：顶层为 dict，{public_description: 3条文本, test_data: N个候选}。

本模块职责：
1. 扫描并配对图片与标注（缺标注的图片会被跳过）。
2. 解析训练集/测试集为结构化样本。
3. 训练样本构建候选目标池（聚合该图 3 条表达式涉及的全部目标），用于生成正负样本对。

数据根目录默认为 D:/VLMOD，可用环境变量 VLMOD_DATA_ROOT 覆盖。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List

from code.datasets.rope3d import Rope3DObject, parse_line, parse_repr


@dataclass
class Expression:
    """一条训练表达式（一个查询文本 + 其匹配目标GT）。"""
    ann_id: int
    image_file_name: str
    public_properties: List[str]
    description: str
    matched_objects: List[Rope3DObject] = field(default_factory=list)


@dataclass
class TrainSample:
    """一个训练样本 = 一张图 + 3 条表达式 + 候选目标池 + 相机标定。"""
    image_path: str
    expressions: List[Expression]
    candidate_pool: List[Rope3DObject]
    calib: dict


@dataclass
class TestSample:
    """一个测试样本 = 一张图 + 3 条文本 + N 个候选目标。"""
    image_path: str
    descriptions: List[str]
    candidates: List[Rope3DObject]


def default_data_root() -> str:
    """数据根目录：环境变量优先，否则默认 D:/VLMOD。"""
    return os.environ.get("VLMOD_DATA_ROOT", r"D:/VLMOD")


def _resolve(ds_root: str, name: str) -> str:
    p = os.path.join(ds_root, name)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"找不到数据子目录: {p}")
    return p


def find_annotation_files(ann_dir: str):
    """扫描目录下所有 json，按文件名基名索引。"""
    files = {}
    for f in os.listdir(ann_dir):
        if f.lower().endswith(".json"):
            files[os.path.splitext(f)[0]] = os.path.join(ann_dir, f)
    return files


def load_train_sample(json_path: str, img_dir: str) -> TrainSample | None:
    """加载一个训练样本。

    若该图片缺标注（json 不存在）返回 None，由上层跳过。
    """
    stem = os.path.splitext(os.path.basename(json_path))[0]
    img_candidates = [
        os.path.join(img_dir, stem + ext)
        for ext in (".jpg", ".jpeg", ".png")
    ]
    image_path = next((p for p in img_candidates if os.path.isfile(p)), None)
    if image_path is None:
        # 有标注但缺图，也跳过
        return None

    with open(json_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    # 训练集顶层是 list: [表达式记录, [标定]]
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError(f"训练集 JSON 结构异常（非 [expressions, calib]）: {json_path}")

    recs, meta = raw[0], raw[1]
    expressions: List[Expression] = []
    for rec in recs:
        matched = [parse_repr(s) for s in rec.get("label_3", [])]
        expressions.append(Expression(
            ann_id=int(rec.get("ann_id", 0)),
            image_file_name=rec.get("image_file_name", ""),
            public_properties=list(rec.get("public_properties", [])),
            description=rec.get("public_description", ""),
            matched_objects=matched,
        ))

    calib = {}
    if meta and isinstance(meta[0], dict):
        calib = {k: meta[0].get(k) for k in ("calib", "denorm")}

    # 候选池：聚合本图 3 条表达式涉及的全部目标，去重
    pool = {}
    for e in expressions:
        for obj in e.matched_objects:
            pool[obj.to_tuple()] = obj
    candidate_pool = list(pool.values())

    return TrainSample(
        image_path=image_path,
        expressions=expressions,
        candidate_pool=candidate_pool,
        calib=calib,
    )


def load_test_sample(json_path: str, img_dir: str) -> TestSample | None:
    """加载一个测试样本（评估格式，无标签）。"""
    stem = os.path.splitext(os.path.basename(json_path))[0]
    img_candidates = [
        os.path.join(img_dir, stem + ext)
        for ext in (".jpg", ".jpeg", ".png")
    ]
    image_path = next((p for p in img_candidates if os.path.isfile(p)), None)
    if image_path is None:
        return None

    with open(json_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"测试集 JSON 结构异常（应为 dict）: {json_path}")

    descriptions = list(raw.get("public_description", []))
    candidates = [parse_line(line) for line in raw.get("test_data", [])]
    return TestSample(
        image_path=image_path,
        descriptions=descriptions,
        candidates=candidates,
    )


def iter_train(ds_root: str | None = None):
    """遍历训练集，产出 TrainSample（跳过缺标注/缺图的样本）。"""
    ds_root = ds_root or default_data_root()
    img_dir = _resolve(ds_root, "img/train")
    ann_dir = _resolve(ds_root, "MonoMulti3D/train")
    anns = find_annotation_files(ann_dir)
    for stem, ann_path in anns.items():
        sample = load_train_sample(ann_path, img_dir)
        if sample is not None:
            yield sample


def iter_test(ds_root: str | None = None):
    """遍历测试集，产出 TestSample（跳过缺图的样本）。"""
    ds_root = ds_root or default_data_root()
    img_dir = _resolve(ds_root, "img/test")
    ann_dir = _resolve(ds_root, "MonoMulti3D/test")
    anns = find_annotation_files(ann_dir)
    for stem, ann_path in anns.items():
        sample = load_test_sample(ann_path, img_dir)
        if sample is not None:
            yield sample
