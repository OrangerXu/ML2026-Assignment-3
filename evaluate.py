#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assignment 3 本地评测脚本（学生自评用）

用法（在仓库根目录，与 solution.py / model.py 同级）：

    python evaluate.py                          # 在 test_public.csv 上评测
    python evaluate.py --student-id 2026001 --name 张三
    python evaluate.py --rows 5000              # 快速试跑前 5000 行

流程：禁用库扫描（AST）-> 在隔离的子进程中：
  计时调用 Solution.fit(train) 训练 -> 10 线程计时逐样本 forward ->
  计算 ROC-AUC / Accuracy / Latency = sqrt(训练时间 × 测试时间) ->
打印并写入 result.json。

说明：你的代码运行在独立子进程中，只会收到不含 label 与 id 的数据；
测试数据与真值始终保留在评测器主进程内。探测评测器进程内存、
命名空间或文件系统获取答案，均按作弊处理。

提交作业时：把 solution.py、model.py 和 result.json 一起上交
（建议附 analysis.py 与 report.md，不计入评测）。
自评分数仅供参考，最终成绩以教师在保留数据上的复测为准。
"""
import argparse
import ast
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

EVALUATOR_VERSION = "2026.3.0"
ASSIGNMENT = "ML2026-Assignment-3"
DEFAULT_TEST = "test_public.csv"
TRAIN_FILE = "train.csv"
CODE_FILES = ["solution.py", "model.py"]
AUC_BAR = 0.98

FORBIDDEN_LIBS = [
    "sklearn", "scikit-learn", "tensorflow", "torch", "pytorch",
    "keras", "xgboost", "lightgbm", "catboost", "statsmodels",
    "libsvm", "liblinear", "thundersvm", "cuml", "shotgun",
]


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_forbidden():
    hits = []
    for name in CODE_FILES:
        tree = ast.parse(Path(name).read_text(encoding="utf-8-sig"), filename=name)
        for node in ast.walk(tree):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                      [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for module in modules:
                if module.split(".")[0] in FORBIDDEN_LIBS:
                    hits.append("{}:{}: {}".format(name, node.lineno, module))
    return hits


def auc_score(y_true, scores):
    """纯 numpy 的 ROC-AUC（Mann-Whitney，并列分数取平均秩）。"""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC 需要两类样本同时存在")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros(len(uniq))
    np.add.at(sum_ranks, inv, ranks)
    avg_ranks = (sum_ranks / counts)[inv]
    return float((avg_ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ---------------- 子进程：训练计时 + 并发推理计时 ----------------

def _run_solution(train_path, samples):
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        import solution
    except Exception as e:
        return None, "导入 solution.py 失败: {}".format(e)
    try:
        sol = solution.Solution()
    except Exception as e:
        return None, "初始化 Solution 失败: {}".format(e)

    df = pd.read_csv(train_path)
    y_train = df["label"].to_numpy()
    X_train = df.drop(columns=["label", "id"], errors="ignore")

    t0 = time.perf_counter()
    try:
        sol.fit(X_train, y_train)
    except Exception as e:
        return None, "fit 失败: {}".format(e)
    training_time = time.perf_counter() - t0

    preds, probs = [None] * len(samples), [None] * len(samples)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(sol.forward, s): i for i, s in enumerate(samples)}
        try:
            for fut in futures:
                r = fut.result()
                i = futures[fut]
                preds[i] = int(r["prediction"])
                probs[i] = float(r["probability"])
        except Exception as e:
            return None, "forward 在第 {} 个样本上失败: {}".format(futures[fut], e)
    testing_time = time.perf_counter() - t0

    p = np.asarray(probs)
    if not np.all(np.isfinite(p)):
        return None, "probability 包含 NaN/Inf"
    return {"predictions": preds, "probabilities": probs,
            "training_time_s": round(training_time, 3),
            "prediction_time_s": round(testing_time, 3)}, None


def _worker_main(workdir):
    samples = []
    with open(os.path.join(workdir, "samples.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    out, error = _run_solution(os.path.join(os.getcwd(), TRAIN_FILE), samples)
    with open(os.path.join(workdir, "worker_out.json"), "w", encoding="utf-8") as f:
        json.dump({"result": out, "error": error}, f, ensure_ascii=False)


def _run_worker(samples, timeout):
    workdir = tempfile.mkdtemp(prefix="ml2026_eval_")
    try:
        with open(os.path.join(workdir, "samples.jsonl"), "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        cmd = [sys.executable, os.path.abspath(__file__), "--_worker", workdir]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="ignore", timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError("评测超时（>{}s），请检查训练/推理是否存在死循环".format(timeout))
        out_file = os.path.join(workdir, "worker_out.json")
        if not os.path.exists(out_file):
            tail = ((proc.stdout or "") + (proc.stderr or ""))[-2000:]
            raise RuntimeError("评测子进程异常退出（code={}）：\n{}".format(
                proc.returncode, tail))
        with open(out_file, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("error"):
            raise RuntimeError(data["error"])
        return data["result"], (proc.stdout or "").strip()[-1500:]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Assignment 3 本地评测")
    ap.add_argument("--test", default=DEFAULT_TEST, help="测试数据文件")
    ap.add_argument("--rows", type=int, default=0, help="只评测前 N 行（0=全部）")
    ap.add_argument("--out", default="result.json", help="结果输出文件")
    ap.add_argument("--student-id", default="", help="学号（可选）")
    ap.add_argument("--name", default="", help="姓名（可选）")
    ap.add_argument("--timeout", type=int, default=3600, help="评测超时秒数")
    ap.add_argument("--_worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker:
        _worker_main(args._worker)
        return

    for name in CODE_FILES + [args.test, TRAIN_FILE]:
        if not Path(name).exists():
            sys.exit("缺少文件: {}（请在仓库根目录运行）".format(name))

    hits = scan_forbidden()
    if hits:
        sys.exit("[禁止] 检测到禁用库引用:\n  " + "\n  ".join(hits))

    df = pd.read_csv(args.test)
    if args.rows:
        df = df.head(args.rows)
    y_true = df["label"].to_numpy(dtype=int)
    samples = df.drop(columns=["label", "id"], errors="ignore").to_dict("records")

    print("Evaluating {} rows from {} (train: {}) ...".format(
        len(df), args.test, TRAIN_FILE))
    res, stdout_tail = _run_worker(samples, args.timeout)
    if stdout_tail:
        print("---- solution 输出 ----\n{}\n----------------------".format(stdout_tail[-800:]))

    probs = np.asarray(res["probabilities"], dtype=float)
    preds = np.asarray(res["predictions"], dtype=int)
    auc = auc_score(y_true, probs)
    acc = float((preds == y_true).mean())
    latency = float(np.sqrt(max(res["training_time_s"], 1e-9) *
                            max(res["prediction_time_s"], 1e-9)))

    print("=" * 50)
    print("ROC-AUC:        {:.6f}{}".format(auc, "  [达标 >= 0.98]" if auc >= AUC_BAR else "  [未达 0.98!]"))
    print("Accuracy:       {:.6f}".format(acc))
    print("Training Time:  {:.3f}s".format(res["training_time_s"]))
    print("Testing Time:   {:.3f}s".format(res["prediction_time_s"]))
    print("Latency:        {:.3f}s".format(latency))
    print("=" * 50)

    result = {
        "assignment": ASSIGNMENT,
        "evaluator_version": EVALUATOR_VERSION,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "student": {"id": args.student_id, "name": args.name},
        "test_file": {"name": Path(args.test).name, "md5": md5_file(args.test),
                      "rows": int(len(df))},
        "metrics": {"auc": round(auc, 6), "accuracy": round(acc, 6)},
        "training_time_s": res["training_time_s"],
        "prediction_time_s": res["prediction_time_s"],
        "latency_s": round(latency, 3),
        "code_md5": {name: md5_file(name) for name in CODE_FILES},
        "code_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                        for name in CODE_FILES},
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("结果已写入 {}（提交作业时与代码一起上交）".format(args.out))


if __name__ == "__main__":
    main()
