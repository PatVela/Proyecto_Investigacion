"""Empaqueta lo mínimo para desplegar la webapp sin ``saved/`` ni ``runs/``.

Genera, a partir de dos checkpoints y de la evaluación del ensemble:

    models/<nombre>.pt           pesos de inferencia (≈19 MB c/u, sin estado del optimizador)
    deploy/manifest.json         huellas SHA-256, época, val_loss, alpha, versiones
    deploy/ensemble.env          variables de entorno que consume la webapp
    deploy/temperatures_{a,b}.csv        calibración por temperatura (bundles anteriores o --temperatures-*)
    deploy/platt_{a,b}.csv, platt_ensemble.csv   calibración afín por modelo y del promedio (--platt-*)
    deploy/results/<eval>/       métricas, curvas ROC/PR y umbrales (sin predicciones por registro)
    deploy/reports/              comparación de modelos y robustez

Las rutas absolutas de la máquina de entrenamiento se eliminan de los JSON
publicados (la webapp los expone por ``/metrics``).

Uso (desde la raíz del repositorio)::

    python tools/export_inference_bundle.py \\
      --checkpoint-a saved/cinc2020/cinc2020_resnet/<run>/best.pt \\
      --checkpoint-b saved/cinc2020/cinc2020_resnet_v2/<run>/best.pt \\
      --eval-dir runs/eval/ensemble-f1-platt \\
      --platt-a runs/eval/eval-resnet/platt_validation.csv \\
      --platt-b runs/eval/eval-resnet-v2/platt_validation.csv \\
      --platt-ensemble runs/eval/ensemble-f1-platt/platt_ensemble_validation.csv \\
      --reports-dir runs/reports

Cada modelo necesita una calibración: ``--temperatures-*`` o ``--platt-*`` (si llegan las dos, la webapp usa la afín).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ecg import load, network, util  # noqa: E402
from ecg.calibration import load_platt_csv, load_temperatures_csv  # noqa: E402

REQUIRED_RESULT_FILES = ("metrics_global.csv", "metrics_per_class.csv", "evaluation_summary.json",
                         "thresholds_validation.csv")
OPTIONAL_RESULT_FILES = ("roc_curves_test.csv", "roc_curves_validation.csv",
                         "pr_curves_test.csv", "pr_curves_validation.csv")
REPORT_FILES = ("model_comparison.csv", "robustness.csv")
PATH_KEYS = ("checkpoint", "checkpoint_a", "checkpoint_b", "temperatures", "temperatures_a", "temperatures_b",
             "platt", "platt_a", "platt_b", "platt_ensemble")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _base_name(value: str) -> str:
    """Nombre de archivo de una ruta de Windows o POSIX (sin directorios de la máquina origen)."""
    return PureWindowsPath(value).name if "\\" in value or ":" in value else PurePosixPath(value).name


def export_model(src: str | Path, dst: str | Path) -> dict:
    """Escribe un checkpoint solo con lo necesario para inferir y devuelve sus metadatos."""
    ckpt = util.load_checkpoint(src)
    schema = ckpt.get("label_schema")
    if not schema:
        raise ValueError(f"{src} no declara label_schema; no se puede empaquetar un modelo ambiguo.")
    bundle = {
        "model_state_dict": {k: v.detach().cpu().contiguous() for k, v in ckpt["model_state_dict"].items()},
        "class_names": list(ckpt["class_names"]),
        "label_schema": schema,
        "config": network.params_from_config(ckpt.get("config", {})),
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
        "problem_type": ckpt.get("problem_type", "multilabel_sigmoid_bce"),
    }
    import torch

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, dst)
    return {"file": dst.name, "bytes": dst.stat().st_size, "sha256": _sha256(dst),
            "epoch": bundle["epoch"], "val_loss": bundle["val_loss"], "label_schema": schema}


def sanitize_summary(summary: dict) -> dict:
    """Quita directorios locales de las rutas que ``evaluate.py`` guarda en el resumen."""
    clean = dict(summary)
    for key in PATH_KEYS:
        if isinstance(clean.get(key), str) and clean[key]:
            clean[key] = _base_name(clean[key])
    return clean


def export_results(eval_dir: str | Path, dst_dir: str | Path) -> list[str]:
    eval_dir, dst_dir = Path(eval_dir), Path(dst_dir)
    missing = [f for f in REQUIRED_RESULT_FILES if not (eval_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Faltan en {eval_dir}: {missing}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in (*REQUIRED_RESULT_FILES, *OPTIONAL_RESULT_FILES):
        src = eval_dir / name
        if not src.exists():
            continue
        if name == "evaluation_summary.json":
            summary = sanitize_summary(json.loads(src.read_text(encoding="utf-8")))
            (dst_dir / name).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            shutil.copyfile(src, dst_dir / name)
        copied.append(name)
    return copied


def export_reports(reports_dir: str | Path | None, dst_dir: str | Path) -> list[str]:
    if not reports_dir:
        return []
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in REPORT_FILES:
        src = Path(reports_dir) / name
        if src.exists():
            shutil.copyfile(src, dst_dir / name)
            copied.append(name)
    return copied


def env_file_text(model_a: str, model_b: str, alpha: float, eval_name: str, deploy_rel: str, models_rel: str,
                  calibration_files: dict | None = None) -> str:
    """``calibration_files``: {variable de entorno: nombre de archivo dentro de deploy/}, en el orden de escritura."""
    if calibration_files is None:
        calibration_files = {"ECG_TEMPERATURES": "temperatures_a.csv", "ECG_TEMPERATURES_B": "temperatures_b.csv"}
    calibration = "".join(f"{var}={deploy_rel}/{name}\n" for var, name in calibration_files.items())
    return (
        "# Generado por tools/export_inference_bundle.py. Rutas relativas a la raíz del repositorio.\n"
        f"ECG_SAVED={models_rel}\n"
        f"ECG_MODEL={models_rel}/{model_a}\n"
        f"ECG_MODEL_B={models_rel}/{model_b}\n"
        f"ECG_ALPHA={alpha}\n"
        f"{calibration}"
        f"ECG_THRESHOLDS={deploy_rel}/results/{eval_name}/thresholds_validation.csv\n"
        f"ECG_EVAL_DIR={deploy_rel}/results/{eval_name}\n"
        f"ECG_EXP_DIR={deploy_rel}/reports\n"
    )


def _stage_calibration(temperatures_a, temperatures_b, platt_a, platt_b, platt_ensemble) -> dict:
    """Valida los CSV de calibración y devuelve {archivo en deploy/: ruta origen} más las variables de entorno."""
    for label, temperature, platt in (("A", temperatures_a, platt_a), ("B", temperatures_b, platt_b)):
        if not temperature and not platt:
            raise ValueError(f"El modelo {label} no tiene calibración: indique --temperatures-{label.lower()} o --platt-{label.lower()}.")
    names = list(load.CLASS_NAMES)
    for path in (temperatures_a, temperatures_b):
        if path:
            load_temperatures_csv(path, names)
    for path in (platt_a, platt_b, platt_ensemble):
        if path:
            load_platt_csv(path, names)
    files, env = {}, {}
    for src, name, var in ((temperatures_a, "temperatures_a.csv", "ECG_TEMPERATURES"),
                           (temperatures_b, "temperatures_b.csv", "ECG_TEMPERATURES_B"),
                           (platt_a, "platt_a.csv", "ECG_PLATT"), (platt_b, "platt_b.csv", "ECG_PLATT_B"),
                           (platt_ensemble, "platt_ensemble.csv", "ECG_PLATT_ENSEMBLE")):
        if src:
            files[name] = src
            env[var] = name
    return {"files": files, "env": env}


def build_bundle(checkpoint_a, checkpoint_b, eval_dir, temperatures_a=None, temperatures_b=None, reports_dir=None,
                 out_root: str | Path = REPO_ROOT, name_a: str = "resnet34_v1", name_b: str = "resnet34_v2",
                 alpha: float | None = None, platt_a=None, platt_b=None, platt_ensemble=None) -> dict:
    calibration = _stage_calibration(temperatures_a, temperatures_b, platt_a, platt_b, platt_ensemble)
    out_root = Path(out_root)
    models_dir, deploy_dir = out_root / "models", out_root / "deploy"
    eval_dir = Path(eval_dir)
    if alpha is None:
        alpha = float(json.loads((eval_dir / "evaluation_summary.json").read_text(encoding="utf-8")).get("ensemble_alpha", 0.5))

    meta_a = export_model(checkpoint_a, models_dir / f"{name_a}.pt")
    meta_b = export_model(checkpoint_b, models_dir / f"{name_b}.pt")
    if meta_a["label_schema"] != meta_b["label_schema"]:
        raise ValueError("Los dos modelos declaran esquemas distintos; no se pueden promediar.")
    results = export_results(eval_dir, deploy_dir / "results" / eval_dir.name)
    reports = export_reports(reports_dir, deploy_dir / "reports")
    deploy_dir.mkdir(parents=True, exist_ok=True)
    for name, src in calibration["files"].items():
        shutil.copyfile(src, deploy_dir / name)
    (deploy_dir / "ensemble.env").write_text(
        env_file_text(meta_a["file"], meta_b["file"], alpha, eval_dir.name, "deploy", "models", calibration["env"]),
        encoding="utf-8")
    if platt_ensemble:
        method = "platt+ensemble"
    elif platt_a or platt_b:
        method = "platt" if not (temperatures_a or temperatures_b) else "mixed"
    else:
        method = "temperature"

    import torch

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "label_schema": meta_a["label_schema"],
        "ensemble_alpha": alpha,
        "calibration": {"method": method, "files": sorted(calibration["files"])},
        "models": {"a": meta_a, "b": meta_b},
        "results_dir": f"results/{eval_dir.name}",
        "results_files": results,
        "report_files": reports,
    }
    (deploy_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Empaqueta modelos y resultados para desplegar la webapp.")
    p.add_argument("--checkpoint-a", required=True)
    p.add_argument("--checkpoint-b", required=True)
    p.add_argument("--eval-dir", required=True, help="Carpeta de evaluación del ensemble (p. ej. runs/eval/ensemble-f1-platt)")
    p.add_argument("--temperatures-a", default=None, help="temperatures_validation.csv del modelo A (calibración por temperatura)")
    p.add_argument("--temperatures-b", default=None, help="temperatures_validation.csv del modelo B")
    p.add_argument("--platt-a", default=None, help="platt_validation.csv del modelo A (calibración afín; gana a la temperatura)")
    p.add_argument("--platt-b", default=None, help="platt_validation.csv del modelo B")
    p.add_argument("--platt-ensemble", default=None, help="platt_ensemble_validation.csv que calibra el promedio del ensemble")
    p.add_argument("--reports-dir", default=None, help="Carpeta con model_comparison.csv y robustness.csv")
    p.add_argument("--out-root", default=str(REPO_ROOT))
    p.add_argument("--name-a", default="resnet34_v1")
    p.add_argument("--name-b", default="resnet34_v2")
    p.add_argument("--alpha", type=float, default=None, help="Por defecto, el de evaluation_summary.json")
    args = p.parse_args()
    manifest = build_bundle(args.checkpoint_a, args.checkpoint_b, args.eval_dir, args.temperatures_a,
                            args.temperatures_b, args.reports_dir, args.out_root, args.name_a, args.name_b, args.alpha,
                            platt_a=args.platt_a, platt_b=args.platt_b, platt_ensemble=args.platt_ensemble)
    for key in ("a", "b"):
        m = manifest["models"][key]
        print(f"models/{m['file']}: {m['bytes'] / 1e6:.1f} MB  sha256={m['sha256'][:16]}…  época={m['epoch']}")
    print(f"calibración: {manifest['calibration']['method']} | alpha={manifest['ensemble_alpha']} | resultados: {len(manifest['results_files'])} archivos | "
          f"reportes: {len(manifest['report_files'])} | env: deploy/ensemble.env")


if __name__ == "__main__":
    main()
