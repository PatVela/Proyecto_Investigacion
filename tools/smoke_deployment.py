"""Prueba de humo de una instancia de la webapp (local o desplegada).

Comprueba por HTTP: página, /health, métricas, experimentos, curvas, modelos,
una predicción real con un ECG sintético (CSV), un registro de ejemplo incluido,
los artefactos descargables (PDF/PNG) y los rechazos esperados (sin archivo,
extensión no permitida, ejemplo inexistente).
Solo usa la biblioteca estándar, así que funciona contra cualquier URL.

    python tools/smoke_deployment.py http://127.0.0.1:5002
    python tools/smoke_deployment.py https://mi-demo.example --expect-ensemble

Código de salida 0 si todo pasa; 1 si algo falla.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def synthetic_csv(seconds: int = 10, fs: int = 500, seed: int = 0) -> bytes:
    """ECG sintético (picos periódicos + ruido) en mV; basta para recorrer el flujo completo."""
    rng = np.random.default_rng(seed)
    n = seconds * fs
    sig = rng.normal(0, 0.03, (n, 12))
    for beat in range(100, n - 50, int(0.8 * fs)):
        sig[beat - 4:beat + 4, :] += np.linspace(0.2, 1.1, 8)[:, None] * rng.uniform(0.6, 1.2, 12)
    lines = [f"# Sampling Rate: {fs} Hz", ",".join(LEADS)]
    lines += [",".join(f"{v:.5f}" for v in row) for row in sig]
    return ("\n".join(lines) + "\n").encode()


def _multipart(files: list[tuple[str, bytes]], fields: dict | None = None) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for key, value in (fields or {}).items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    for name, content in files:
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
                      "Content-Type: application/octet-stream\r\n\r\n").encode() + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _request(url: str, data: bytes | None = None, headers: dict | None = None, timeout: float = 120):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as err:
        return err.code, err.read(), dict(err.headers)


def _json(body: bytes):
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return {}


def run_smoke(base_url: str, expect_ensemble: bool | None = None, timeout: float = 120, log=print) -> list[tuple[str, bool, str]]:
    base = base_url.rstrip("/")
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))
        log(f"{'PASS' if ok else 'FAIL'} - {name}" + (f" | {detail}" if detail else ""))

    def get(path: str):
        return _request(base + path, timeout=timeout)

    status, body, _ = get("/")
    check("GET / responde 200 con HTML", status == 200 and b"<html" in body.lower(), f"{len(body)} bytes")

    status, body, _ = get("/health")
    health = _json(body)
    check("/health: status ok", status == 200 and health.get("status") == "ok")
    check("/health: umbrales y temperaturas encontrados", bool(health.get("thresholds_found") and health.get("temperatures_found")))
    if expect_ensemble is not None:
        check(f"/health: ensemble == {expect_ensemble}", bool(health.get("ensemble")) == expect_ensemble)
    ensemble = bool(health.get("ensemble"))

    status, body, _ = get("/metrics")
    metrics = _json(body)
    check("/metrics: resultados encontrados", status == 200 and metrics.get("found") is True,
          f"{len(metrics.get('per_class_test', []))} clases")
    status, body, _ = get("/experiments")
    experiments = _json(body)
    check("/experiments: comparación y robustez", status == 200 and experiments.get("comparison_found") and experiments.get("robustness_found"))
    for split in ("test", "validation"):
        status, body, _ = get(f"/curves?split={split}")
        curves = _json(body)
        check(f"/curves {split}", status == 200 and curves.get("found") and len(curves.get("classes", [])) == 12)
    status, body, _ = get("/models")
    check("/models lista al menos un checkpoint", status == 200 and len(_json(body).get("models", [])) >= 1)

    csv = synthetic_csv()
    payload, ctype = _multipart([("smoke.csv", csv)], {"threshold": "0.5"})
    start = time.time()
    status, body, _ = _request(base + "/predict", payload, {"Content-Type": ctype}, timeout)
    elapsed = time.time() - start
    pred = _json(body)
    ok = status == 200 and pred.get("ok") is True and len(pred.get("predictions", [])) == 12
    check("/predict con CSV sintético", ok, f"{elapsed:.1f}s" if ok else f"{status} {str(pred.get('error'))[:100]}")
    if ok:
        probs = [p["probability"] for p in pred["predictions"]]
        check("/predict: probabilidades válidas", all(0.0 <= p <= 1.0 for p in probs))
        check("/predict: modo ensemble coherente con /health", bool(pred.get("ensemble")) == ensemble)
        for label, key, magic in (("PDF", "pdf_url", b"%PDF-"), ("PNG del trazado", "plot_url", b"\x89PNG")):
            status, content, _ = get(pred.get(key) or "/missing")
            check(f"artefacto: {label} descargable", status == 200 and content[:len(magic)] == magic, f"{len(content) // 1024} KB")

    status, body, _ = get("/examples")
    listing = _json(body).get("examples", []) if status == 200 else []
    check("/examples responde", status == 200)
    if listing:
        chosen = listing[0]["id"]
        payload, ctype = _multipart([], {"example": chosen})
        start = time.time()
        status, body, _ = _request(base + "/predict", payload, {"Content-Type": ctype}, timeout)
        example = _json(body)
        check(f"/predict con el ejemplo incluido {chosen}",
              status == 200 and example.get("ok") is True and example.get("record_name") == chosen,
              f"{time.time() - start:.1f}s")
        payload, ctype = _multipart([], {"example": "../../etc/passwd"})
        status, body, _ = _request(base + "/predict", payload, {"Content-Type": ctype}, timeout)
        check("/predict rechaza un ejemplo inexistente", status == 400 and _json(body).get("ok") is False, str(status))

    status, body, _ = _request(base + "/predict", b"", {"Content-Type": "multipart/form-data; boundary=x"}, timeout)
    check("/predict sin archivos se rechaza (4xx, JSON)", 400 <= status < 500 and _json(body).get("ok") is False, str(status))
    payload, ctype = _multipart([("evil.pt", b"no es un ecg")])
    status, body, _ = _request(base + "/predict", payload, {"Content-Type": ctype}, timeout)
    check("/predict rechaza extensiones no permitidas", status >= 400 and _json(body).get("ok") is False, str(status))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prueba de humo de la webapp ECG")
    parser.add_argument("url", help="URL base, p. ej. http://127.0.0.1:5002")
    parser.add_argument("--expect-ensemble", action="store_true", help="Exige que /health reporte modo ensemble")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    results = run_smoke(args.url, expect_ensemble=True if args.expect_ensemble else None, timeout=args.timeout)
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} comprobaciones OK")
    for name in failed:
        print("  FALLÓ:", name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
