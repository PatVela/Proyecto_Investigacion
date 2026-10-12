"""Paquete ECG para la adaptación CINC2020-12 tipo Hannun.

El paquete usa imports perezosos para que `python -m ecg.load` no cargue el
submódulo antes de ejecutarlo y para que el preprocesamiento funcione aunque
PyTorch todavía no esté instalado.
"""

__all__ = [
    "CLASS_GROUPS_12",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "NUM_LEADS",
    "STANDARD_LEAD_ORDER",
    "TARGET_FS",
    "WINDOW_LENGTH",
    "build_datasets",
    "codes_to_vector",
    "preprocess_ecg_array",
    "ECGClassifier",
    "ECGRegularCNN",
    "ECGResNet34",
    "build_network",
    "count_parameters",
]

_LOAD_EXPORTS = {
    "CLASS_GROUPS_12",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "NUM_LEADS",
    "STANDARD_LEAD_ORDER",
    "TARGET_FS",
    "WINDOW_LENGTH",
    "build_datasets",
    "codes_to_vector",
    "preprocess_ecg_array",
}
_NETWORK_EXPORTS = {"ECGClassifier", "ECGRegularCNN", "ECGResNet34", "build_network"}
_UTIL_EXPORTS = {"count_parameters"}


def __getattr__(name):
    if name in _LOAD_EXPORTS:
        from . import load
        return getattr(load, name)
    if name in _NETWORK_EXPORTS:
        from . import network
        return getattr(network, name)
    if name in _UTIL_EXPORTS:
        from . import util
        return getattr(util, name)
    raise AttributeError(f"module 'ecg' has no attribute {name!r}")
