"""tools/check_missing y tools/download_missing sin red (monkeypatch)."""

import sys
import urllib.request
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import check_missing as cm
import download_missing as dm


class _FakeResp:
    def __init__(self, html):
        self._raw = html.encode("utf-8")

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


FAKE_INDEX = """
<html><body>
<a href="A0001.hea">A0001.hea</a>
<a href="A0002.hea">A0002.hea</a>
<a href="other.mat">other.mat</a>
</body></html>
"""


def _pair(root, name, with_mat=True, with_hea=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if with_hea:
        (root / f"{name}.hea").write_text("x")
    if with_mat:
        (root / f"{name}.mat").write_bytes(b"x")


# ---- check_missing.verificar_faltantes (solo disco) -------------------------


def test_verificar_faltantes_detecta_mat_faltante(tmp_path, capsys):
    _pair(tmp_path / "g1", "A", with_mat=False)
    _pair(tmp_path / "g1", "B")
    cm.verificar_faltantes(str(tmp_path), want_ext=".mat")
    out = capsys.readouterr().out
    assert "falta .mat" in out
    assert "Todo perfecto" not in out


def test_verificar_faltantes_todo_completo(tmp_path, capsys):
    _pair(tmp_path / "g1", "A")
    cm.verificar_faltantes(str(tmp_path), want_ext=".mat")
    assert "Todo perfecto" in capsys.readouterr().out


def test_verificar_faltantes_dir_vacio(tmp_path, capsys):
    cm.verificar_faltantes(str(tmp_path), want_ext=".mat")
    assert "No se encontraron" in capsys.readouterr().out


# ---- obtener_archivos_remotos (HTML simulado) -------------------------------


def test_obtener_remotos_parsea_indice(monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req: _FakeResp(FAKE_INDEX)
    )
    assert cm.obtener_archivos_remotos("http://x/") == {"A0001", "A0002"}
    assert dm.obtener_archivos_remotos("http://x/") == {"A0001", "A0002"}


def test_obtener_remotos_error_devuelve_vacio(monkeypatch, capsys):
    def boom(req):
        raise ConnectionError("sin red")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert cm.obtener_archivos_remotos("http://x/") == set()
    assert "Error" in capsys.readouterr().out


# ---- verificar_parejas_faltantes (remoto simulado) ---------------------------


def test_parejas_detecta_registro_sin_ambos(tmp_path, monkeypatch, capsys):
    leaf = tmp_path / "physionet.org" / "files" / "challenge-2020" / "1.0.2" / "training" / "cpsc" / "g1"
    _pair(leaf, "R1", with_mat=False)
    monkeypatch.setattr(cm, "obtener_archivos_remotos", lambda url: {"R1", "R2"})
    cm.verificar_parejas_faltantes(str(tmp_path))
    out = capsys.readouterr().out
    assert "cpsc/g1/R2" in out
    assert "FALTA AMBOS (.hea y .mat): 1" in out


# ---- download_missing._remote_subpath (puro) ---------------------------------


def test_remote_subpath():
    full = str(Path("d/physionet.org/files/challenge-2020/1.0.2/training/cpsc/g1/A.hea"))
    assert dm._remote_subpath(full) == "training/cpsc/g1/A.hea"
    assert dm._remote_subpath(str(Path("x/training/ptb/g1/A.mat"))) == "training/ptb/g1/A.mat"
    assert dm._remote_subpath(str(Path("plain/A.mat"))) == "plain/A.mat"


# ---- descargar_faltantes (descarga simulada) ---------------------------------


def test_descargar_faltantes_guarda_mat(tmp_path, monkeypatch, capsys):
    _pair(tmp_path / "g1", "A", with_mat=False)
    monkeypatch.setattr(
        urllib.request, "urlretrieve", lambda url, dest: Path(dest).write_bytes(b"fake")
    )
    dm.descargar_faltantes(str(tmp_path), want_ext=".mat")
    assert (tmp_path / "g1" / "A.mat").exists()
    assert "Exitosos: 1" in capsys.readouterr().out


def test_descargar_faltantes_nada_pendiente(tmp_path, capsys):
    _pair(tmp_path / "g1", "A")
    dm.descargar_faltantes(str(tmp_path), want_ext=".mat")
    assert "No faltan" in capsys.readouterr().out


# ---- descargar_pares_faltantes (remoto vacío simulado) -----------------------


def test_descargar_pares_sin_pendientes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dm, "obtener_archivos_remotos", lambda url: set())
    dm.descargar_pares_faltantes(str(tmp_path))
    assert "100% completo" in capsys.readouterr().out


# ---- main() ruteo ------------------------------------------------------------


def test_main_ruteo_check(monkeypatch):
    calls = []
    monkeypatch.setattr(cm, "verificar_faltantes", lambda *a, **k: calls.append(("f", a, k)))
    monkeypatch.setattr(cm, "verificar_parejas_faltantes", lambda *a, **k: calls.append(("p", a, k)))
    cm.main(["--kind", "heas", "--data-dir", "X"])
    cm.main(["--kind", "pairs", "--data-dir", "Y"])
    assert calls[0][0] == "f" and calls[0][1] == ("X",) and calls[0][2] == {"want_ext": ".hea"}
    assert calls[1][0] == "p" and calls[1][1] == ("Y",)


def test_main_ruteo_download(monkeypatch):
    calls = []
    monkeypatch.setattr(dm, "descargar_faltantes", lambda *a, **k: calls.append(("f", a, k)))
    monkeypatch.setattr(dm, "descargar_pares_faltantes", lambda *a, **k: calls.append(("p", a, k)))
    dm.main(["--kind", "mats", "--data-dir", "X"])
    dm.main(["--kind", "pairs", "--data-dir", "Y"])
    assert calls[0][0] == "f" and calls[0][1] == ("X",) and calls[0][2] == {"want_ext": ".mat"}
    assert calls[1][0] == "p" and calls[1][1] == ("Y",)
