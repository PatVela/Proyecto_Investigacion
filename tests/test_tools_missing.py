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
        urllib.request, "urlopen", lambda req, timeout=None: _FakeResp(FAKE_INDEX)
    )
    assert cm.obtener_archivos_remotos("http://x/") == {"A0001", "A0002"}
    assert dm.obtener_archivos_remotos("http://x/") == {"A0001", "A0002"}


def test_obtener_remotos_error_devuelve_none_no_un_conjunto_vacio(monkeypatch, capsys):
    """Un conjunto vacío significaba "no falta nada": sin red el script decía que el dataset estaba completo."""
    def boom(req, timeout=None):
        raise ConnectionError("sin red")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert cm.obtener_archivos_remotos("http://x/") is None
    assert "Error" in capsys.readouterr().out


def test_indice_sin_registros_tambien_se_considera_ilegible(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _FakeResp("<html>pagina distinta</html>"))
    assert dm.obtener_archivos_remotos("http://x/") is None


def test_check_missing_comparte_el_lector_de_indices_de_download_missing():
    import inspect

    assert inspect.getsourcefile(cm.obtener_archivos_remotos).endswith("download_missing.py")


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
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["timeout"] = timeout
        return _FakeResp("contenido")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert dm.descargar_faltantes(str(tmp_path), want_ext=".mat") == 0
    assert (tmp_path / "g1" / "A.mat").read_bytes() == b"contenido"
    assert not list((tmp_path / "g1").glob("*.part"))
    assert seen["timeout"] == dm.TIMEOUT_S                           # sin tope una conexión colgada bloqueaba el script
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


# ---- una descarga cortada o un índice ilegible no se dan por buenos ------------------------------------------------------

class _CutResp(_FakeResp):
    """Respuesta cuya lectura falla a mitad (conexión cortada o cuerpo más corto que Content-Length)."""

    def __init__(self, error):
        super().__init__("")
        self._error = error

    def read(self):
        raise self._error


def _cut_connection(error=None):
    error = error or ConnectionResetError("conexion cortada")
    return lambda req, timeout=None: _CutResp(error)


def test_una_descarga_cortada_no_deja_archivos_con_el_nombre_final(tmp_path, monkeypatch):
    """Antes quedaba un A.mat vacío y la ejecución siguiente decía "tu dataset ya está 100 % completo"."""
    import http.client

    _pair(tmp_path / "g1", "A", with_mat=False)
    for error in (ConnectionResetError("x"), http.client.IncompleteRead(b"ab", 10)):
        monkeypatch.setattr(urllib.request, "urlopen", _cut_connection(error))
        assert dm.descargar_faltantes(str(tmp_path), want_ext=".mat") == 1
        assert sorted(p.name for p in (tmp_path / "g1").iterdir()) == ["A.hea"]    # ni A.mat ni A.mat.part
    # con red otra vez, sigue figurando como pendiente y se descarga
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _FakeResp("completo"))
    assert dm.descargar_faltantes(str(tmp_path), want_ext=".mat") == 0
    assert (tmp_path / "g1" / "A.mat").read_bytes() == b"completo"


def test_pares_sin_red_no_afirma_que_el_dataset_esta_completo(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dm, "obtener_archivos_remotos", lambda url: None)
    assert dm.descargar_pares_faltantes(str(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "100% completo" not in out and "NO se pudo comprobar 46 carpeta(s)" in out


def test_pares_con_una_descarga_cortada_devuelve_1_y_no_deja_basura(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dm, "obtener_archivos_remotos", lambda url: {"S0001"} if url.endswith("ptb/g1/") else set())
    monkeypatch.setattr(urllib.request, "urlopen", _cut_connection())
    assert dm.descargar_pares_faltantes(str(tmp_path)) == 1
    folder = tmp_path / "physionet.org/files/challenge-2020/1.0.2/training/ptb/g1"
    assert list(folder.iterdir()) == []
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _FakeResp("ok"))
    assert dm.descargar_pares_faltantes(str(tmp_path)) == 0                       # ahora sí los baja
    assert sorted(p.name for p in folder.iterdir()) == ["S0001.hea", "S0001.mat"]


def test_check_pares_sin_red_no_dice_excelente(tmp_path, monkeypatch, capsys):
    leaf = tmp_path / "physionet.org" / "files" / "challenge-2020" / "1.0.2" / "training" / "cpsc" / "g1"
    _pair(leaf, "R1")
    monkeypatch.setattr(cm, "obtener_archivos_remotos", lambda url: None)
    assert cm.verificar_parejas_faltantes(str(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "Excelente" not in out and "NO se pudo comprobar 1 carpeta(s): cpsc/g1" in out


def test_check_pares_completo_devuelve_0(tmp_path, monkeypatch, capsys):
    leaf = tmp_path / "physionet.org" / "files" / "challenge-2020" / "1.0.2" / "training" / "cpsc" / "g1"
    _pair(leaf, "R1")
    monkeypatch.setattr(cm, "obtener_archivos_remotos", lambda url: {"R1"})
    assert cm.verificar_parejas_faltantes(str(tmp_path)) == 0
    assert "Excelente" in capsys.readouterr().out


def test_el_codigo_de_salida_de_main_refleja_el_resultado(monkeypatch):
    monkeypatch.setattr(dm, "descargar_faltantes", lambda *a, **k: 1)
    monkeypatch.setattr(cm, "verificar_faltantes", lambda *a, **k: 0)
    assert dm.main(["--kind", "mats"]) == 1 and cm.main(["--kind", "mats"]) == 0


def test_descargar_en_una_carpeta_sin_registros_no_dice_que_todo_esta_completo(tmp_path, capsys):
    """Una --data-dir equivocada daba "¡No faltan archivos! Todo está completo." con código 0."""
    assert dm.descargar_faltantes(str(tmp_path), want_ext=".mat") == 1
    out = capsys.readouterr().out
    assert "No se encontraron archivos .hea" in out and "completo" not in out
