"""Incrusta en el PDF un PNG de paleta tal cual, sin reconvertirlo a RGB.

reportlab pasa todo PNG a RGB y lo recomprime con zlib sin predictor: el trazado ECG, que como PNG de paleta pesa
≈ 300 KB, ocupaba ≈ 640 KB dentro del PDF. Aquí se reutilizan los datos comprimidos del propio PNG (los bloques
IDAT), que el PDF admite como ``FlateDecode`` con ``/Predictor 15``, y su paleta como espacio ``Indexed``.
Los píxeles son idénticos; solo cambia el tamaño. Si el PNG no es de paleta o está entrelazado, no se usa.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

from reportlab.pdfbase import pdfdoc
from reportlab.platypus.flowables import Flowable

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def read_indexed_png(path: str | Path) -> dict | None:
    """``{width, height, bit_depth, palette, idat, digest}`` de un PNG de paleta sin entrelazar; si no, ``None``."""
    data = Path(path).read_bytes()
    if not data.startswith(_SIGNATURE):
        return None
    header, palette, idat, transparency = None, b"", [], False
    position = len(_SIGNATURE)
    while position + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[position:position + 8])
        body = data[position + 8:position + 8 + length]
        position += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"PLTE":
            palette = body
        elif kind == b"tRNS":
            transparency = True
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
    if header is None or not idat or not palette or transparency:
        return None
    width, height, bit_depth, color_type, _compression, _filter, interlace = header
    if color_type != 3 or interlace != 0 or bit_depth not in (1, 2, 4, 8):
        return None
    return {"width": width, "height": height, "bit_depth": bit_depth, "palette": palette,
            "idat": b"".join(idat), "digest": hashlib.md5(data).hexdigest()}


class _HexString(pdfdoc.PDFObject):
    def __init__(self, raw: bytes):
        self.raw = raw

    def format(self, document):
        return b"<" + self.raw.hex().encode("ascii") + b">"


class _IndexedPngXObject(pdfdoc.PDFImageXObject):
    def __init__(self, name: str, info: dict):
        super().__init__(name)
        self.info = info
        self.width, self.height = info["width"], info["height"]
        self.streamContent = info["idat"]

    def format(self, document):
        info = self.info
        stream = pdfdoc.PDFStream(content=info["idat"])
        entries = stream.dictionary
        entries["Type"] = pdfdoc.PDFName("XObject")
        entries["Subtype"] = pdfdoc.PDFName("Image")
        entries["Width"] = info["width"]
        entries["Height"] = info["height"]
        entries["BitsPerComponent"] = info["bit_depth"]
        entries["ColorSpace"] = pdfdoc.PDFArray([pdfdoc.PDFName("Indexed"), pdfdoc.PDFName("DeviceRGB"),
                                                 len(info["palette"]) // 3 - 1, _HexString(info["palette"])])
        entries["Filter"] = pdfdoc.PDFArray([pdfdoc.PDFName("FlateDecode")])
        entries["DecodeParms"] = pdfdoc.PDFArray([pdfdoc.PDFDictionary(
            {"Predictor": 15, "Colors": 1, "BitsPerComponent": info["bit_depth"], "Columns": info["width"]})])
        return stream.format(document)


class IndexedPngImage(Flowable):
    """Imagen que ocupa ``draw_width × draw_height`` puntos y cuyo PNG de paleta va incrustado sin reconvertir."""

    def __init__(self, info: dict, draw_width: float, draw_height: float):
        super().__init__()
        self.info = info
        self.drawWidth, self.drawHeight = draw_width, draw_height

    def wrap(self, avail_width, avail_height):
        return self.drawWidth, self.drawHeight

    def draw(self):
        canvas, name = self.canv, self.info["digest"]
        registered = canvas._doc.getXObjectName(name)
        if canvas._doc.idToObject.get(registered) is None:          # misma secuencia que Canvas.drawImage
            xobject = _IndexedPngXObject(name, self.info)
            canvas._setXObjects(xobject)
            canvas._doc.Reference(xobject, registered)
            canvas._doc.addForm(name, xobject)
        canvas._currentPageHasImages = 1
        canvas.saveState()
        canvas.scale(self.drawWidth, self.drawHeight)
        canvas._code.append(f"/{registered} Do")
        canvas.restoreState()
        canvas._formsinuse.append(name)


def indexed_png_image(path: str | Path, max_width: float, max_height: float) -> IndexedPngImage | None:
    """Flowable ajustado a ``max_width × max_height`` (sin ampliar, 1 píxel = 1 punto como ``platypus.Image``), o ``None``."""
    info = read_indexed_png(path)
    if info is None:
        return None
    scale = min(max_width / info["width"], max_height / info["height"], 1.0)
    return IndexedPngImage(info, info["width"] * scale, info["height"] * scale)
