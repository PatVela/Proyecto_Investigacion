# -*- coding: utf-8 -*-
"""R11 (MM-01): reimplementación en PyTorch del modelo publicado de Ribeiro et al. (Nat. Commun. 11, 1760, 2020) para poder EJECUTAR sus pesos publicados sin instalar TensorFlow.
Arquitectura: la de model.py de su repositorio público (get_model(6): Conv1D 64 + 4 ResidualUnit con 128, 196, 256 y 320 filtros, núcleo 16, preactivación, salida densa sigmoide).
Los pesos son los del archivo model.hdf5 de Zenodo (formato Keras HDF5; DOI 10.5281/zenodo.3625017). NO es el código de los autores: su fidelidad se comprueba comparando con las
predicciones publicadas por ellos sobre su conjunto de prueba (scripts/validar_port_ribeiro.py) antes de usarlo en otra parte.
Convenciones de Keras reproducidas: relleno 'same' asimétrico (el sobrante va a la derecha), BatchNormalization con epsilon 1e-3 y estadísticas móviles, Dropout inactivo, Flatten en orden (tiempo, canal)."""
import re
import h5py
import numpy as np
import torch
import torch.nn.functional as F

CLASES = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]

def _sufijo(nombre):
    m = re.search(r"_(\d+)$", nombre)
    return int(m.group(1)) if m else 0

def cargar_pesos_keras(ruta):
    """Lee los pesos de un modelo Keras HDF5. Devuelve (convs, bns, dense) en orden de creación (por el sufijo numérico del nombre de la capa)."""
    convs, bns, dense = {}, {}, None
    with h5py.File(ruta, "r") as f:
        raiz = f["model_weights"]
        for nombre in [n.decode() if isinstance(n, bytes) else n for n in raiz.attrs["layer_names"]]:
            g = raiz[nombre]
            pesos = [n.decode() if isinstance(n, bytes) else n for n in g.attrs["weight_names"]]
            if not pesos:
                continue
            d = {re.sub(r"_\d+$", "", p.split("/")[-1].split(":")[0]): np.array(g[p]) for p in pesos}   # p. ej. «kernel_7:0» → «kernel»
            if nombre.startswith("conv1d"):
                convs[_sufijo(nombre)] = d["kernel"]
            elif nombre.startswith("batch_normalization"):
                bns[_sufijo(nombre)] = d
            elif nombre.startswith("dense"):
                dense = d
            else:
                raise ValueError(f"capa con pesos no esperada: {nombre}")
    convs = [convs[k] for k in sorted(convs)]
    bns = [bns[k] for k in sorted(bns)]
    return convs, bns, dense

def conv_same(x, kernel, stride=1):
    """Conv1D de Keras con padding='same' y sin sesgo. x: (N, C, L); kernel Keras: (k, in, out)."""
    w = torch.from_numpy(np.ascontiguousarray(kernel.transpose(2, 1, 0)))
    k, n = kernel.shape[0], x.shape[-1]
    salida = -(-n // stride)
    total = max((salida - 1) * stride + k - n, 0)
    izq = total // 2
    return F.conv1d(F.pad(x, (izq, total - izq)), w, stride=stride)

def bn(x, d, eps=1e-3):
    g, b, m, v = (torch.from_numpy(d[k]).view(1, -1, 1) for k in ("gamma", "beta", "moving_mean", "moving_variance"))
    return g * (x - m) / torch.sqrt(v + eps) + b

class RibeiroResNet:
    def __init__(self, ruta_hdf5):
        self.convs, self.bns, self.dense = cargar_pesos_keras(ruta_hdf5)
        assert len(self.convs) == 13 and len(self.bns) == 9 and self.dense is not None, (len(self.convs), len(self.bns))

    @torch.no_grad()
    def __call__(self, x):
        """x: tensor float32 (N, 4096, 12) como en el repositorio de los autores. Devuelve probabilidades (N, 6)."""
        x = x.permute(0, 2, 1).contiguous()                      # (N, 12, 4096)
        x = F.relu(bn(conv_same(x, self.convs[0]), self.bns[0]))
        y = x; ci, bi = 1, 1
        for _ in range(4):                                        # 4 unidades residuales (submuestreo por 4 en cada una)
            ds = 4
            y = F.max_pool1d(y, ds, ds)                           # 'same' sin relleno: las longitudes son múltiplos de 4
            y = conv_same(y, self.convs[ci])                      # conexión de salto 1x1 (cambia los filtros)
            x = conv_same(x, self.convs[ci + 1])
            x = F.relu(bn(x, self.bns[bi]))
            x = conv_same(x, self.convs[ci + 2], stride=ds)
            x = x + y; y = x
            x = F.relu(bn(x, self.bns[bi + 1]))
            ci += 3; bi += 2
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1)            # Flatten en orden (tiempo, canal)
        z = x @ torch.from_numpy(self.dense["kernel"]) + torch.from_numpy(self.dense["bias"])
        return torch.sigmoid(z)
