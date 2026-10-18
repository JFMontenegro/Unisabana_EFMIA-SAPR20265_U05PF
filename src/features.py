"""Indicadores del estado, construidos de forma estrictamente causal.

La regla que gobierna este módulo: **la fila `t` solo puede depender de precios
en los índices `0..t`**. Es la condición que hace honesto todo el experimento y
la que se viola con más facilidad — basta un `rolling(center=True)`, un
`bfill()`, un `pct_change(-1)` o normalizar con la media de toda la serie.

Por eso la regla no se declara, se prueba: `tests/test_sin_fuga.py` recalcula
cada indicador sobre la serie truncada en `t` y exige que la fila `t` sea
idéntica a la de la serie completa. Si alguien introduce una fuga, ese test
falla.

Decisión de normalización: las medias y desviaciones se estiman **solo en el
tramo de entrenamiento** y se congelan. Normalizar con estadísticas de toda la
serie es una fuga sutil pero real: la escala del estado en 2019 llevaría dentro
información de la volatilidad de 2021.

Ventanas y frecuencia
---------------------
Los diez indicadores son los mismos en los dos mercados; lo que cambia son las
ventanas, porque «21 barras» significa un mes en datos diarios y menos de un día
en datos horarios. `VENTANAS` traduce horizontes de calendario a número de
barras para cada frecuencia, de modo que `vol_corta` siga queriendo decir «la
volatilidad reciente» en los dos casos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NOMBRES = (
    "ret_1",
    "ret_corto",
    "ret_medio",
    "vol_corta",
    "razon_vol",
    "dist_sma_corta",
    "dist_sma_larga",
    "rsi",
    "drawdown",
    "pos_rango",
)


@dataclass(frozen=True)
class Ventanas:
    """Longitud de cada ventana, en barras.

    `larga` es la ventana más grande y por tanto el calentamiento: antes de esa
    barra el estado está incompleto y no se opera.
    """

    nombre: str
    ret_corto: int
    ret_medio: int
    vol_corta: int
    vol_larga: int
    sma_corta: int
    sma_larga: int
    rsi: int
    larga: int
    descripcion: str = ""

    @property
    def calentamiento(self) -> int:
        return self.larga


# Diario (Brent): mes, trimestre, año — los horizontes clásicos de una serie
# diaria de 26 años de entrenamiento.
DIARIO = Ventanas(
    nombre="diario",
    ret_corto=5,
    ret_medio=21,
    vol_corta=21,
    vol_larga=252,
    sma_corta=21,
    sma_larga=63,
    rsi=14,
    larga=252,
    descripcion="1 semana · 1 mes · 1 trimestre · 1 año",
)

# Horario (BTC/USD, 24/7): día, semana, mes. La ventana larga es de 720 barras
# (30 días), que sobre un entrenamiento de ~21 700 barras ocupa la misma
# fracción que 252 días sobre los 5 863 del Brent (3.3 % frente a 4.3 %).
HORARIO = Ventanas(
    nombre="horario",
    ret_corto=24,
    ret_medio=168,
    vol_corta=24,
    vol_larga=720,
    sma_corta=24,
    sma_larga=168,
    rsi=14,
    larga=720,
    descripcion="1 hora · 1 día · 1 semana · 1 mes",
)

PRESETS = {"diario": DIARIO, "horario": HORARIO}

# Compatibilidad: el calentamiento por defecto sigue siendo el diario.
CALENTAMIENTO = DIARIO.calentamiento


def _rsi(precio: pd.Series, ventana: int = 14) -> pd.Series:
    """RSI de Wilder. `diff()` mira hacia atrás, `ewm` tampoco adelanta."""
    cambio = precio.diff()
    sube = cambio.clip(lower=0.0)
    baja = (-cambio).clip(lower=0.0)
    media_sube = sube.ewm(alpha=1 / ventana, adjust=False, min_periods=ventana).mean()
    media_baja = baja.ewm(alpha=1 / ventana, adjust=False, min_periods=ventana).mean()
    fuerza = media_sube / media_baja.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + fuerza)


def construir(precio: pd.Series, ventanas: Ventanas = DIARIO) -> pd.DataFrame:
    """Matriz de indicadores sin normalizar, alineada al índice de `precio`.

    Todas las columnas son adimensionales o están en unidades de retorno, para
    que la escala no dependa del nivel del precio (el Brent pasó de 18 a 144 USD
    y el BTC de 3 100 a 68 600).
    """
    v = ventanas
    logp = np.log(precio)
    r = logp.diff()

    ind = pd.DataFrame(index=precio.index)
    ind["ret_1"] = r
    ind["ret_corto"] = logp.diff(v.ret_corto)
    ind["ret_medio"] = logp.diff(v.ret_medio)

    vol_corta = r.rolling(v.vol_corta).std()
    vol_larga = r.rolling(v.vol_larga).std()
    ind["vol_corta"] = vol_corta
    ind["razon_vol"] = np.log(vol_corta / vol_larga)

    ind["dist_sma_corta"] = logp - logp.rolling(v.sma_corta).mean()
    ind["dist_sma_larga"] = logp - logp.rolling(v.sma_larga).mean()

    ind["rsi"] = _rsi(precio, v.rsi) / 100.0 - 0.5

    maximo = precio.rolling(v.larga, min_periods=v.larga).max()
    minimo = precio.rolling(v.larga, min_periods=v.larga).min()
    ind["drawdown"] = np.log(precio / maximo)
    ind["pos_rango"] = (precio - minimo) / (maximo - minimo) - 0.5

    return ind[list(NOMBRES)]


@dataclass(frozen=True)
class Escalador:
    """Media y desviación congeladas del tramo de entrenamiento."""

    medias: np.ndarray
    desviaciones: np.ndarray
    nombres: tuple[str, ...] = NOMBRES
    ventanas: str = field(default="diario")

    @classmethod
    def ajustar(cls, indicadores: pd.DataFrame, ventanas: Ventanas = DIARIO) -> Escalador:
        limpio = indicadores.dropna()
        if limpio.empty:
            raise ValueError("no quedan filas completas para ajustar el escalador")
        # pandas 3 devuelve vistas de solo lectura en `.to_numpy()`; hay que
        # copiar antes de escribir sobre el arreglo.
        desv = np.array(limpio.std(ddof=0).to_numpy(dtype=np.float64), copy=True)
        desv[desv < 1e-12] = 1.0
        return cls(
            medias=np.array(limpio.mean().to_numpy(dtype=np.float64), copy=True),
            desviaciones=desv,
            nombres=tuple(limpio.columns),
            ventanas=ventanas.nombre,
        )

    def aplicar(self, indicadores: pd.DataFrame) -> np.ndarray:
        x = indicadores[list(self.nombres)].to_numpy(dtype=np.float64)
        z = (x - self.medias) / self.desviaciones
        # Recorte a ±5 sigmas de la distribución de entrenamiento. No es
        # cosmético: el 21-abr-2020 el Brent cayó 64 % en una sesión, un valor
        # a ~24 sigmas del entrenamiento. Sin recorte ese estado domina el
        # gradiente y desestabiliza la red; con recorte el agente sabe que el
        # mundo se salió de escala sin que un solo día borre lo aprendido.
        return np.clip(z, -5.0, 5.0)

    def a_dict(self) -> dict:
        return {
            "nombres": list(self.nombres),
            "medias": self.medias.tolist(),
            "desviaciones": self.desviaciones.tolist(),
            "ventanas": self.ventanas,
        }

    @classmethod
    def de_dict(cls, d: dict) -> Escalador:
        """Restaura el escalador, incluidos los nombres diarios antiguos de Brent."""
        nombres = tuple(d["nombres"])
        ventanas = d.get("ventanas", "diario")
        antiguos = {
            "ret_5": "ret_corto",
            "ret_21": "ret_medio",
            "vol_21": "vol_corta",
            "dist_sma_21": "dist_sma_corta",
            "dist_sma_63": "dist_sma_larga",
            "rsi_14": "rsi",
            "drawdown_252": "drawdown",
            "pos_rango_252": "pos_rango",
        }
        if ventanas != "diario" and any(n in antiguos for n in nombres):
            raise ValueError("los nombres antiguos del escalador requieren ventanas diarias")
        # Cada estadística conserva su posición; solo se traduce el nombre.
        return cls(
            medias=np.asarray(d["medias"], dtype=np.float64),
            desviaciones=np.asarray(d["desviaciones"], dtype=np.float64),
            nombres=tuple(antiguos.get(n, n) for n in nombres),
            ventanas=ventanas,
        )
