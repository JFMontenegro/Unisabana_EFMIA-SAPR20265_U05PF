"""Descarga y preparación de las series de precios.

Mercados
--------
- `btc`    : **mercado principal**. BTC/USD horario, 33 259 barras del
             15-may-2018 al 1-mar-2022. Es el archivo `examples/data/
             BTC_USD-Hourly.csv` del repositorio de **Gym Trading Env**
             (ClementPerroud), la librería que el profesor señaló como fuente
             de datos para este proyecto, y es el mismo archivo que usa su
             ejemplo oficial.
- `brent`  : réplica de robustez. Precio spot diario del crudo Brent (EIA),
             9 102 barras desde 1987. Mismo código, otro activo, otra
             frecuencia y 39 años en vez de 4.
- `sp500m` : segunda réplica, S&P 500 mensual de Shiller (1871-2026).

Sobre el descargador de gym-trading-env
---------------------------------------
`gym_trading_env.downloader.download()` baja OHLCV de Binance, Bitfinex o Huobi
vía ccxt y guarda un pickle por par. Esas APIs no son alcanzables desde el
entorno donde se corrió este proyecto (la red de egreso las bloquea), así que
se usó el CSV que la propia librería distribuye. `cargar_pickle_gte()` lee los
pickles de `download()` para que cambiar de fuente sea una línea:

    from data import cargar_pickle_gte
    mercado = cargar_pickle_gte("data/binance-BTCUSDT-1h.pkl")

El script es idempotente: vuelve a bajar y sobrescribe `data/`.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import features

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data"

FUENTES = {
    "btc": (
        "https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/main/"
        "examples/data/BTC_USD-Hourly.csv",
        "BTC/USD horario — repositorio de Gym Trading Env (examples/data/BTC_USD-Hourly.csv)",
    ),
    "brent": (
        "https://raw.githubusercontent.com/datasets/oil-prices/main/data/brent-daily.csv",
        "EIA — Europe Brent Spot Price FOB (diario), vía datasets/oil-prices",
    ),
    "sp500m": (
        "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv",
        "Robert Shiller — S&P 500 (mensual), vía datasets/s-and-p-500",
    ),
}

VENTANAS = {
    "btc": features.HORARIO,
    "brent": features.DIARIO,
    "sp500m": features.DIARIO,
}

# Barras por año para anualizar métricas. El cripto cotiza 24/7.
BARRAS_POR_ANIO = {"btc": 24 * 365, "brent": 252, "sp500m": 12}

# Largo del episodio de entrenamiento, en barras. Se busca un horizonte
# comparable entre mercados: un mes de cripto (720 h) y un año bursátil (252
# días) son ambos del orden de 3-4 % del tramo de entrenamiento.
LARGO_EPISODIO = {"btc": 720, "brent": 252, "sp500m": 120}

# Cada cuántas barras puede decidir el agente, por mercado. Se comparan en la
# fase 6; el primero de cada lista es el valor por defecto.
FRECUENCIAS = {"btc": (1, 24, 168), "brent": (1, 5, 21), "sp500m": (1, 3)}


# ---------------------------------------------------------------------------
# Partición temporal
#
# CONGELADA antes de mirar un solo gráfico de cada mercado. Se define aquí, en
# el módulo de datos, para que ningún script de entrenamiento pueda moverla
# «solo esta vez».
#
# BTC (congelada el 15-oct-2026): 65 / 15 / 20 % en barras.
#   El tramo de validación (nov-2020 → may-2021) es el mercado alcista de 2021
#   y el de prueba (jun-2021 → mar-2022) es el techo de 69 000 USD y la caída
#   posterior. No se eligieron por eso: se eligieron por orden cronológico.
#
# Brent (congelada el 14-oct-2026): 64 / 14 / 22 %.
PARTICION = {
    "btc": {
        "entrenamiento": ("2018-05-15", "2020-10-31"),
        "validacion": ("2020-11-01", "2021-05-31"),
        "prueba": ("2021-06-01", "2022-12-31"),
    },
    "brent": {
        "entrenamiento": ("1987-01-01", "2013-12-31"),
        "validacion": ("2014-01-01", "2018-12-31"),
        "prueba": ("2019-01-01", "2026-12-31"),
    },
    "sp500m": {
        "entrenamiento": ("1871-01-01", "1979-12-31"),
        "validacion": ("1980-01-01", "1999-12-31"),
        "prueba": ("2000-01-01", "2026-12-31"),
    },
}


@dataclass(frozen=True)
class Mercado:
    """Una serie de precios ya validada."""

    nombre: str
    precios: pd.DataFrame  # índice DatetimeIndex, columna 'precio' (+ OHLCV si hay)
    fuente: str
    ventanas: features.Ventanas
    barras_anio: int
    avisos: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return len(self.precios)


def partir(precios: pd.DataFrame, mercado: str, tramo: str) -> pd.DataFrame:
    """Devuelve un tramo de la partición congelada.

    El tramo arranca `ventanas.calentamiento` barras antes de su fecha de
    inicio para que el primer estado esté completo. Esas barras previas se usan
    **solo** para calcular indicadores, nunca se opera sobre ellas; el entorno
    recibe el índice del primer paso operable y no retrocede de ahí.
    """
    calentamiento = VENTANAS[mercado].calentamiento
    desde, hasta = PARTICION[mercado][tramo]
    mascara = (precios.index >= desde) & (precios.index <= hasta)
    posiciones = np.flatnonzero(mascara)
    if posiciones.size == 0:
        raise ValueError(f"{mercado}/{tramo}: la partición no cubre ninguna observación")
    inicio = max(0, int(posiciones[0]) - calentamiento)
    fin = int(posiciones[-1]) + 1
    tramo_df = precios.iloc[inicio:fin].copy()
    # Si no hay suficientes barras previas (el inicio de la serie), el primer
    # paso operable se corre hacia adelante hasta tener historia.
    tramo_df.attrs["primer_paso"] = max(int(posiciones[0]) - inicio, calentamiento)
    tramo_df.attrs["tramo"] = tramo
    tramo_df.attrs["mercado"] = mercado
    return tramo_df


def descargar(nombre: str) -> pd.DataFrame:
    """Baja el CSV crudo de la fuente y lo deja en `data/<nombre>_crudo.csv`."""
    url, _ = FUENTES[nombre]
    with urllib.request.urlopen(url, timeout=120) as r:
        bruto = r.read().decode("utf-8")
    DATOS.mkdir(parents=True, exist_ok=True)
    (DATOS / f"{nombre}_crudo.csv").write_text(bruto, encoding="utf-8")
    return pd.read_csv(io.StringIO(bruto))


def _normalizar(nombre: str, crudo: pd.DataFrame) -> pd.DataFrame:
    if nombre == "btc":
        df = crudo.rename(columns={"date": "fecha", "close": "precio"})
        columnas = ["fecha", "precio"] + [
            c for c in ("open", "high", "low", "volume", "Volume USD") if c in df.columns
        ]
        df = df[columnas].copy()
    elif nombre == "brent":
        df = crudo.rename(columns={"Date": "fecha", "Price": "precio"})[["fecha", "precio"]].copy()
    elif nombre == "sp500m":
        df = crudo.rename(columns={"Date": "fecha", "SP500": "precio"})[["fecha", "precio"]].copy()
    else:  # pragma: no cover
        raise KeyError(nombre)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["precio"] = pd.to_numeric(df["precio"], errors="coerce")
    return df


def _revisar_volumen(df: pd.DataFrame, avisos: list[str]) -> pd.DataFrame:
    """Detecta y corrige las columnas de volumen intercambiadas.

    **Defecto real del dataset recomendado.** En `BTC_USD-Hourly.csv`, hasta
    principios de 2021 las columnas `volume` y `Volume USD` están al revés: la
    que dice «volume» trae los dólares y la que dice «Volume USD» trae los
    bitcoins. Se detecta con una identidad que tiene que cumplirse siempre:

        Volume USD ≈ volume × precio

    En 2021 la razón `Volume USD / volume` es igual al precio de cierre; antes
    de 2021 es del orden de 1/precio². El ejemplo oficial de la librería
    construye `feature_volume` a partir de `Volume USD`, así que un agente
    entrenado tal cual recibe una variable que cambia de significado a mitad de
    la serie. Aquí se corrige fila por fila y se deja constancia.

    Tras corregir, la identidad se cumple con una mediana de error del 0.04 %,
    pero no exactamente: el volumen en dólares está calculado con un precio
    medio de la hora, no con el cierre, así que en horas de mucho movimiento
    (el 13-mar-2020, por ejemplo) la diferencia llega al 9 %. Eso no es un
    defecto, es la definición de la columna.

    Este proyecto no usa volumen en el estado —los diez indicadores son de
    precio, para que BTC y Brent sean comparables— así que la corrección no
    afecta ningún resultado reportado. Se hace igual porque el archivo queda
    disponible para quien sí quiera usarlo.
    """
    if not {"volume", "Volume USD"} <= set(df.columns):
        return df
    vol = df["volume"].to_numpy(dtype=np.float64)
    usd = df["Volume USD"].to_numpy(dtype=np.float64)
    precio = df["precio"].to_numpy(dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        # ¿Qué orientación cumple mejor «USD = unidades × precio»?
        err_actual = np.abs(np.log((usd + 1e-9) / (vol * precio + 1e-9)))
        err_invertida = np.abs(np.log((vol + 1e-9) / (usd * precio + 1e-9)))
    invertidas = np.nan_to_num(err_invertida, nan=np.inf) < np.nan_to_num(err_actual, nan=np.inf)

    if invertidas.any():
        n = int(invertidas.sum())
        hasta = df.loc[invertidas, "fecha"].max()
        avisos.append(
            f"columnas de volumen intercambiadas en {n} de {len(df)} filas "
            f"({100 * n / len(df):.1f} %), hasta {hasta:%Y-%m-%d}; corregidas"
        )
        df = df.copy()
        v, u = df["volume"].to_numpy().copy(), df["Volume USD"].to_numpy().copy()
        v2, u2 = v.copy(), u.copy()
        v2[invertidas], u2[invertidas] = u[invertidas], v[invertidas]
        df["volume"] = v2
        df["Volume USD"] = u2
    return df


def validar(df: pd.DataFrame, nombre: str) -> tuple[pd.DataFrame, list[str]]:
    """Controles de calidad que detienen el pipeline si la serie no sirve.

    Son afirmaciones duras a propósito: una serie de precios con huecos,
    duplicados o ceros produce retornos inventados y el error no se vería hasta
    la curva de patrimonio.
    """
    avisos: list[str] = []
    df = df.sort_values("fecha").reset_index(drop=True)

    if df["fecha"].duplicated().any():
        duplicadas = df.loc[df["fecha"].duplicated(), "fecha"].tolist()[:5]
        raise ValueError(f"{nombre}: fechas duplicadas {duplicadas}")

    nulos = int(df["precio"].isna().sum())
    if nulos:
        # Una serie de precios no se interpola: eso inventa observaciones.
        df = df.dropna(subset=["precio"]).reset_index(drop=True)
        avisos.append(f"{nulos} precios nulos descartados")

    if (df["precio"] <= 0).any():
        n = int((df["precio"] <= 0).sum())
        raise ValueError(f"{nombre}: {n} precios no positivos; el log-retorno no existe")

    df = _revisar_volumen(df, avisos)

    r = np.log(df["precio"]).diff()
    extremos = int((r.abs() > 0.5).sum())
    if extremos:
        avisos.append(f"{extremos} saltos de |log-retorno| > 50 %")

    df = df.set_index("fecha")
    df.index.name = "fecha"
    return df, avisos


def cargar(nombre: str = "btc", *, refrescar: bool = False) -> Mercado:
    """Devuelve el mercado listo para el entorno, bajándolo si hace falta."""
    archivo = DATOS / f"{nombre}_crudo.csv"
    crudo = descargar(nombre) if refrescar or not archivo.exists() else pd.read_csv(archivo)
    precios, avisos = validar(_normalizar(nombre, crudo), nombre)
    return Mercado(
        nombre=nombre,
        precios=precios,
        fuente=FUENTES[nombre][1],
        ventanas=VENTANAS[nombre],
        barras_anio=BARRAS_POR_ANIO[nombre],
        avisos=tuple(avisos),
    )


def cargar_pickle_gte(ruta: str | Path, *, nombre: str = "btc") -> Mercado:
    """Carga un pickle producido por `gym_trading_env.downloader.download()`.

    Ese descargador guarda un `DataFrame` con índice de fechas y columnas
    `open, high, low, close, volume`. Aquí se renombra `close` a `precio` y se
    pasa por la misma validación que el resto, de modo que el pipeline no
    distingue entre una fuente y la otra.
    """
    df = pd.read_pickle(ruta)
    if "close" not in df.columns:
        raise ValueError(f"{ruta}: se esperaba una columna 'close' (formato de gym-trading-env)")
    plano = df.reset_index()
    plano = plano.rename(columns={plano.columns[0]: "fecha", "close": "precio"})
    precios, avisos = validar(plano, nombre)
    return Mercado(
        nombre=nombre,
        precios=precios,
        fuente=f"gym_trading_env.downloader → {Path(ruta).name}",
        ventanas=VENTANAS[nombre],
        barras_anio=BARRAS_POR_ANIO[nombre],
        avisos=tuple(avisos),
    )


def main() -> None:
    for nombre in FUENTES:
        m = cargar(nombre, refrescar=True)
        p = m.precios["precio"]
        r = np.log(p).diff().dropna()
        print(
            f"{nombre:7s} {m.n:6d} obs  "
            f"{p.index.min():%Y-%m-%d} -> {p.index.max():%Y-%m-%d}  "
            f"precio [{p.min():.2f}, {p.max():.2f}]  "
            f"sigma por barra {r.std():.5f}  "
            f"anualizada {r.std() * np.sqrt(m.barras_anio):.2f}"
        )
        print(f"        fuente: {m.fuente}")
        print(f"        ventanas: {m.ventanas.nombre} ({m.ventanas.descripcion})")
        for aviso in m.avisos:
            print(f"        aviso: {aviso}")


if __name__ == "__main__":
    main()
