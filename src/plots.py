"""Figuras del informe.

Reglas que se siguen en todas:

- Un solo eje por panel. Cuando hay dos magnitudes de escalas distintas
  (precio y exposición, entrenamiento y validación) se usan paneles apilados
  que comparten el eje x, nunca dos ejes y — un gráfico de doble eje permite
  sugerir cualquier correlación moviendo una escala.
- Leyenda siempre que haya dos o más series, y etiqueta directa cuando caben.
- Rejilla y ejes recesivos; las series llevan el color, el texto no.
- Paleta categórica fija, asignada por entidad y nunca reciclada: si una figura
  tiene menos series, las que quedan conservan su color.

La paleta se validó con el verificador de la guía de visualización: los cuatro
colores de serie pasan banda de luminosidad, piso de croma, separación para
daltonismo (ΔE adyacente mínimo 9.2 en deuteranopía) y piso de visión normal
(ΔE 27.6). El verde agua queda en 2.74:1 de contraste contra el fondo, por
debajo de 3:1, así que cada figura que lo usa lleva además leyenda y la tabla de
métricas completa en el README — la identidad nunca depende solo del color. El
gris de la banda aleatoria no es una serie categórica sino una referencia, y por
eso se deja deliberadamente acromático.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

RAIZ = Path(__file__).resolve().parents[1]
FIGURAS = RAIZ / "figuras"

TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
REJILLA = "#dedcd6"
SUPERFICIE = "#fcfcfb"

# Orden fijo de la paleta categórica. Se asigna por entidad cuando la entidad
# tiene color propio en COLOR; si no, por posición en el diccionario que pasa
# quien llama — nunca se recicla dentro de una figura.
SECUENCIA = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100", "#e34948")

# Color por entidad, no por orden de aparición.
COLOR = {
    "DQN": "#2a78d6",
    "DQN (mejor semilla)": "#e87ba4",
    "Comprar y mantener": "#eb6834",
    "Cruce de medias 21/63": "#1baf7a",
    "50 % fijo": "#6250d6",
    "Todo en efectivo": "#52514e",
    "Aleatoria": "#8a8880",
    "log_neto": "#2a78d6",
    "drawdown": "#eb6834",
    "sharpe_dif": "#1baf7a",
}


def _estilo(ax, titulo: str = "", y: str = "", x: str = "") -> None:
    ax.set_facecolor(SUPERFICIE)
    ax.grid(True, color=REJILLA, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    ax.tick_params(colors=TINTA_2, labelsize=9)
    if titulo:
        ax.set_title(titulo, color=TINTA, fontsize=11, loc="left", pad=10)
    if y:
        ax.set_ylabel(y, color=TINTA_2, fontsize=9)
    if x:
        ax.set_xlabel(x, color=TINTA_2, fontsize=9)


def _guardar(fig, nombre: str) -> Path:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = FIGURAS / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=SUPERFICIE)
    plt.close(fig)
    return ruta


def curvas_de_patrimonio(
    bitacoras: dict[str, pd.DataFrame],
    aleatorias: pd.DataFrame | None = None,
    titulo: str = "Patrimonio en el período de prueba",
    nombre: str = "equity_prueba.png",
) -> Path:
    """Dos paneles: patrimonio arriba, drawdown abajo, mismo eje de fechas.

    El drawdown va en su propio panel y no en un segundo eje y: con dos escalas
    superpuestas se puede hacer que cualquier estrategia parezca dominar.
    """
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[2.2, 1], facecolor=SUPERFICIE
    )

    if aleatorias is not None and not aleatorias.empty:
        pivote = aleatorias.pivot_table(
            index=aleatorias.index, columns="semilla", values="patrimonio"
        )
        ax.fill_between(
            pivote.index,
            pivote.quantile(0.05, axis=1),
            pivote.quantile(0.95, axis=1),
            color=COLOR["Aleatoria"],
            alpha=0.18,
            linewidth=0,
            label="Aleatoria (p5–p95, 30 semillas)",
        )
        ax.plot(
            pivote.index,
            pivote.median(axis=1),
            color=COLOR["Aleatoria"],
            linewidth=1.2,
            linestyle=":",
        )

    for nombre_serie, b in bitacoras.items():
        color = COLOR.get(nombre_serie, "#8a8880")
        ancho = 2.0 if nombre_serie == "DQN" else 1.5
        ax.plot(b.index, b["patrimonio"], color=color, linewidth=ancho, label=nombre_serie)
        equity = np.concatenate([[1.0], b["patrimonio"].to_numpy()])
        dd = equity / np.maximum.accumulate(equity) - 1.0
        ax2.plot(b.index, dd[1:], color=color, linewidth=ancho * 0.8)

    ax.axhline(1.0, color=TINTA_2, linewidth=0.8, linestyle="--", alpha=0.5)
    _estilo(ax, titulo, "Patrimonio (capital inicial = 1)")
    _estilo(ax2, "", "Drawdown", "Fecha")
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(
        frameon=False,
        fontsize=9,
        labelcolor=TINTA_2,
        loc="upper left",
        ncols=2,
    )
    return _guardar(fig, nombre)


def brecha_entrenamiento_validacion(
    curvas: dict[str, pd.DataFrame],
    titulo: str = "Memorización: patrimonio final en entrenamiento y en validación",
    nombre: str = "brecha_memorizacion.png",
) -> Path:
    """Las dos series en un solo eje logarítmico de patrimonio.

    Es la única forma honesta de verlas juntas: el entrenamiento llega a 10¹⁰
    veces el capital y la validación se queda en 0.5. Con escala lineal una de
    las dos es invisible; con dos ejes y la comparación sería una ilusión.
    """
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SUPERFICIE)
    for i, (etiqueta, curva) in enumerate(curvas.items()):
        color = COLOR.get(etiqueta, SECUENCIA[i % len(SECUENCIA)])
        ax.plot(
            curva["paso"],
            1.0 + curva["tr_retorno"],
            color=color,
            linewidth=2.0,
            label=f"{etiqueta} — entrenamiento",
        )
        ax.plot(
            curva["paso"],
            1.0 + curva["val_retorno"],
            color=color,
            linewidth=1.6,
            linestyle="--",
            label=f"{etiqueta} — validación",
        )
    ax.axhline(1.0, color=TINTA_2, linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:g}×" if v >= 0.1 and v < 1e4 else f"$10^{{{int(np.log10(v))}}}$×")
    )
    _estilo(ax, titulo, "Patrimonio final (escala log)", "Pasos del entorno")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TINTA_2, loc="center right")
    return _guardar(fig, nombre)


def curva_de_aprendizaje(
    curvas: dict[str, pd.DataFrame],
    columna: str = "val_retorno",
    titulo: str = "Retorno en validación durante el entrenamiento",
    nombre: str = "curva_aprendizaje.png",
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SUPERFICIE)
    for i, (etiqueta, curva) in enumerate(curvas.items()):
        ax.plot(
            curva["paso"],
            curva[columna],
            color=COLOR.get(etiqueta, SECUENCIA[i % len(SECUENCIA)]),
            linewidth=2.0,
            label=etiqueta,
        )
    ax.axhline(0.0, color=TINTA_2, linewidth=0.8, linestyle="--", alpha=0.5)
    _estilo(ax, titulo, "Retorno acumulado en validación", "Pasos del entorno")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(frameon=False, fontsize=9, labelcolor=TINTA_2)
    return _guardar(fig, nombre)


def exposicion_y_precio(
    bitacora: pd.DataFrame,
    precios: pd.Series,
    titulo: str = "Qué hizo el agente: exposición frente al precio",
    nombre: str = "exposicion.png",
) -> Path:
    """Precio arriba, exposición abajo. Paneles separados, eje x compartido."""
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, height_ratios=[1.6, 1], facecolor=SUPERFICIE
    )
    serie = precios.reindex(bitacora.index)
    ax.plot(serie.index, serie.to_numpy(), color=TINTA_2, linewidth=1.4)
    _estilo(ax, titulo, "Precio del activo (USD)")

    ax2.fill_between(
        bitacora.index,
        0.0,
        bitacora["exposicion"].to_numpy(),
        step="post",
        color=COLOR["DQN"],
        alpha=0.75,
        linewidth=0,
    )
    _estilo(ax2, "", "Exposición", "Fecha")
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    return _guardar(fig, nombre)


def comparacion_de_recompensas(
    tabla: pd.DataFrame,
    titulo: str = "Retorno en validación por función de recompensa",
    nombre: str = "recompensas.png",
) -> Path:
    """Barras agrupadas por recompensa, un punto por semilla encima.

    La barra es el promedio y los puntos son las semillas: sin ellos una
    diferencia entre recompensas puede ser enteramente ruido de inicialización.
    """
    recompensas = list(dict.fromkeys(tabla["recompensa"]))
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SUPERFICIE)
    for i, r in enumerate(recompensas):
        sub = tabla[tabla["recompensa"] == r]
        media = sub["val_retorno"].mean()
        ax.bar(
            i,
            media,
            width=0.56,
            color=COLOR.get(r, "#8a8880"),
            alpha=0.85,
            linewidth=0,
            label=r,
        )
        ax.scatter(
            np.full(len(sub), i) + np.linspace(-0.12, 0.12, len(sub)),
            sub["val_retorno"],
            color=SUPERFICIE,
            edgecolor=TINTA,
            linewidth=1.2,
            s=42,
            zorder=3,
        )
        ax.annotate(
            f"{100 * media:+.1f} %",
            (i, media),
            textcoords="offset points",
            xytext=(0, 8 if media >= 0 else -16),
            ha="center",
            color=TINTA,
            fontsize=9,
        )
    ax.axhline(0.0, color=TINTA_2, linewidth=0.9)
    ax.set_xticks(range(len(recompensas)))
    ax.set_xticklabels(recompensas, color=TINTA_2, fontsize=9.5)
    _estilo(ax, titulo, "Retorno acumulado en validación")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    return _guardar(fig, nombre)


def eda_precio_y_retornos(
    precios: pd.Series,
    particion: dict[str, tuple[str, str]],
    titulo: str = "Serie de precios y la partición temporal congelada",
    nombre: str = "eda_serie.png",
    etiqueta_y: str = "Precio",
    log: bool = True,
) -> Path:
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[1.5, 1], facecolor=SUPERFICIE
    )
    ax.plot(precios.index, precios.to_numpy(), color=TINTA_2, linewidth=1.1)
    sombras = {"entrenamiento": "#2a78d6", "validacion": "#eda100", "prueba": "#e34948"}
    for tramo, (desde, hasta) in particion.items():
        ax.axvspan(
            pd.Timestamp(desde),
            min(pd.Timestamp(hasta), precios.index[-1]),
            color=sombras[tramo],
            alpha=0.10,
            linewidth=0,
            label=tramo.capitalize(),
        )
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _estilo(ax, titulo, etiqueta_y)
    ax.legend(frameon=False, fontsize=9, labelcolor=TINTA_2, loc="upper left", ncols=3)

    r = np.log(precios).diff()
    ax2.plot(r.index, r.to_numpy(), color="#6250d6", linewidth=0.4, alpha=0.75)
    _estilo(ax2, "", "Log-retorno por barra", "Fecha")
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    return _guardar(fig, nombre)
