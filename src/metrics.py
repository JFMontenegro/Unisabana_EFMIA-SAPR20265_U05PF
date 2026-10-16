"""Métricas de desempeño que exige el enunciado, y algunas que lo contextualizan.

Las tres obligatorias son retorno acumulado, máximo drawdown y costos de
transacción. Se añaden cuatro más porque sin ellas la comparación engaña:

- **Sharpe anualizado**: dos estrategias con el mismo retorno no son
  equivalentes si una lo consiguió con el doble de volatilidad.
- **Rotación anual**: dice si el agente aprendió a decidir o a vibrar. Una
  rotación de 50 veces al año con costo de 10 bps se come 5 % anual.
- **Tiempo invertido** y **exposición media**: detectan la degeneración más
  común — un agente que se queda siempre en el mismo nivel y cuya aparente
  habilidad es solo la del activo.
- **Peor año**: el drawdown máximo puede ocurrir en un día; el peor año
  calendario es lo que un inversor real habría tenido que aguantar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def resumir(bitacora: pd.DataFrame, barras_anio: int = 252) -> dict[str, float]:
    """Resume una bitácora de `environment.rodar` en un diccionario plano."""
    patrimonio = bitacora["patrimonio"].to_numpy(dtype=np.float64)
    if patrimonio.size == 0:
        raise ValueError("bitácora vacía")

    equity = np.concatenate([[1.0], patrimonio])
    r = equity[1:] / equity[:-1] - 1.0
    n = r.size
    anios = n / barras_anio

    pico = np.maximum.accumulate(equity)
    drawdown = equity / pico - 1.0

    desv = r.std(ddof=1) if n > 1 else 0.0
    sharpe = float(r.mean() / desv * np.sqrt(barras_anio)) if desv > 1e-15 else 0.0

    por_anio = pd.Series(r, index=bitacora.index).groupby(
        pd.Series(bitacora.index).dt.year.to_numpy()
    ).apply(lambda s: float(np.prod(1.0 + s.to_numpy()) - 1.0))

    return {
        "retorno_acumulado": float(equity[-1] - 1.0),
        "retorno_anualizado": float(equity[-1] ** (1.0 / anios) - 1.0) if anios > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "costos_totales": float(bitacora["costos"].iloc[-1]),
        "sharpe": sharpe,
        "volatilidad_anual": float(desv * np.sqrt(barras_anio)),
        "rotacion_anual": float(bitacora["rotacion"].iloc[-1] / anios) if anios > 0 else 0.0,
        "tiempo_invertido": float((bitacora["exposicion"] > 0).mean()),
        "exposicion_media": float(bitacora["exposicion"].mean()),
        "peor_anio": float(por_anio.min()) if len(por_anio) else 0.0,
        "mejor_anio": float(por_anio.max()) if len(por_anio) else 0.0,
        "barras": int(n),
        "patrimonio_final": float(equity[-1]),
    }


def tabla(resumenes: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Arma la tabla comparativa con las filas en el orden en que se pasaron."""
    df = pd.DataFrame(resumenes).T
    orden = [
        "patrimonio_final",
        "retorno_acumulado",
        "retorno_anualizado",
        "max_drawdown",
        "costos_totales",
        "sharpe",
        "volatilidad_anual",
        "rotacion_anual",
        "tiempo_invertido",
        "exposicion_media",
        "peor_anio",
    ]
    return df[[c for c in orden if c in df.columns]]


def formatear(df: pd.DataFrame) -> pd.DataFrame:
    """Versión legible de la tabla, para pegar en el README."""
    salida = pd.DataFrame(index=df.index)
    if "patrimonio_final" in df.columns:
        salida["Capital final"] = (100_000 * df["patrimonio_final"]).map("${:,.0f}".format)
    salida["Retorno acum."] = (100 * df["retorno_acumulado"]).map("{:+.1f} %".format)
    salida["Anualizado"] = (100 * df["retorno_anualizado"]).map("{:+.2f} %".format)
    salida["Máx. drawdown"] = (100 * df["max_drawdown"]).map("{:.1f} %".format)
    salida["Costos"] = (100 * df["costos_totales"]).map("{:.2f} %".format)
    salida["Sharpe"] = df["sharpe"].map("{:.2f}".format)
    salida["Vol. anual"] = (100 * df["volatilidad_anual"]).map("{:.1f} %".format)
    salida["Rotación/año"] = df["rotacion_anual"].map("{:.1f}".format)
    salida["% invertido"] = (100 * df["tiempo_invertido"]).map("{:.0f} %".format)
    salida["Peor año"] = (100 * df["peor_anio"]).map("{:+.1f} %".format)
    return salida
