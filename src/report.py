"""Ensambla las tablas y figuras del informe a partir de las corridas guardadas.

Separado de `plots.py` (que solo dibuja) y de `experiments.py` (que solo
entrena): este módulo lee las corridas del mercado elegido y produce los
artefactos que cita el README, para que ninguna cifra del informe se escriba a
mano.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import data
import plots
from train import directorio_resultados


def curvas(patron: str, mercado: str = "btc") -> dict[str, pd.DataFrame]:
    """Carga las bitácoras de entrenamiento de las corridas que casan."""
    salida = {}
    for carpeta in sorted((directorio_resultados(mercado) / "corridas").glob(patron)):
        archivo = carpeta / "entrenamiento.csv"
        if archivo.exists():
            salida[carpeta.name] = pd.read_csv(archivo)
    return salida


def resumen_de_fase(patron: str, mercado: str = "btc") -> pd.DataFrame:
    """Una fila por corrida: el mejor punto de validación y el estado final."""
    filas = []
    for carpeta in sorted((directorio_resultados(mercado) / "corridas").glob(patron)):
        if not (carpeta / "mejor.json").exists():
            continue
        mejor = json.loads((carpeta / "mejor.json").read_text(encoding="utf-8"))
        cfg = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
        curva = pd.read_csv(carpeta / "entrenamiento.csv")
        filas.append(
            {
                "corrida": carpeta.name,
                "recompensa": cfg["recompensa"],
                "bootstrap": bool(cfg.get("bootstrap")),
                "semilla": cfg["semilla"],
                "oculta": cfg["agente"]["oculta"],
                "capas": cfg["agente"]["capas"],
                "decaimiento": cfg["agente"]["decaimiento_pesos"],
                "paso_mejor": mejor["paso"],
                "val_retorno": mejor["retorno_acumulado"],
                "val_sharpe": mejor["sharpe"],
                "val_drawdown": mejor["max_drawdown"],
                "val_rotacion": mejor["rotacion_anual"],
                "val_expuesto": mejor["tiempo_invertido"],
                "tr_retorno_max": curva["tr_retorno"].max(),
                "val_retorno_final": curva["val_retorno"].iloc[-1],
            }
        )
    return pd.DataFrame(filas)


def tabla_por_recompensa(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega por función de recompensa: media y desviación entre semillas."""
    g = df.groupby("recompensa", sort=False)
    return pd.DataFrame(
        {
            "semillas": g.size(),
            "val_retorno_medio": g["val_retorno"].mean(),
            "val_retorno_desv": g["val_retorno"].std(ddof=1),
            "val_sharpe_medio": g["val_sharpe"].mean(),
            "val_drawdown_medio": g["val_drawdown"].mean(),
            "val_rotacion_media": g["val_rotacion"].mean(),
            "val_expuesto_medio": g["val_expuesto"].mean(),
            "tr_retorno_max": g["tr_retorno_max"].max(),
        }
    )


def figura_memorizacion(mercado: str = "btc") -> Path | None:
    """Compara la fase 1 (serie real) con la fase 2 (remuestreo por bloques).

    Es la figura central del informe: muestra que el retorno de entrenamiento
    cae de 10¹² % a ~10² % y que eso es lo que hace que la validación signifique
    algo.
    """
    sin = curvas("f1_log_neto_s0", mercado)
    con = curvas("f2_log_neto_s0", mercado)
    if not sin or not con:
        return None
    return plots.brecha_entrenamiento_validacion(
        {
            "Serie real (fase 1)": next(iter(sin.values())),
            "Remuestreo por bloques (fase 2)": next(iter(con.values())),
        },
        titulo="Memorización: patrimonio final en entrenamiento (línea continua) y validación (línea discontinua)",
        nombre="brecha_memorizacion.png"
        if mercado == "btc"
        else f"brecha_memorizacion_{mercado}.png",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Ensambla tablas y figuras del informe")
    p.add_argument("--mercado", default="btc", choices=list(data.FUENTES))
    p.add_argument("--fases", nargs="*", default=["f1_*", "f2_*", "f3_*", "f4_*"])
    a = p.parse_args()

    resultados = directorio_resultados(a.mercado)
    resultados.mkdir(parents=True, exist_ok=True)
    todo = []
    for patron in a.fases:
        df = resumen_de_fase(patron, mercado=a.mercado)
        if df.empty:
            continue
        fase = patron.split("_")[0]
        df.insert(0, "fase", fase)
        todo.append(df)
        print(f"\n=== {patron}")
        print(df.to_string(index=False))
        if df["recompensa"].nunique() > 1:
            print()
            print(tabla_por_recompensa(df).to_string())
            plots.comparacion_de_recompensas(
                df,
                titulo=f"Retorno en validación por función de recompensa ({fase})",
                nombre=f"recompensas_{fase}.png"
                if a.mercado == "btc"
                else f"recompensas_{a.mercado}_{fase}.png",
            )

    if todo:
        completo = pd.concat(todo, ignore_index=True)
        completo.to_csv(resultados / "resumen_experimentos.csv", index=False)
        print(f"\n-> {resultados / 'resumen_experimentos.csv'}")

    ruta = figura_memorizacion(a.mercado)
    if ruta:
        print(f"-> {ruta}")


if __name__ == "__main__":
    main()
