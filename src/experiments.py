"""Los experimentos del proyecto, en el orden en que se corrieron.

Cuatro fases, cada una con su pregunta:

- `fase1_recompensas` — ¿cuál de las tres funciones de recompensa produce un
  agente que generaliza, entrenando sobre la serie real? Las tres con las
  mismas semillas e hiperparámetros, para que la única diferencia sea la
  recompensa. **Resultado: ninguna.** Las tres memorizan el tramo de
  entrenamiento (retornos de entrenamiento de 10¹² %) y las tres dan
  validación negativa.
- `fase2_bootstrap` — mismas tres recompensas, pero entrenando sobre
  trayectorias remuestreadas por bloques (`EntornoBootstrap`), donde memorizar
  es imposible porque ninguna serie se repite.
- `fase3_deriva` — el remuestreo conserva la deriva alcista del tramo fuente,
  así que en la fase 2 el agente aprende que estar invertido paga y sale con
  exposición alta y constante. Aquí se sortea una deriva distinta por episodio:
  ninguna exposición fija es óptima y el agente tiene que inferir el régimen.
- `fase4_capacidad` — con la recompensa ganadora, ¿cuánta red hace falta?
  Cuatro tamaños con su decaimiento de pesos.
- `fase5_final` — la configuración ganadora con cinco semillas, para que el
  modelo que va a prueba no sea el afortunado de una corrida.

Todo se decide con el tramo de validación. El de prueba no aparece en este
archivo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from agent import Config
from environment import ConfigEntorno
from train import CORRIDAS, entrenar

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "results"
PROCESOS = 2  # dos núcleos disponibles


def _correr(trabajo: dict) -> dict:
    cfg_ag = Config(**trabajo.get("agente", {}))
    cfg_env = ConfigEntorno(recompensa=trabajo["recompensa"], **trabajo.get("entorno", {}))
    return entrenar(
        recompensa=trabajo["recompensa"],
        pasos=trabajo["pasos"],
        eval_cada=trabajo.get("eval_cada", 10_000),
        semilla=trabajo["semilla"],
        cfg_agente=cfg_ag,
        cfg_entorno=cfg_env,
        bootstrap=trabajo.get("bootstrap", False),
        largo_bloque=trabajo.get("largo_bloque"),
        deriva_aleatoria=trabajo.get("deriva_aleatoria", 0.0),
        etiqueta=trabajo["etiqueta"],
        verboso=True,
    )


def _ejecutar(trabajos: list[dict], nombre: str) -> pd.DataFrame:
    hechos = Parallel(n_jobs=PROCESOS, backend="loky")(
        delayed(_correr)(t) for t in trabajos
    )
    df = pd.DataFrame(hechos)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTADOS / f"{nombre}.csv", index=False)
    return df


def fase1_recompensas(pasos: int = 150_000, semillas: tuple[int, ...] = (0, 1)) -> pd.DataFrame:
    trabajos = [
        {
            "recompensa": r,
            "semilla": s,
            "pasos": pasos,
            "etiqueta": f"f1_{r}_s{s}",
        }
        for r in ("log_neto", "drawdown", "sharpe_dif")
        for s in semillas
    ]
    return _ejecutar(trabajos, "fase1_recompensas")


def fase2_bootstrap(pasos: int = 150_000, semillas: tuple[int, ...] = (0, 1)) -> pd.DataFrame:
    trabajos = [
        {
            "recompensa": r,
            "semilla": s,
            "pasos": pasos,
            "bootstrap": True,
            "etiqueta": f"f2_{r}_s{s}",
        }
        for r in ("log_neto", "drawdown", "sharpe_dif")
        for s in semillas
    ]
    return _ejecutar(trabajos, "fase2_bootstrap")


DERIVA = 0.25  # desviación de la deriva anualizada sorteada por episodio


def fase3_deriva(pasos: int = 150_000, semillas: tuple[int, ...] = (0, 1)) -> pd.DataFrame:
    """Remuestreo + deriva aleatoria por episodio.

    La fase 2 quitó la memorización pero dejó al agente con una exposición alta
    y constante: en el mundo remuestreado el activo hereda la deriva alcista de
    1987-2013, así que estar invertido paga en promedio y no hay nada que
    decidir. Aquí cada episodio tiene su propio régimen y la exposición fija
    deja de ser una respuesta.
    """
    trabajos = [
        {
            "recompensa": r,
            "semilla": s,
            "pasos": pasos,
            "bootstrap": True,
            "deriva_aleatoria": DERIVA,
            "etiqueta": f"f3_{r}_s{s}",
        }
        for r in ("log_neto", "drawdown", "sharpe_dif")
        for s in semillas
    ]
    return _ejecutar(trabajos, "fase3_deriva")


def fase4_capacidad(
    recompensa: str, pasos: int = 150_000, semillas: tuple[int, ...] = (0, 1)
) -> pd.DataFrame:
    variantes = {
        "grande": {"oculta": 128, "capas": 2, "decaimiento_pesos": 1e-4},
        "mediana": {"oculta": 64, "capas": 2, "decaimiento_pesos": 1e-3},
        "pequena": {"oculta": 32, "capas": 1, "decaimiento_pesos": 1e-2},
        "minima": {"oculta": 16, "capas": 1, "decaimiento_pesos": 1e-2},
    }
    trabajos = [
        {
            "recompensa": recompensa,
            "semilla": s,
            "pasos": pasos,
            "bootstrap": True,
            "deriva_aleatoria": DERIVA,
            "agente": v,
            "etiqueta": f"f4_{nombre}_s{s}",
        }
        for nombre, v in variantes.items()
        for s in semillas
    ]
    return _ejecutar(trabajos, "fase4_capacidad")


def fase6_frecuencia(
    recompensa: str,
    agente: dict,
    pasos: int = 150_000,
    semillas: tuple[int, ...] = (0, 1),
    frecuencias: tuple[int, ...] | None = None,
    mercado: str = "btc",
) -> pd.DataFrame:
    """¿Cada cuánto conviene decidir?

    Las fases anteriores dejan un agente con rotación de ~50 veces al año. Con
    10 bps por operación eso es ~5 % anual de fricción, frente a un retorno
    esperado del activo de ~9 % anual: la mitad de la rentabilidad se va en
    costos antes de que la política acierte una sola vez. Decidir cada 5 o cada
    21 barras no es una optimización de cómputo, es una decisión de diseño sobre
    la razón señal/costo del problema.
    """
    import data

    frecuencias = frecuencias or data.FRECUENCIAS[mercado]
    trabajos = [
        {
            "recompensa": recompensa,
            "semilla": s,
            "pasos": pasos,
            "bootstrap": True,
            "deriva_aleatoria": DERIVA,
            "agente": agente,
            "entorno": {"frecuencia_decision": f},
            "etiqueta": f"f6_cada{f}_s{s}",
        }
        for f in frecuencias
        for s in semillas
    ]
    return _ejecutar(trabajos, "fase6_frecuencia")


def fase7_final(
    recompensa: str,
    agente: dict,
    frecuencia: int,
    pasos: int = 150_000,
    semillas: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> pd.DataFrame:
    trabajos = [
        {
            "recompensa": recompensa,
            "semilla": s,
            "pasos": pasos,
            "bootstrap": True,
            "deriva_aleatoria": DERIVA,
            "agente": agente,
            "entorno": {"frecuencia_decision": frecuencia},
            "etiqueta": f"f7_final_s{s}",
        }
        for s in semillas
    ]
    return _ejecutar(trabajos, "fase7_final")


def fase5_final(
    recompensa: str,
    agente: dict,
    pasos: int = 150_000,
    semillas: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> pd.DataFrame:
    trabajos = [
        {
            "recompensa": recompensa,
            "semilla": s,
            "pasos": pasos,
            "bootstrap": True,
            "deriva_aleatoria": DERIVA,
            "agente": agente,
            "etiqueta": f"f5_final_s{s}",
        }
        for s in semillas
    ]
    return _ejecutar(trabajos, "fase5_final")


def tabla_validacion(patron: str) -> pd.DataFrame:
    """Lee las bitácoras de las corridas que casan con `patron` y resume el
    mejor punto de validación de cada una."""
    filas = []
    for carpeta in sorted(CORRIDAS.glob(patron)):
        mejor = carpeta / "mejor.json"
        if not mejor.exists():
            continue
        d = json.loads(mejor.read_text(encoding="utf-8"))
        cfg = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
        curva = pd.read_csv(carpeta / "entrenamiento.csv")
        filas.append(
            {
                "corrida": carpeta.name,
                "recompensa": cfg["recompensa"],
                "semilla": cfg["semilla"],
                "oculta": cfg["agente"]["oculta"],
                "capas": cfg["agente"]["capas"],
                "decaimiento": cfg["agente"]["decaimiento_pesos"],
                "paso_mejor": d["paso"],
                "val_retorno": d["retorno_acumulado"],
                "val_sharpe": d["sharpe"],
                "val_drawdown": d["max_drawdown"],
                "val_rotacion": d["rotacion_anual"],
                "val_expuesto": d["tiempo_invertido"],
                "tr_retorno_final": curva["tr_retorno"].iloc[-1],
                "val_retorno_final": curva["val_retorno"].iloc[-1],
            }
        )
    return pd.DataFrame(filas)


def main() -> None:
    p = argparse.ArgumentParser(description="Corre una fase de experimentos")
    p.add_argument("fase", choices=["1", "2", "3", "4", "5"])
    p.add_argument("--pasos", type=int, default=150_000)
    p.add_argument("--recompensa", default="log_neto")
    p.add_argument("--oculta", type=int, default=128)
    p.add_argument("--capas", type=int, default=2)
    p.add_argument("--decaimiento", type=float, default=1e-4)
    a = p.parse_args()

    if a.fase == "1":
        df = fase1_recompensas(pasos=a.pasos)
    elif a.fase == "2":
        df = fase2_bootstrap(pasos=a.pasos)
    elif a.fase == "3":
        df = fase3_deriva(pasos=a.pasos)
    elif a.fase == "4":
        df = fase4_capacidad(a.recompensa, pasos=a.pasos)
    else:
        df = fase5_final(
            a.recompensa,
            {"oculta": a.oculta, "capas": a.capas, "decaimiento_pesos": a.decaimiento},
            pasos=a.pasos,
        )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
