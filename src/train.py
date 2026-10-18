"""Entrenamiento del agente y selección de modelo en validación.

Protocolo (declarado ANTES de correr nada, para que no haya margen de ajustarlo
a los resultados):

1. El agente se entrena solo sobre el tramo de entrenamiento, en episodios de
   252 barras con inicio aleatorio dentro de ese tramo.
2. Cada `eval_cada` pasos se evalúa en modo greedy, de una sola pasada, sobre
   el tramo de **validación**, y también sobre entrenamiento (para ver la
   brecha, no para seleccionar).
3. **Criterio de selección: retorno acumulado en validación.** Es el objetivo
   declarado del problema y no premia la solución degenerada de quedarse en
   efectivo, que da exactamente 0 %. Las métricas de riesgo se reportan al lado
   pero no entran en la selección.
4. El tramo de prueba no se toca en ningún punto de este archivo. La única
   lectura es la de `evaluate.py`, una vez, al final.

La regla 4 es la que de verdad cuesta cumplir: basta mirar el test una vez y
cambiar un hiperparámetro para que el resultado deje de ser fuera de muestra.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data
import features
import metrics
from agent import AgenteDQN, Config
from environment import (
    ConfigEntorno,
    EntornoBootstrap,
    EntornoInversion,
    politica_tabulada,
    recorrido_completo,
    rodar,
)

RAIZ = Path(__file__).resolve().parents[1]


def directorio_resultados(mercado: str) -> Path:
    """Carpeta compartida por el entrenamiento y la evaluación de un mercado."""
    if mercado not in data.FUENTES:
        raise ValueError(f"mercado desconocido: {mercado}")
    return RAIZ / ("results" if mercado == "btc" else f"results_{mercado}")


# Ruta histórica de BTC, conservada para consumidores externos.
CORRIDAS = RAIZ / "results" / "corridas"

# Un hilo por proceso: la red es diminuta (11→128→128→3) y el paralelismo
# interno de PyTorch cuesta más de lo que ahorra. Dos corridas en paralelo con
# un hilo cada una van más rápido que una corrida con dos hilos.
torch.set_num_threads(1)


def montar(mercado: str = "btc", *, incluir_prueba: bool = False):
    """Carga el mercado, aplica la partición congelada y ajusta el escalador.

    **El tramo de prueba no se carga salvo que se pida explícitamente.** No es
    cosmética: así el código de entrenamiento y de búsqueda no tiene siquiera
    acceso al objeto, y `KeyError: 'prueba'` detendría a quien lo intente.
    El único módulo que pasa `incluir_prueba=True` es `evaluate.py`.
    """
    m = data.cargar(mercado)
    nombres = ["entrenamiento", "validacion"] + (["prueba"] if incluir_prueba else [])
    tramos = {t: data.partir(m.precios, mercado, t) for t in nombres}
    escalador = features.Escalador.ajustar(
        features.construir(tramos["entrenamiento"]["precio"], m.ventanas), m.ventanas
    )
    return tramos, escalador


def evaluar_greedy(
    agente: AgenteDQN,
    precios: pd.DataFrame,
    escalador: features.Escalador,
    cfg_env: ConfigEntorno,
    barras_anio: int = 252,
) -> tuple[dict[str, float], pd.DataFrame]:
    env = recorrido_completo(precios, escalador, cfg_env)
    bitacora = rodar(env, politica_tabulada(env, agente))
    return metrics.resumir(bitacora, barras_anio), bitacora


def entrenar(
    *,
    mercado: str = "btc",
    recompensa: str = "log_neto",
    pasos: int = 200_000,
    eval_cada: int = 10_000,
    semilla: int = 0,
    cfg_agente: Config | None = None,
    cfg_entorno: ConfigEntorno | None = None,
    bootstrap: bool = False,
    largo_bloque: int | None = None,
    deriva_aleatoria: float = 0.0,
    largo_episodio: int | None = None,
    etiqueta: str | None = None,
    verboso: bool = True,
) -> dict:
    tramos, escalador = montar(mercado)
    barras_anio = data.BARRAS_POR_ANIO[mercado]

    cfg_env = cfg_entorno or ConfigEntorno()
    cfg_env.recompensa = recompensa
    cfg_env.semilla = semilla
    # Las longitudes se resuelven por mercado: «252 barras» es un año en datos
    # diarios y diez días en datos horarios.
    if largo_episodio is None:
        largo_episodio = data.LARGO_EPISODIO[mercado]
    cfg_env.largo_episodio = largo_episodio
    if largo_bloque is None:
        # El bloque tiene que ser bastante más corto que el episodio, o el
        # remuestreo deja de mezclar: con un bloque igual al episodio, cada
        # episodio sería un trozo contiguo de la serie real y la memorización
        # volvería. Se usa ~1/10 del episodio en los dos mercados: 21 barras
        # (un mes) sobre 252 en el Brent, 72 barras (tres días) sobre 720 en
        # BTC. En ambos casos el bloque cubre varios días de calendario, que
        # es la escala a la que opera el agrupamiento de volatilidad.
        largo_bloque = 72 if mercado == "btc" else 21

    cfg_ag = cfg_agente or Config()
    cfg_ag.semilla = semilla

    if bootstrap:
        env = EntornoBootstrap(
            tramos["entrenamiento"],
            escalador,
            cfg_env,
            largo_bloque=largo_bloque,
            deriva_aleatoria=deriva_aleatoria,
        )
    else:
        env = EntornoInversion(tramos["entrenamiento"], escalador, cfg_env)
    agente = AgenteDQN(env.observation_space.shape[0], env.action_space.n, cfg_ag)

    etiqueta = etiqueta or f"{mercado}_{recompensa}_s{semilla}"
    carpeta = directorio_resultados(mercado) / "corridas" / etiqueta
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "config.json").write_text(
        json.dumps(
            {
                "mercado": mercado,
                "recompensa": recompensa,
                "pasos": pasos,
                "semilla": semilla,
                "bootstrap": bootstrap,
                "largo_bloque": largo_bloque if bootstrap else None,
                "largo_episodio": largo_episodio,
                "deriva_aleatoria": deriva_aleatoria if bootstrap else None,
                "agente": cfg_ag.__dict__,
                "entorno": {k: v for k, v in cfg_env.__dict__.items() if not k.startswith("_")},
                "particion": data.PARTICION[mercado],
                "escalador": escalador.a_dict(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    bitacora_entrenamiento: list[dict] = []
    mejor = -np.inf
    obs, _ = env.reset(seed=semilla)
    retorno_episodio = 0.0
    retornos = []
    perdidas = []
    t0 = time.time()

    for paso in range(1, pasos + 1):
        accion = agente.actuar(obs)
        obs2, recompensa_paso, terminado, truncado, _ = env.step(accion)
        agente.recordar(obs, accion, recompensa_paso, obs2, terminado)
        perdida = agente.aprender()
        if perdida is not None:
            perdidas.append(perdida)
        retorno_episodio += recompensa_paso
        obs = obs2

        if truncado or terminado:
            retornos.append(retorno_episodio)
            retorno_episodio = 0.0
            obs, _ = env.reset()

        if paso % eval_cada == 0:
            val, _ = evaluar_greedy(agente, tramos["validacion"], escalador, cfg_env, barras_anio)
            tr, _ = evaluar_greedy(agente, tramos["entrenamiento"], escalador, cfg_env, barras_anio)
            fila = {
                "paso": paso,
                "epsilon": agente.epsilon,
                "perdida": float(np.mean(perdidas[-1000:])) if perdidas else np.nan,
                "retorno_episodio": float(np.mean(retornos[-20:])) if retornos else np.nan,
                "val_retorno": val["retorno_acumulado"],
                "val_sharpe": val["sharpe"],
                "val_drawdown": val["max_drawdown"],
                "val_rotacion": val["rotacion_anual"],
                "val_expuesto": val["tiempo_invertido"],
                "tr_retorno": tr["retorno_acumulado"],
                "tr_sharpe": tr["sharpe"],
                "minutos": (time.time() - t0) / 60.0,
            }
            bitacora_entrenamiento.append(fila)
            pd.DataFrame(bitacora_entrenamiento).to_csv(carpeta / "entrenamiento.csv", index=False)

            # Criterio de selección declarado: retorno acumulado en validación.
            if val["retorno_acumulado"] > mejor:
                mejor = val["retorno_acumulado"]
                agente.guardar(carpeta / "mejor.pt")
                (carpeta / "mejor.json").write_text(
                    json.dumps({"paso": paso, **val}, indent=2), encoding="utf-8"
                )
            if verboso:
                print(
                    f"  [{etiqueta}] {paso:>7,}  eps {agente.epsilon:.2f}  "
                    f"val {100 * val['retorno_acumulado']:+7.1f} %  "
                    f"sharpe {val['sharpe']:+.2f}  dd {100 * val['max_drawdown']:6.1f} %  "
                    f"rot {val['rotacion_anual']:5.1f}  exp {100 * val['tiempo_invertido']:3.0f} %  "
                    f"tr {100 * tr['retorno_acumulado']:+8.1f} %  ({fila['minutos']:.1f} min)",
                    flush=True,
                )

    agente.guardar(carpeta / "ultimo.pt")
    resumen = {
        "etiqueta": etiqueta,
        "mercado": mercado,
        "recompensa": recompensa,
        "semilla": semilla,
        "pasos": pasos,
        "bootstrap": bootstrap,
        "mejor_val_retorno": float(mejor),
        "minutos": (time.time() - t0) / 60.0,
        "carpeta": str(carpeta),
    }
    (carpeta / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return resumen


def main() -> None:
    p = argparse.ArgumentParser(description="Entrena el agente inversor con DQN")
    p.add_argument("--mercado", default="btc", choices=list(data.FUENTES))
    p.add_argument(
        "--recompensa", default="log_neto", choices=["log_neto", "drawdown", "sharpe_dif"]
    )
    p.add_argument("--pasos", type=int, default=200_000)
    p.add_argument("--eval-cada", type=int, default=10_000)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--oculta", type=int, default=None)
    p.add_argument("--lambda-drawdown", type=float, default=None)
    p.add_argument("--largo-episodio", type=int, default=None)
    p.add_argument("--etiqueta", default=None)
    a = p.parse_args()

    cfg_ag = Config()
    if a.gamma is not None:
        cfg_ag.gamma = a.gamma
    if a.lr is not None:
        cfg_ag.lr = a.lr
    if a.oculta is not None:
        cfg_ag.oculta = a.oculta

    cfg_env = ConfigEntorno(recompensa=a.recompensa)
    if a.lambda_drawdown is not None:
        cfg_env.lambda_drawdown = a.lambda_drawdown
    if a.largo_episodio is not None:
        cfg_env.largo_episodio = a.largo_episodio

    resumen = entrenar(
        mercado=a.mercado,
        recompensa=a.recompensa,
        pasos=a.pasos,
        eval_cada=a.eval_cada,
        semilla=a.semilla,
        cfg_agente=cfg_ag,
        cfg_entorno=cfg_env,
        etiqueta=a.etiqueta,
    )
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
