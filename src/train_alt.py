"""Entrenamiento de PPO y Q-Learning tabular con el MISMO protocolo que el DQN.

Que la comparación signifique algo depende de que lo único distinto entre los
tres agentes sea el algoritmo. Por eso este módulo repite, sin atajos, todo lo
que hace `train.py`:

- el mismo entorno, con remuestreo por bloques y deriva sorteada por episodio;
- el mismo tramo de entrenamiento y el mismo de validación;
- el mismo criterio de selección — retorno acumulado en validación — y la
  misma cadencia de evaluación;
- las mismas cinco semillas;
- el mismo comité declarado de antemano;
- y el mismo tramo de prueba, leído una sola vez al final por `evaluate.py`.

Lo único que cambia es qué aprende dentro.
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
from environment import (
    EXPOSICIONES,
    ConfigEntorno,
    EntornoBootstrap,
    politica_tabulada,
    recorrido_completo,
    rodar,
)
from ppo import AgentePPO, ConfigPPO
from tabular import AgenteTabular, ConfigTabular
from train import directorio_resultados, montar

torch.set_num_threads(1)

# La configuración del entorno que ganó en validación para el DQN (README §6).
# Se reutiliza tal cual: cambiarla por agente convertiría la comparación de
# algoritmos en una comparación de entornos.
RECOMPENSA = "sharpe_dif"
DERIVA = 0.25
LARGO_BLOQUE = 72


def _entorno(tramos, escalador, mercado: str, semilla: int, frecuencia: int = 1):
    cfg = ConfigEntorno(
        recompensa=RECOMPENSA,
        largo_episodio=data.LARGO_EPISODIO[mercado],
        frecuencia_decision=frecuencia,
        semilla=semilla,
    )
    env = EntornoBootstrap(
        tramos["entrenamiento"],
        escalador,
        cfg,
        largo_bloque=LARGO_BLOQUE,
        deriva_aleatoria=DERIVA,
    )
    return env, cfg


def _evaluar(agente, precios, escalador, cfg, barras_anio):
    env = recorrido_completo(precios, escalador, cfg)
    bitacora = rodar(env, politica_tabulada(env, agente))
    return metrics.resumir(bitacora, barras_anio)


def _registrar(carpeta: Path, filas: list[dict]) -> None:
    pd.DataFrame(filas).to_csv(carpeta / "entrenamiento.csv", index=False)


def entrenar_ppo(
    *,
    mercado: str = "btc",
    pasos: int = 120_000,
    eval_cada: int = 10_000,
    semilla: int = 0,
    cfg_ppo: ConfigPPO | None = None,
    etiqueta: str | None = None,
) -> dict:
    tramos, escalador = montar(mercado)
    barras_anio = data.BARRAS_POR_ANIO[mercado]
    env, cfg_env = _entorno(tramos, escalador, mercado, semilla)

    cfg = cfg_ppo or ConfigPPO()
    cfg.semilla = semilla
    agente = AgentePPO.nuevo(env, cfg)

    etiqueta = etiqueta or f"ppo_{mercado}_s{semilla}"
    carpeta = directorio_resultados(mercado) / "corridas" / etiqueta
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "config.json").write_text(
        json.dumps(
            {
                "algoritmo": "PPO (Stable-Baselines3)",
                "mercado": mercado,
                "recompensa": RECOMPENSA,
                "pasos": pasos,
                "semilla": semilla,
                "bootstrap": True,
                "largo_bloque": LARGO_BLOQUE,
                "deriva_aleatoria": DERIVA,
                "agente": asdict_safe(cfg),
                "entorno": {k: v for k, v in cfg_env.__dict__.items() if not k.startswith("_")},
                "particion": data.PARTICION[mercado],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    filas: list[dict] = []
    mejor = -np.inf
    t0 = time.time()
    for bloque in range(1, pasos // eval_cada + 1):
        agente.modelo.learn(total_timesteps=eval_cada, reset_num_timesteps=(bloque == 1))
        paso = bloque * eval_cada
        val = _evaluar(agente, tramos["validacion"], escalador, cfg_env, barras_anio)
        tr = _evaluar(agente, tramos["entrenamiento"], escalador, cfg_env, barras_anio)
        filas.append(
            {
                "paso": paso,
                "val_retorno": val["retorno_acumulado"],
                "val_sharpe": val["sharpe"],
                "val_drawdown": val["max_drawdown"],
                "val_rotacion": val["rotacion_anual"],
                "val_expuesto": val["tiempo_invertido"],
                "tr_retorno": tr["retorno_acumulado"],
                "tr_sharpe": tr["sharpe"],
                "minutos": (time.time() - t0) / 60.0,
            }
        )
        _registrar(carpeta, filas)
        if val["retorno_acumulado"] > mejor:
            mejor = val["retorno_acumulado"]
            agente.guardar(carpeta / "mejor")
            (carpeta / "mejor.json").write_text(
                json.dumps({"paso": paso, **val}, indent=2), encoding="utf-8"
            )
        print(
            f"  [PPO {etiqueta}] {paso:>7,}  val {100 * val['retorno_acumulado']:+8.1f} %  "
            f"sharpe {val['sharpe']:+.2f}  dd {100 * val['max_drawdown']:6.1f} %  "
            f"rot {val['rotacion_anual']:7.1f}  exp {100 * val['tiempo_invertido']:3.0f} %  "
            f"tr {100 * tr['retorno_acumulado']:+8.1f} %  ({filas[-1]['minutos']:.1f} min)",
            flush=True,
        )

    resumen = {
        "etiqueta": etiqueta,
        "algoritmo": "PPO",
        "mercado": mercado,
        "semilla": semilla,
        "pasos": pasos,
        "mejor_val_retorno": float(mejor),
        "minutos": (time.time() - t0) / 60.0,
    }
    (carpeta / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return resumen


def entrenar_tabular(
    *,
    mercado: str = "btc",
    pasos: int = 120_000,
    eval_cada: int = 10_000,
    semilla: int = 0,
    cfg_tab: ConfigTabular | None = None,
    etiqueta: str | None = None,
) -> dict:
    tramos, escalador = montar(mercado)
    barras_anio = data.BARRAS_POR_ANIO[mercado]
    env, cfg_env = _entorno(tramos, escalador, mercado, semilla)

    cfg = cfg_tab or ConfigTabular()
    cfg.semilla = semilla

    # Los cortes se estiman con los indicadores del TRAMO DE ENTRENAMIENTO y se
    # congelan, igual que las medias y desviaciones del escalador.
    ventanas = data.VENTANAS[mercado]
    ind_tr = features.construir(tramos["entrenamiento"]["precio"], ventanas)
    estados_tr = escalador.aplicar(ind_tr)[tramos["entrenamiento"].attrs["primer_paso"] :]
    bordes = AgenteTabular.bordes_de_entrenamiento(estados_tr, cortes=cfg.cortes)
    agente = AgenteTabular(bordes, n_acciones=env.action_space.n, cfg=cfg)

    etiqueta = etiqueta or f"tab_{mercado}_s{semilla}"
    carpeta = directorio_resultados(mercado) / "corridas" / etiqueta
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "config.json").write_text(
        json.dumps(
            {
                "algoritmo": "Q-Learning tabular",
                "mercado": mercado,
                "recompensa": RECOMPENSA,
                "pasos": pasos,
                "semilla": semilla,
                "bootstrap": True,
                "largo_bloque": LARGO_BLOQUE,
                "deriva_aleatoria": DERIVA,
                "indicadores_usados": list(__import__("tabular").INDICADORES),
                "agente": asdict_safe(cfg),
                "entorno": {k: v for k, v in cfg_env.__dict__.items() if not k.startswith("_")},
                "particion": data.PARTICION[mercado],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    filas: list[dict] = []
    mejor = -np.inf
    obs, _ = env.reset(seed=semilla)
    t0 = time.time()
    for paso in range(1, pasos + 1):
        accion = agente.actuar(obs)
        obs2, recompensa, terminado, truncado, _ = env.step(accion)
        agente.aprender(obs, accion, recompensa, obs2, terminado)
        obs = obs2
        if truncado or terminado:
            obs, _ = env.reset()

        if paso % eval_cada == 0:
            val = _evaluar(agente, tramos["validacion"], escalador, cfg_env, barras_anio)
            tr = _evaluar(agente, tramos["entrenamiento"], escalador, cfg_env, barras_anio)
            cob = agente.cobertura()
            filas.append(
                {
                    "paso": paso,
                    "epsilon": agente.epsilon,
                    "cobertura": cob["cobertura"],
                    "val_retorno": val["retorno_acumulado"],
                    "val_sharpe": val["sharpe"],
                    "val_drawdown": val["max_drawdown"],
                    "val_rotacion": val["rotacion_anual"],
                    "val_expuesto": val["tiempo_invertido"],
                    "tr_retorno": tr["retorno_acumulado"],
                    "minutos": (time.time() - t0) / 60.0,
                }
            )
            _registrar(carpeta, filas)
            if val["retorno_acumulado"] > mejor:
                mejor = val["retorno_acumulado"]
                agente.guardar(carpeta / "mejor.npz")
                (carpeta / "mejor.json").write_text(
                    json.dumps({"paso": paso, **val, **cob}, indent=2), encoding="utf-8"
                )
            print(
                f"  [TAB {etiqueta}] {paso:>7,}  val {100 * val['retorno_acumulado']:+8.1f} %  "
                f"sharpe {val['sharpe']:+.2f}  rot {val['rotacion_anual']:7.1f}  "
                f"exp {100 * val['tiempo_invertido']:3.0f} %  "
                f"cobertura {100 * cob['cobertura']:4.0f} %  ({filas[-1]['minutos']:.1f} min)",
                flush=True,
            )

    resumen = {
        "etiqueta": etiqueta,
        "algoritmo": "Tabular",
        "mercado": mercado,
        "semilla": semilla,
        "pasos": pasos,
        "mejor_val_retorno": float(mejor),
        **agente.cobertura(),
        "minutos": (time.time() - t0) / 60.0,
    }
    (carpeta / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return resumen


def asdict_safe(cfg) -> dict:
    from dataclasses import asdict

    return {k: v for k, v in asdict(cfg).items() if not k.startswith("_")}


def main() -> None:
    p = argparse.ArgumentParser(description="Entrena PPO o Q-Learning tabular")
    p.add_argument("algoritmo", choices=["ppo", "tabular"])
    p.add_argument("--mercado", default="btc", choices=list(data.FUENTES))
    p.add_argument("--pasos", type=int, default=120_000)
    p.add_argument("--eval-cada", type=int, default=10_000)
    p.add_argument("--semilla", type=int, default=0)
    a = p.parse_args()

    fn = entrenar_ppo if a.algoritmo == "ppo" else entrenar_tabular
    print(
        json.dumps(
            fn(
                mercado=a.mercado,
                pasos=a.pasos,
                eval_cada=a.eval_cada,
                semilla=a.semilla,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["EXPOSICIONES", "entrenar_ppo", "entrenar_tabular"]
