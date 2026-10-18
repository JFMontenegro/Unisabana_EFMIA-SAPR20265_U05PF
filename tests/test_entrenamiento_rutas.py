"""Las corridas nuevas quedan disponibles para evaluar en su propio mercado."""

from __future__ import annotations

import json

import pytest

import evaluate
import plots
import train
import train_alt
from agent import AgenteDQN, Config
from ppo import ConfigPPO


@pytest.fixture
def salida_temporal(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "RAIZ", tmp_path)
    monkeypatch.setattr(plots, "FIGURAS", tmp_path / "figuras")
    return tmp_path


def _verificar_corrida(raiz, mercado, etiqueta, extension):
    nombre = "results" if mercado == "btc" else "results_brent"
    carpeta = raiz / nombre / "corridas" / etiqueta
    for archivo in ("config.json", "mejor.json", "resumen.json", "entrenamiento.csv"):
        assert (carpeta / archivo).exists()
    assert (carpeta / f"mejor{extension}").exists()
    cfg = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
    resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    assert cfg["mercado"] == resumen["mercado"] == mercado
    assert cfg["pasos"] == resumen["pasos"] == 4
    otra = "results_brent" if mercado == "btc" else "results"
    assert not (raiz / otra).exists()
    return carpeta


@pytest.mark.parametrize("mercado", ["btc", "brent"])
def test_dqn_se_entrena_y_evalua_sin_mover_sus_archivos(salida_temporal, mercado) -> None:
    etiqueta = "prueba_rutas"
    resumen = train.entrenar(
        mercado=mercado,
        pasos=4,
        eval_cada=4,
        cfg_agente=Config(oculta=8, capas=1, capacidad=32, calentamiento=2, lote=2),
        largo_episodio=4,
        etiqueta=etiqueta,
        verboso=False,
    )
    carpeta = _verificar_corrida(salida_temporal, mercado, etiqueta, ".pt")
    assert resumen["carpeta"] == str(carpeta)
    assert (carpeta / "ultimo.pt").exists()
    elegido, mejor = evaluate.seleccionar(etiqueta, mercado=mercado)
    assert elegido == carpeta
    assert mejor["paso"] == AgenteDQN.cargar(carpeta / "mejor.pt").pasos == 4

    evaluate.evaluar(patron=etiqueta, mercado=mercado, semillas_aleatorias=1)

    generado = json.loads((carpeta.parent.parent / "evaluacion_prueba.json").read_text())
    assert generado["mercado"] == mercado
    assert generado["corrida_elegida_individual"] == etiqueta
    assert generado["miembros_del_comite"] == [etiqueta]
    assert generado["paso_del_checkpoint"] == 4
    assert "DQN" in generado["metricas"]
    assert "DQN (mejor semilla)" in generado["metricas"]


@pytest.mark.parametrize("mercado", ["btc", "brent"])
@pytest.mark.parametrize("algoritmo", ["tabular", "ppo"])
def test_los_entrenadores_alternativos_guardan_en_su_mercado(
    salida_temporal, mercado, algoritmo
) -> None:
    etiqueta = "prueba_rutas"
    if algoritmo == "ppo":
        pytest.importorskip(
            "stable_baselines3",
            reason="PPO requiere el extra opcional: uv sync --extra dev --extra ppo",
        )
        train_alt.entrenar_ppo(
            mercado=mercado,
            pasos=4,
            eval_cada=4,
            cfg_ppo=ConfigPPO(n_steps=4, lote=4, n_epocas=1, oculta=8),
            etiqueta=etiqueta,
        )
        extension = ".zip"
    else:
        train_alt.entrenar_tabular(mercado=mercado, pasos=4, eval_cada=4, etiqueta=etiqueta)
        extension = ".npz"
    _verificar_corrida(salida_temporal, mercado, etiqueta, extension)
