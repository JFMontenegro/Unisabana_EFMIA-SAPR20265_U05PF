"""Carga de los modelos incluidos y reproducción de la evaluación publicada."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch

import evaluate
import plots
from agent import AgenteDQN

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture
def montaje(tmp_path, monkeypatch):
    resultados = tmp_path / "results"
    comite = resultados / "comite"
    comite.mkdir(parents=True)
    for nombre in ("semilla_1.pt", "semilla_1.config.json"):
        shutil.copy2(RAIZ / "results" / "comite" / nombre, comite / nombre)
    carpeta = tmp_path / "corridas" / "f7_final_s1"
    carpeta.mkdir(parents=True)
    original = RAIZ / "results" / "corridas" / carpeta.name
    cfg = json.loads((original / "config.json").read_text(encoding="utf-8"))
    mejor = json.loads((original / "mejor.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(evaluate, "RESULTADOS", resultados)
    return carpeta, cfg, mejor, comite


def test_recupera_el_individuo_del_comite_si_faltan_los_pesos_de_la_corrida(montaje) -> None:
    carpeta, cfg, mejor, comite = montaje
    agente = evaluate.cargar_individual(carpeta, cfg, mejor)
    esperado = AgenteDQN.cargar(comite / "semilla_1.pt")

    assert agente.cfg.semilla == cfg["semilla"]
    assert agente.pasos == mejor["paso"]
    for clave, tensor in esperado.q.state_dict().items():
        assert torch.equal(agente.q.state_dict()[clave], tensor)
    assert not (carpeta / "mejor.pt").exists()


def test_prefiere_los_pesos_existentes_de_la_corrida(montaje) -> None:
    carpeta, cfg, mejor, comite = montaje
    shutil.copy2(comite / "semilla_1.pt", carpeta / "mejor.pt")
    # La configuración de respaldo no se consulta cuando está el original.
    (comite / "semilla_1.config.json").write_text("JSON inválido", encoding="utf-8")
    agente = evaluate.cargar_individual(carpeta, cfg, mejor)
    assert agente.cfg.semilla == cfg["semilla"]
    assert agente.pasos == mejor["paso"]


@pytest.mark.parametrize("archivo", ["semilla_1.pt", "semilla_1.config.json"])
def test_informa_si_el_respaldo_del_individuo_esta_incompleto(montaje, archivo) -> None:
    carpeta, cfg, mejor, comite = montaje
    (comite / archivo).unlink()
    with pytest.raises(FileNotFoundError, match="para recuperar el individuo"):
        evaluate.cargar_individual(carpeta, cfg, mejor)


@pytest.mark.parametrize("alteracion", ["semilla", "mercado", "hiperparametros", "paso"])
def test_rechaza_un_miembro_que_no_corresponde_al_individuo_elegido(montaje, alteracion) -> None:
    carpeta, cfg, mejor, comite = montaje
    if alteracion in ("semilla", "mercado"):
        ruta = comite / "semilla_1.config.json"
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        datos[alteracion] = 0 if alteracion == "semilla" else "brent"
        ruta.write_text(json.dumps(datos), encoding="utf-8")
        mensaje = "configuración"
    else:
        ruta = comite / "semilla_1.pt"
        agente = AgenteDQN.cargar(ruta)
        if alteracion == "hiperparametros":
            agente.cfg.lr *= 2
            mensaje = "hiperparámetros"
        else:
            agente.pasos += 10_000
            mensaje = "checkpoint"
        agente.guardar(ruta)

    with pytest.raises(ValueError, match=mensaje):
        evaluate.cargar_individual(carpeta, cfg, mejor)


def test_la_evaluacion_completa_reproduce_las_metricas_publicadas(montaje, monkeypatch) -> None:
    _, _, _, comite = montaje
    shutil.copytree(RAIZ / "results" / "comite", comite, dirs_exist_ok=True)
    figuras = comite.parent / "figuras"
    monkeypatch.setattr(plots, "FIGURAS", figuras)

    evaluate.evaluar()

    esperado = json.loads((RAIZ / "results/evaluacion_prueba.json").read_text(encoding="utf-8"))
    generado = json.loads((comite.parent / "evaluacion_prueba.json").read_text(encoding="utf-8"))
    assert generado["corrida_elegida_individual"] == esperado["corrida_elegida_individual"]
    assert generado["paso_del_checkpoint"] == esperado["paso_del_checkpoint"]
    assert generado["metricas"] == esperado["metricas"]
    pd.testing.assert_frame_equal(
        pd.read_csv(comite.parent / "bitacora_dqn_prueba.csv"),
        pd.read_csv(RAIZ / "results/bitacora_dqn_prueba.csv"),
        check_exact=True,
    )
    assert (comite.parent / "evaluacion_prueba.csv").exists()
    assert (figuras / "equity_prueba.png").exists()
    assert (figuras / "exposicion_prueba.png").exists()
