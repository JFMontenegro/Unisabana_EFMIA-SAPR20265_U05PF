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
import train
from agent import AgenteDQN

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture
def montaje(tmp_path, monkeypatch):
    resultados = tmp_path / "results"
    comite = resultados / "comite"
    comite.mkdir(parents=True)
    for nombre in ("semilla_1.pt", "semilla_1.config.json"):
        shutil.copy2(RAIZ / "results" / "comite" / nombre, comite / nombre)
    carpeta = resultados / "corridas" / "f7_final_s1"
    carpeta.mkdir(parents=True)
    original = RAIZ / "results" / "corridas" / carpeta.name
    cfg = json.loads((original / "config.json").read_text(encoding="utf-8"))
    mejor = json.loads((original / "mejor.json").read_text(encoding="utf-8"))
    shutil.copy2(original / "config.json", carpeta / "config.json")
    shutil.copy2(original / "mejor.json", carpeta / "mejor.json")
    monkeypatch.setattr(train, "RAIZ", tmp_path)
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


@pytest.mark.parametrize("mercado", ["btc", "brent"])
def test_la_evaluacion_completa_reproduce_las_metricas_publicadas(
    tmp_path, monkeypatch, mercado
) -> None:
    carpeta_resultados = "results" if mercado == "btc" else "results_brent"
    originales = RAIZ / carpeta_resultados
    resultados = tmp_path / carpeta_resultados
    shutil.copytree(originales / "comite", resultados / "comite")
    for corrida in originales.glob("corridas/f7_final_s*"):
        destino = resultados / "corridas" / corrida.name
        destino.mkdir(parents=True)
        for archivo in ("config.json", "mejor.json"):
            shutil.copy2(corrida / archivo, destino / archivo)
    otro_mercado = tmp_path / ("results_brent" if mercado == "btc" else "results")
    otro_mercado.mkdir()
    testigo = otro_mercado / "sin_modificar.txt"
    testigo.write_text("resultados del otro mercado", encoding="utf-8")
    figuras = tmp_path / "figuras"
    figuras.mkdir()
    sufijo_figura = "prueba" if mercado == "btc" else "brent_prueba"
    figura_ajena = figuras / (
        "equity_brent_prueba.png" if mercado == "btc" else "equity_prueba.png"
    )
    figura_ajena.write_bytes(b"figura del otro mercado")
    monkeypatch.setattr(train, "RAIZ", tmp_path)
    monkeypatch.setattr(plots, "FIGURAS", figuras)

    evaluate.evaluar(mercado=mercado)

    esperado = json.loads((originales / "evaluacion_prueba.json").read_text(encoding="utf-8"))
    generado = json.loads((resultados / "evaluacion_prueba.json").read_text(encoding="utf-8"))
    assert generado["mercado"] == mercado
    assert generado["corrida_elegida_individual"] == esperado["corrida_elegida_individual"]
    assert generado["paso_del_checkpoint"] == esperado["paso_del_checkpoint"]
    assert generado["metricas"] == esperado["metricas"]
    pd.testing.assert_frame_equal(
        pd.read_csv(resultados / "bitacora_dqn_prueba.csv"),
        pd.read_csv(originales / "bitacora_dqn_prueba.csv"),
        check_exact=True,
    )
    assert (resultados / "evaluacion_prueba.csv").exists()
    assert (figuras / f"equity_{sufijo_figura}.png").exists()
    assert (figuras / f"exposicion_{sufijo_figura}.png").exists()
    assert list(otro_mercado.iterdir()) == [testigo]
    assert testigo.read_text(encoding="utf-8") == "resultados del otro mercado"
    assert figura_ajena.read_bytes() == b"figura del otro mercado"


def test_no_usa_los_modelos_de_btc_si_faltan_los_de_brent(montaje) -> None:
    with pytest.raises(FileNotFoundError, match="results_brent"):
        evaluate.evaluar(mercado="brent")


def test_rechaza_una_corrida_de_otro_mercado(montaje) -> None:
    carpeta, cfg, _, _ = montaje
    cfg["mercado"] = "brent"
    (carpeta / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="pertenece al mercado brent"):
        evaluate.evaluar()


@pytest.mark.parametrize("mercado", ["btc", "brent"])
def test_carga_el_comite_y_selecciona_el_individuo_desde_las_corridas_del_mercado(
    tmp_path, monkeypatch, mercado
) -> None:
    monkeypatch.setattr(train, "RAIZ", tmp_path)
    for resultados in ("results", "results_brent"):
        original = RAIZ / resultados
        destino = tmp_path / resultados / "corridas" / "f7_final_s1"
        destino.mkdir(parents=True)
        shutil.copy2(original / "comite/semilla_1.pt", destino / "mejor.pt")
        shutil.copy2(original / "corridas/f7_final_s1/mejor.json", destino / "mejor.json")

    elegido, _ = evaluate.seleccionar("f7_final_s*", mercado=mercado)
    comite, miembros = evaluate.cargar_comite("f7_final_s*", mercado=mercado)
    resultados = "results" if mercado == "btc" else "results_brent"
    assert elegido == tmp_path / resultados / "corridas" / "f7_final_s1"
    assert miembros == ["f7_final_s1"]
    esperado = AgenteDQN.cargar(RAIZ / resultados / "comite/semilla_1.pt")
    for clave, tensor in esperado.q.state_dict().items():
        assert torch.equal(comite.agentes[0].q.state_dict()[clave], tensor)
