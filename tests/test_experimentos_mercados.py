"""Las fases, las decisiones y los reportes usan un único mercado por ejecución."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

import experiments
import pipeline
import plots
import report
import train

RAIZ = Path(__file__).resolve().parents[1]
FASES = (
    ("fase1_recompensas", {}),
    ("fase2_bootstrap", {}),
    ("fase3_deriva", {}),
    ("fase4_capacidad", {"recompensa": "log_neto"}),
    ("fase5_final", {"recompensa": "log_neto", "agente": {"oculta": 8}}),
    ("fase6_frecuencia", {"recompensa": "log_neto", "agente": {"oculta": 8}}),
    ("fase7_final", {"recompensa": "log_neto", "agente": {"oculta": 8}, "frecuencia": 5}),
)


@pytest.mark.parametrize("mercado", ["btc", "brent"])
@pytest.mark.parametrize(("fase", "parametros"), FASES)
def test_cada_fase_envia_el_mercado_al_entrenador_y_separa_su_resumen(
    tmp_path, monkeypatch, mercado, fase, parametros
) -> None:
    monkeypatch.setattr(train, "RAIZ", tmp_path)
    monkeypatch.setattr(experiments, "PROCESOS", 1)
    recibidos = []

    def entrenar(**kwargs):
        recibidos.append(kwargs)
        return {"etiqueta": kwargs["etiqueta"], "mercado": kwargs["mercado"]}

    monkeypatch.setattr(experiments, "entrenar", entrenar)
    resultado = getattr(experiments, fase)(pasos=4, semillas=(0,), mercado=mercado, **parametros)
    assert recibidos
    assert all(t["mercado"] == mercado and t["pasos"] == 4 for t in recibidos)
    assert all(t["cfg_agente"].semilla == t["semilla"] == 0 for t in recibidos)
    if fase == "fase6_frecuencia":
        frecuencias = {t["cfg_entorno"].frecuencia_decision for t in recibidos}
        assert frecuencias == ({1, 24, 168} if mercado == "btc" else {1, 5, 21})
    carpeta = "results" if mercado == "btc" else "results_brent"
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / carpeta / f"{fase}.csv"), resultado)
    assert not (tmp_path / ("results_brent" if mercado == "btc" else "results")).exists()


def _copiar_corridas(tmp_path, monkeypatch, mercado):
    carpeta = "results" if mercado == "btc" else "results_brent"
    fuente = RAIZ / carpeta
    destino = tmp_path / carpeta
    shutil.copytree(
        fuente / "corridas",
        destino / "corridas",
        ignore=shutil.ignore_patterns("*.pt", "*.zip", "*.npz"),
    )
    otro = tmp_path / ("results_brent" if mercado == "btc" else "results")
    otro.mkdir()
    testigo = otro / "sin_modificar.txt"
    testigo.write_text("otro mercado", encoding="utf-8")
    monkeypatch.setattr(train, "RAIZ", tmp_path)
    monkeypatch.setattr(plots, "FIGURAS", tmp_path / "figuras")
    return fuente, destino, testigo


@pytest.mark.parametrize("mercado", ["btc", "brent"])
def test_pipeline_reproduce_las_decisiones_de_validacion_del_mercado(
    tmp_path, monkeypatch, mercado
) -> None:
    fuente, destino, testigo = _copiar_corridas(tmp_path, monkeypatch, mercado)
    llamadas = []

    def fase_ya_entrenada(*args, **kwargs):
        llamadas.append(kwargs)

    for fase, _ in FASES[2:]:
        monkeypatch.setattr(experiments, fase, fase_ya_entrenada)
    esperado = json.loads((fuente / "decisiones.json").read_text(encoding="utf-8"))

    actual = pipeline.correr(pasos=esperado["pasos_por_corrida"], mercado=mercado)

    assert len(llamadas) == 5
    assert all(llamada["mercado"] == mercado for llamada in llamadas)
    for fase in ("fase3", "fase4", "fase6"):
        assert actual[fase]["elegido"] == esperado[fase]["elegido"]
    assert actual["fase7"]["config"] == esperado["fase7"]["config"]
    assert actual["fase7"]["frecuencia_decision"] == esperado["fase7"]["frecuencia_decision"]
    assert actual["fase7"]["val_retorno"] == esperado["fase7"]["val_retorno"]
    guardado = json.loads((destino / "decisiones.json").read_text(encoding="utf-8"))
    assert guardado["mercado"] == mercado
    assert guardado["fase7"] == actual["fase7"]
    assert list(testigo.parent.iterdir()) == [testigo]


@pytest.mark.parametrize("mercado", ["btc", "brent"])
def test_reporte_y_tabla_de_validacion_leen_y_escriben_en_su_mercado(
    tmp_path, monkeypatch, mercado
) -> None:
    fuente, destino, testigo = _copiar_corridas(tmp_path, monkeypatch, mercado)
    figuras = tmp_path / "figuras"
    figuras.mkdir()
    ajena = figuras / ("recompensas_brent_f3.png" if mercado == "btc" else "recompensas_f3.png")
    ajena.write_bytes(b"figura del otro mercado")
    monkeypatch.setattr(sys, "argv", ["report.py", "--mercado", mercado, "--fases", "f3_*"])

    report.main()

    generado = pd.read_csv(destino / "resumen_experimentos.csv", float_precision="round_trip")
    tabla = experiments.tabla_validacion("f3_*", mercado=mercado)
    curvas = report.curvas("f3_*", mercado=mercado)
    esperados = {
        p.parent.name: json.loads(p.read_text(encoding="utf-8"))["retorno_acumulado"]
        for p in fuente.glob("corridas/f3_*/mejor.json")
    }
    assert dict(zip(generado["corrida"], generado["val_retorno"], strict=True)) == esperados
    assert dict(zip(tabla["corrida"], tabla["val_retorno"], strict=True)) == esperados
    assert set(curvas) == set(esperados)
    sufijo = "f3" if mercado == "btc" else "brent_f3"
    assert (figuras / f"recompensas_{sufijo}.png").exists()
    brecha = "brecha_memorizacion.png" if mercado == "btc" else "brecha_memorizacion_brent.png"
    assert (figuras / brecha).exists()
    assert ajena.read_bytes() == b"figura del otro mercado"
    assert list(testigo.parent.iterdir()) == [testigo]
