"""Compatibilidad de los escaladores guardados y conservación de sus estados."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import data
import features

RAIZ = Path(__file__).resolve().parents[1]


def _guardado(mercado: str, semilla: int = 0) -> dict:
    resultados = "results" if mercado == "btc" else "results_brent"
    archivo = RAIZ / resultados / "comite" / f"semilla_{semilla}.config.json"
    return json.loads(archivo.read_text(encoding="utf-8"))["escalador"]


def _entrenamiento(mercado: str):
    m = data.cargar(mercado)
    precios = data.partir(m.precios, mercado, "entrenamiento")
    indicadores = features.construir(precios["precio"], m.ventanas)
    referencia = features.Escalador.ajustar(indicadores, m.ventanas)
    return indicadores, referencia


@pytest.mark.parametrize("mercado", ["btc", "brent"])
def test_los_escaladores_guardados_conservan_los_estados_de_entrenamiento(mercado) -> None:
    indicadores, referencia = _entrenamiento(mercado)
    esperado = referencia.aplicar(indicadores)
    for semilla in range(5):
        guardado = _guardado(mercado, semilla)
        original = copy.deepcopy(guardado)
        escalador = features.Escalador.de_dict(guardado)

        assert escalador.nombres == referencia.nombres
        assert escalador.ventanas == referencia.ventanas
        np.testing.assert_array_equal(escalador.medias, guardado["medias"])
        np.testing.assert_array_equal(escalador.desviaciones, guardado["desviaciones"])
        np.testing.assert_array_equal(escalador.aplicar(indicadores), esperado)
        assert guardado == original

        recargado = features.Escalador.de_dict(escalador.a_dict())
        np.testing.assert_array_equal(recargado.aplicar(indicadores), esperado)
        if mercado == "btc":
            assert escalador.a_dict() == guardado


def test_los_nombres_antiguos_conservan_el_orden_de_sus_estadisticas() -> None:
    indicadores, referencia = _entrenamiento("brent")
    guardado = _guardado("brent")
    orden = [4, 0, 8, 1, 9, 2, 3, 5, 7, 6]
    for campo in ("nombres", "medias", "desviaciones"):
        guardado[campo] = [guardado[campo][i] for i in orden]

    escalador = features.Escalador.de_dict(guardado)
    assert escalador.nombres == tuple(features.NOMBRES[i] for i in orden)
    np.testing.assert_array_equal(escalador.medias, guardado["medias"])
    np.testing.assert_array_equal(escalador.desviaciones, guardado["desviaciones"])
    np.testing.assert_array_equal(
        escalador.aplicar(indicadores), referencia.aplicar(indicadores)[:, orden]
    )


def test_los_nombres_antiguos_diarios_no_se_interpretan_como_horarios() -> None:
    guardado = _guardado("brent")
    guardado["ventanas"] = "horario"
    with pytest.raises(ValueError, match="diarias"):
        features.Escalador.de_dict(guardado)
