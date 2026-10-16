"""Las líneas base tienen que ser lo que dicen ser.

Una línea base mal implementada es peor que no tenerla: inflan artificialmente
el mérito del agente y el error no se nota en ninguna gráfica.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import baselines
import features
import metrics
from environment import ConfigEntorno, recorrido_completo, rodar
from test_entorno import serie_sintetica


@pytest.fixture(scope="module")
def montaje():
    df = serie_sintetica(n=1500)
    esc = features.Escalador.ajustar(features.construir(df["precio"]))
    return df, esc, ConfigEntorno(largo_episodio=0, aleatorizar_inicio=False)


def test_comprar_y_mantener_iguala_la_razon_de_precios(montaje) -> None:
    df, esc, cfg = montaje
    env = recorrido_completo(df, esc, cfg)
    bitacora = rodar(env, baselines.comprar_y_mantener)

    precio = df["precio"].to_numpy()
    esperado = precio[env.t] / precio[env.primer_paso] * (1.0 - cfg.costo)
    assert bitacora["patrimonio"].iloc[-1] == pytest.approx(esperado, rel=1e-9)
    # Un solo costo de entrada en toda la serie.
    assert bitacora["rotacion"].iloc[-1] == pytest.approx(1.0)


def test_el_cruce_de_medias_replica_el_calculo_directo(montaje) -> None:
    """La política lee indicadores normalizados; la verdad se calcula sobre los
    precios. Deben coincidir salvo en las barras recortadas a ±5 sigmas."""
    df, esc, cfg = montaje
    env = recorrido_completo(df, esc, cfg)
    bitacora = rodar(env, baselines.cruce_de_medias(esc))

    logp = np.log(df["precio"])
    sma21 = logp.rolling(21).mean()
    sma63 = logp.rolling(63).mean()
    verdad = (sma21 > sma63).astype(int) * 2  # 2 = 100 % invertido, 0 = efectivo
    verdad = verdad.reindex(bitacora.index)

    coincide = (bitacora["accion"].to_numpy() == verdad.to_numpy()).mean()
    assert coincide > 0.995, f"la regla coincide solo en {coincide:.3%} de las barras"


def test_la_politica_aleatoria_pierde_dinero_en_costos(montaje) -> None:
    """Con 10 bps y rotación esperada ~0.67 por barra, operar al azar es una
    máquina de quemar capital. Si esto no se cumple, el costo no se está
    cobrando."""
    df, esc, cfg = montaje
    _, apiladas = baselines.correr_todas(df, esc, cfg, semillas_aleatorias=5)
    resumen = baselines.resumir_aleatorias(apiladas)

    assert resumen["rotacion_anual"] > 100, resumen["rotacion_anual"]
    assert resumen["costos_totales"] > 0.05, resumen["costos_totales"]


def test_el_efectivo_no_mueve_el_patrimonio(montaje) -> None:
    df, esc, cfg = montaje
    env = recorrido_completo(df, esc, cfg)
    bitacora = rodar(env, baselines.efectivo)
    assert bitacora["patrimonio"].iloc[-1] == pytest.approx(1.0)
    r = metrics.resumir(bitacora)
    assert r["max_drawdown"] == pytest.approx(0.0)
    assert r["sharpe"] == 0.0


def test_las_metricas_se_calculan_sobre_la_serie_correcta(montaje) -> None:
    """Prueba de la métrica, no de la estrategia: un patrimonio conocido debe
    producir un drawdown conocido."""
    idx = pd.bdate_range("2000-01-03", periods=5, name="fecha")
    bitacora = pd.DataFrame(
        {
            "patrimonio": [1.10, 1.00, 0.88, 0.95, 1.20],
            "drawdown": [0.0, 0.0, 0.0, 0.0, 0.0],
            "costos": [0.0, 0.0, 0.0, 0.0, 0.001],
            "rotacion": [1.0, 1.0, 1.0, 1.0, 1.0],
            "exposicion": [1.0, 1.0, 1.0, 1.0, 1.0],
            "accion": [2, 2, 2, 2, 2],
            "ret_activo": [0.1, -0.09, -0.12, 0.08, 0.26],
            "recompensa": [0.0] * 5,
        },
        index=idx,
    )
    r = metrics.resumir(bitacora)
    assert r["retorno_acumulado"] == pytest.approx(0.20)
    # Pico 1.10 -> fondo 0.88  =>  −20 %
    assert r["max_drawdown"] == pytest.approx(0.88 / 1.10 - 1.0)
    assert r["costos_totales"] == pytest.approx(0.001)
