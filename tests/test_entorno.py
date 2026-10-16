"""Contabilidad del entorno: costos, patrimonio y recompensas."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features
from environment import EXPOSICIONES, ConfigEntorno, EntornoInversion


def serie_sintetica(n: int = 1200, deriva: float = 0.0004, sigma: float = 0.02) -> pd.DataFrame:
    """Precios artificiales: así los tests no dependen de la red."""
    rng = np.random.default_rng(7)
    r = rng.normal(deriva, sigma, size=n)
    precio = 50.0 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("1990-01-01", periods=n, name="fecha")
    df = pd.DataFrame({"precio": precio}, index=idx)
    df.attrs["primer_paso"] = features.CALENTAMIENTO
    return df


@pytest.fixture(scope="module")
def montaje():
    df = serie_sintetica()
    esc = features.Escalador.ajustar(features.construir(df["precio"]))
    return df, esc


def test_el_costo_se_cobra_sobre_la_rotacion(montaje) -> None:
    df, esc = montaje
    cfg = ConfigEntorno(largo_episodio=4, aleatorizar_inicio=False, exposicion_inicial=0.0)
    env = EntornoInversion(df, esc, cfg)
    env.reset(seed=0)

    # 0 % -> 100 %: rotación 1.0, costo = 10 bps del capital (que vale 1.0).
    _, _, _, _, info = env.step(2)
    assert info["rotacion"] == pytest.approx(1.0)
    assert info["costos"] == pytest.approx(cfg.costo)
    patrimonio_1 = info["patrimonio"]

    # 100 % -> 100 %: mantener no cuesta nada, ni rota.
    _, _, _, _, info = env.step(2)
    assert info["rotacion"] == pytest.approx(1.0)
    assert info["costos"] == pytest.approx(cfg.costo)
    patrimonio_2 = info["patrimonio"]
    assert patrimonio_2 != patrimonio_1  # el mercado sí se movió

    # 100 % -> 50 %: rotación 0.5 sobre el patrimonio vigente, no sobre el
    # capital inicial. El costo acumulado se reporta en unidades de capital.
    _, _, _, _, info = env.step(1)
    assert info["rotacion"] == pytest.approx(1.5)
    assert info["costos"] == pytest.approx(cfg.costo * (1.0 + 0.5 * patrimonio_2))


def test_quedarse_en_efectivo_no_cuesta_ni_rinde(montaje) -> None:
    df, esc = montaje
    cfg = ConfigEntorno(largo_episodio=100, aleatorizar_inicio=False, exposicion_inicial=0.0)
    env = EntornoInversion(df, esc, cfg)
    env.reset(seed=0)
    for _ in range(100):
        _, r, _, truncado, info = env.step(0)
        assert r == pytest.approx(0.0)
        if truncado:
            break
    assert info["patrimonio"] == pytest.approx(1.0)
    assert info["costos"] == pytest.approx(0.0)


def test_exposicion_plena_replica_comprar_y_mantener(montaje) -> None:
    """Con 100 % desde el primer paso, el patrimonio debe igualar la razón de
    precios menos el costo de entrada única."""
    df, esc = montaje
    cfg = ConfigEntorno(largo_episodio=0, aleatorizar_inicio=False, exposicion_inicial=0.0)
    env = EntornoInversion(df, esc, cfg)
    env.reset(seed=0)
    inicio = env.t
    truncado = False
    while not truncado:
        _, _, _, truncado, info = env.step(2)
    fin = env.t

    precio = df["precio"].to_numpy()
    esperado = (precio[fin] / precio[inicio]) * (1.0 - cfg.costo)
    assert info["patrimonio"] == pytest.approx(esperado, rel=1e-9)


def test_la_recompensa_log_suma_al_log_del_patrimonio(montaje) -> None:
    """Propiedad que hace de `log_neto` la recompensa correcta: la suma de
    recompensas del episodio ES el log del patrimonio final."""
    df, esc = montaje
    cfg = ConfigEntorno(
        largo_episodio=0,
        aleatorizar_inicio=False,
        recompensa="log_neto",
        escala_recompensa=1.0,
    )
    env = EntornoInversion(df, esc, cfg)
    env.reset(seed=0)
    rng = np.random.default_rng(3)
    total = 0.0
    truncado = False
    while not truncado:
        _, r, _, truncado, info = env.step(int(rng.integers(3)))
        total += r
    assert total == pytest.approx(np.log(info["patrimonio"]), rel=1e-9)


def test_la_penalizacion_de_drawdown_solo_castiga_el_dano_nuevo(montaje) -> None:
    """Dos barras dentro del mismo pozo: la segunda no debe pagar otra vez si
    el patrimonio no empeoró."""
    df, esc = montaje
    cfg = ConfigEntorno(
        largo_episodio=0,
        aleatorizar_inicio=False,
        recompensa="drawdown",
        escala_recompensa=1.0,
    )
    env = EntornoInversion(df, esc, cfg)
    env.reset(seed=0)
    env.step(2)  # entrar
    # Forzar un pozo y luego quedarse quieto en efectivo.
    env.patrimonio = 0.8
    env.pico = 1.0
    env.drawdown = -0.2
    _, r_quieto, *_ = env.step(0)  # efectivo: r_port = 0 (ya estaba en 0 tras... )
    # Al pasar a efectivo se paga rotación, pero el drawdown no empeora por
    # retorno de mercado; la penalización debe ser prácticamente la del costo.
    assert r_quieto > -0.05, f"penalización repetida del mismo drawdown: {r_quieto}"


def test_el_espacio_de_acciones_es_el_declarado(montaje) -> None:
    df, esc = montaje
    env = EntornoInversion(df, esc, ConfigEntorno())
    assert env.action_space.n == 3
    assert EXPOSICIONES == (0.0, 0.5, 1.0)
    assert env.observation_space.shape == (len(features.NOMBRES) + 1,)
    with pytest.raises(ValueError):
        env.reset(seed=0)
        env.step(3)


def test_mismo_semilla_mismo_episodio(montaje) -> None:
    df, esc = montaje
    cfg = ConfigEntorno(largo_episodio=60, aleatorizar_inicio=True, semilla=11)
    a = EntornoInversion(df, esc, cfg)
    b = EntornoInversion(df, esc, cfg)
    oa, _ = a.reset(seed=11)
    ob, _ = b.reset(seed=11)
    np.testing.assert_array_equal(oa, ob)
    assert a.t == b.t


def test_decidir_cada_k_barras_mantiene_la_exposicion_y_cobra_una_vez(montaje) -> None:
    """Con `frecuencia_decision = 5` el agente decide una vez por semana: la
    exposición se mantiene las cinco barras y el costo se cobra en la primera."""
    df, esc = montaje
    cfg = ConfigEntorno(
        largo_episodio=50,
        aleatorizar_inicio=False,
        frecuencia_decision=5,
        exposicion_inicial=0.0,
    )
    env = EntornoInversion(df, esc, cfg)
    env.reset(seed=0)

    t0 = env.t
    _, _, _, _, info = env.step(2)  # pasar a 100 %
    assert env.t == t0 + 5, "un paso debe abarcar 5 barras"
    assert len(env.sub_barras) == 5
    assert info["costos"] == pytest.approx(cfg.costo)

    # Repetir la misma acción no cuesta nada aunque pasen otras 5 barras.
    _, _, _, _, info = env.step(2)
    assert info["costos"] == pytest.approx(cfg.costo)
    assert info["rotacion"] == pytest.approx(1.0)


def test_la_bitacora_tiene_una_fila_por_barra_no_por_decision(montaje) -> None:
    """Las métricas se anualizan con 252 barras: si la bitácora tuviera una
    fila por decisión semanal, todas las métricas de riesgo saldrían mal."""
    from environment import recorrido_completo, rodar

    df, esc = montaje
    cfg_diario = ConfigEntorno(largo_episodio=0, aleatorizar_inicio=False, frecuencia_decision=1)
    cfg_semanal = ConfigEntorno(largo_episodio=0, aleatorizar_inicio=False, frecuencia_decision=5)

    b_diario = rodar(recorrido_completo(df, esc, cfg_diario), lambda _: 2)
    b_semanal = rodar(recorrido_completo(df, esc, cfg_semanal), lambda _: 2)

    assert abs(len(b_diario) - len(b_semanal)) <= 5
    # Comprar y mantener da lo mismo decida cuando decida: nunca cambia nada.
    assert b_semanal["patrimonio"].iloc[-1] == pytest.approx(
        b_diario["patrimonio"].iloc[-1], rel=1e-9
    )
    # Y la suma de recompensas sigue siendo el log del patrimonio final.
    cfg_semanal.escala_recompensa = 1.0
    b = rodar(recorrido_completo(df, esc, cfg_semanal), lambda _: 2)
    assert b["recompensa"].sum() == pytest.approx(np.log(b["patrimonio"].iloc[-1]), rel=1e-9)
