"""La evaluación tabulada tiene que ser la misma política, no una parecida.

Es una optimización de dos órdenes de magnitud sobre la ruta crítica del
proyecto (cada corrida evalúa 15 veces sobre decenas de miles de barras). Una
optimización que cambia la política en silencio invalidaría todos los números,
así que se fija la equivalencia exacta.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features
from agent import AgenteDQN, Config
from environment import (
    EXPOSICIONES,
    ConfigEntorno,
    politica_tabulada,
    recorrido_completo,
    rodar,
)
from test_entorno import serie_sintetica


@pytest.fixture(scope="module")
def montaje():
    df = serie_sintetica(n=1800)
    esc = features.Escalador.ajustar(features.construir(df["precio"]))
    agente = AgenteDQN(len(features.NOMBRES) + 1, 3, Config(semilla=5))
    return df, esc, agente


@pytest.mark.parametrize("frecuencia", [1, 5])
def test_la_tabla_q_coincide_con_la_evaluacion_barra_a_barra(montaje, frecuencia: int) -> None:
    df, esc, agente = montaje
    cfg = ConfigEntorno(
        largo_episodio=0, aleatorizar_inicio=False, frecuencia_decision=frecuencia
    )

    env_a = recorrido_completo(df, esc, cfg)
    lenta = rodar(env_a, agente.politica_greedy())

    env_b = recorrido_completo(df, esc, cfg)
    rapida = rodar(env_b, politica_tabulada(env_b, agente))

    np.testing.assert_array_equal(
        lenta["accion"].to_numpy(),
        rapida["accion"].to_numpy(),
        err_msg="la política tabulada eligió acciones distintas",
    )
    np.testing.assert_allclose(
        lenta["patrimonio"].to_numpy(), rapida["patrimonio"].to_numpy(), rtol=1e-12
    )


def test_los_valores_de_la_tabla_son_los_de_la_red(montaje) -> None:
    df, esc, agente = montaje
    env = recorrido_completo(df, esc, ConfigEntorno(largo_episodio=0))
    tabla = agente.tabla_q(env.estados, EXPOSICIONES)
    assert tabla.shape == (len(env.estados), len(EXPOSICIONES), 3)

    rng = np.random.default_rng(0)
    for t in rng.integers(env.primer_paso, env.ultimo_paso, size=20):
        for i, e in enumerate(EXPOSICIONES):
            obs = np.concatenate([env.estados[t], [e - 0.5]]).astype(np.float32)
            # No es igualdad bit a bit: en float32 el orden de acumulación de
            # un lote de 3n filas no es el de un lote de 1, y la diferencia
            # relativa llega a ~2e-5.
            np.testing.assert_allclose(tabla[t, i], agente.valores_q(obs), rtol=1e-4)


def test_el_margen_entre_acciones_supera_el_error_numerico(montaje) -> None:
    """La preocupación legítima de la optimización anterior.

    Si los valores Q de dos acciones estuvieran más cerca entre sí que el error
    de redondeo en float32 (~2e-5 relativo), el `argmax` podría volcarse y la
    política tabulada dejaría de ser la misma. Se mide el margen y se exige que
    las barras en riesgo sean una fracción despreciable.

    El margen real de los agentes entrenados de este proyecto es del orden del
    1 % (README §7.4): pequeño para la estabilidad del aprendizaje, pero mil
    veces mayor que el error numérico.
    """
    df, esc, agente = montaje
    env = recorrido_completo(df, esc, ConfigEntorno(largo_episodio=0))
    tabla = agente.tabla_q(env.estados, EXPOSICIONES)[env.primer_paso : env.ultimo_paso]

    ordenados = np.sort(tabla, axis=-1)
    margen = ordenados[..., -1] - ordenados[..., -2]
    escala = np.abs(ordenados[..., -1]) + 1e-12
    en_riesgo = float((margen / escala < 1e-4).mean())
    assert en_riesgo < 0.01, f"{100 * en_riesgo:.2f} % de las barras con margen numérico"


def test_la_tabla_es_mas_rapida(montaje) -> None:
    """No es un test de rendimiento estricto —el tiempo depende de la máquina—
    sino una guarda contra una regresión que devuelva la ruta lenta."""
    df, esc, agente = montaje
    cfg = ConfigEntorno(largo_episodio=0, aleatorizar_inicio=False)

    t0 = time.perf_counter()
    rodar(recorrido_completo(df, esc, cfg), agente.politica_greedy())
    lenta = time.perf_counter() - t0

    env = recorrido_completo(df, esc, cfg)
    t0 = time.perf_counter()
    rodar(env, politica_tabulada(env, agente))
    rapida = time.perf_counter() - t0

    assert rapida < lenta, f"tabulada {rapida:.3f}s vs barra a barra {lenta:.3f}s"


def test_el_entorno_de_evaluacion_conserva_la_frecuencia_de_decision(montaje) -> None:
    """Regresión de un error real.

    `recorrido_completo` enumeraba a mano los campos que copiaba de la
    configuración de entrenamiento y se dejó fuera `frecuencia_decision`. Como
    consecuencia, TODOS los agentes se evaluaban decidiendo en cada barra, sin
    importar con qué frecuencia se hubieran entrenado — y la fase que comparaba
    frecuencias medía siempre lo mismo. No lanzaba ninguna excepción y los
    números parecían razonables; se detectó porque un agente «semanal» reportó
    2 136 operaciones al año, cuando el máximo posible es 52.
    """
    df, esc, _ = montaje
    for frecuencia in (1, 5, 24, 168):
        cfg = ConfigEntorno(
            costo=0.002,
            recompensa="drawdown",
            lambda_drawdown=7.0,
            frecuencia_decision=frecuencia,
            largo_episodio=360,
            aleatorizar_inicio=True,
            escala_recompensa=50.0,
        )
        env = recorrido_completo(df, esc, cfg)
        assert env.cfg.frecuencia_decision == frecuencia
        # Y el resto de la configuración sobrevive igual, salvo el recorrido.
        assert env.cfg.costo == cfg.costo
        assert env.cfg.recompensa == cfg.recompensa
        assert env.cfg.lambda_drawdown == cfg.lambda_drawdown
        assert env.cfg.escala_recompensa == cfg.escala_recompensa
        assert env.cfg.largo_episodio == 0
        assert env.cfg.aleatorizar_inicio is False


def test_la_rotacion_esta_acotada_por_la_frecuencia(montaje) -> None:
    """Una política que cambia en cada decisión no puede rotar más de una vez
    por decisión. Es la cota que delató el error anterior."""
    df, esc, _ = montaje
    rng = np.random.default_rng(0)
    for frecuencia in (1, 24, 168):
        cfg = ConfigEntorno(
            largo_episodio=0, aleatorizar_inicio=False, frecuencia_decision=frecuencia
        )
        env = recorrido_completo(df, esc, cfg)
        bitacora = rodar(env, lambda _obs: int(rng.integers(3)))
        decisiones = np.ceil(len(bitacora) / frecuencia)
        assert bitacora["rotacion"].iloc[-1] <= decisiones, (
            f"frecuencia {frecuencia}: rotación {bitacora['rotacion'].iloc[-1]:.1f} "
            f"sobre {decisiones:.0f} decisiones posibles"
        )
