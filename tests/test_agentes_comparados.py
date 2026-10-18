"""PPO y Q-Learning tabular comparten la interfaz del DQN, o la comparación miente.

Los tres agentes se evalúan con el mismo `environment.politica_tabulada` y el
mismo código de contabilidad. Si la tabla de alguno no representara de verdad
su política, la comparación del informe estaría midiendo artefactos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features
from environment import (
    EXPOSICIONES,
    ConfigEntorno,
    EntornoInversion,
    politica_tabulada,
    recorrido_completo,
    rodar,
)
from tabular import INDICADORES, AgenteTabular, ComiteTabular, ConfigTabular
from test_entorno import serie_sintetica


@pytest.fixture(scope="module")
def requiere_sb3():
    pytest.importorskip(
        "stable_baselines3",
        reason="PPO requiere el extra opcional: uv sync --extra dev --extra ppo",
    )


@pytest.fixture(scope="module")
def montaje():
    df = serie_sintetica(n=2000)
    esc = features.Escalador.ajustar(features.construir(df["precio"]))
    return df, esc


def test_el_entorno_pasa_el_verificador_de_stable_baselines3(montaje, requiere_sb3) -> None:
    """El entorno es un Gymnasium válido para un framework externo, no solo
    para nuestro propio código."""
    from stable_baselines3.common.env_checker import check_env

    df, esc = montaje
    env = EntornoInversion(df, esc, ConfigEntorno(largo_episodio=360))
    check_env(env, warn=True)  # lanza si algo no cumple la API


def test_la_tabla_de_ppo_da_la_misma_accion_que_predict(montaje, requiere_sb3) -> None:
    """`tabla_q` de PPO devuelve probabilidades; su argmax tiene que coincidir
    con `model.predict(deterministic=True)`, que es la política determinista
    oficial de SB3. Si no coincidieran, el comité de PPO no sería PPO."""
    from ppo import AgentePPO, ConfigPPO

    df, esc = montaje
    env = EntornoInversion(df, esc, ConfigEntorno(largo_episodio=360))
    agente = AgentePPO.nuevo(env, ConfigPPO(n_steps=180, lote=60, semilla=0))

    tabla = agente.tabla_q(env.estados, EXPOSICIONES)
    rng = np.random.default_rng(0)
    for t in rng.integers(env.primer_paso, env.ultimo_paso, size=30):
        for i, e in enumerate(EXPOSICIONES):
            obs = np.concatenate([env.estados[t], [e - 0.5]]).astype(np.float32)
            esperada, _ = agente.modelo.predict(obs, deterministic=True)
            assert int(np.argmax(tabla[t, i])) == int(esperada)


def test_las_probabilidades_de_ppo_suman_uno(montaje, requiere_sb3) -> None:
    from ppo import AgentePPO, ConfigPPO

    df, esc = montaje
    env = EntornoInversion(df, esc, ConfigEntorno(largo_episodio=360))
    agente = AgentePPO.nuevo(env, ConfigPPO(n_steps=180, lote=60, semilla=1))
    tabla = agente.tabla_q(env.estados, EXPOSICIONES)[env.primer_paso : env.ultimo_paso]
    np.testing.assert_allclose(tabla.sum(axis=-1), 1.0, rtol=1e-5)


def test_las_barras_de_calentamiento_quedan_marcadas_no_inventadas(montaje, requiere_sb3) -> None:
    """Antes del primer paso operable los indicadores tienen NaN. La tabla debe
    devolver NaN ahí —no una acción cualquiera— y la política debe negarse a
    decidir si alguna vez se la llamara sobre una de esas barras."""
    from ppo import AgentePPO, ConfigPPO

    df, esc = montaje
    env = EntornoInversion(df, esc, ConfigEntorno(largo_episodio=360))
    agente = AgentePPO.nuevo(env, ConfigPPO(n_steps=180, lote=60, semilla=2))
    tabla = agente.tabla_q(env.estados, EXPOSICIONES)

    assert np.isnan(tabla[0]).all(), "la primera barra debería estar marcada como NaN"
    assert np.isfinite(tabla[env.primer_paso]).all()

    politica = politica_tabulada(env, agente)
    env.reset(seed=0)
    env.t = 0
    with pytest.raises(ValueError, match="estado incompleto"):
        politica(np.zeros(env.observation_space.shape, dtype=np.float32))


# --------------------------------------------------------------------- tabular


def test_el_tabular_discretiza_con_cuantiles_de_entrenamiento(montaje) -> None:
    df, esc = montaje
    estados = esc.aplicar(features.construir(df["precio"]))[features.DIARIO.calentamiento :]
    bordes = AgenteTabular.bordes_de_entrenamiento(estados, cortes=4)

    assert bordes.shape == (len(INDICADORES), 3)
    # Los cortes son cuantiles: cada banda debe llevarse ~1/4 de las barras.
    for k, nombre in enumerate(INDICADORES):
        col = estados[:, features.NOMBRES.index(nombre)]
        celdas = np.digitize(col, bordes[k])
        proporciones = np.bincount(celdas, minlength=4) / len(col)
        assert proporciones.min() > 0.2, (nombre, proporciones)


def test_la_tabla_del_tabular_coincide_con_su_politica(montaje) -> None:
    """Igual que con PPO: `tabla_q` tiene que reproducir lo que elige `actuar`."""
    df, esc = montaje
    estados = esc.aplicar(features.construir(df["precio"]))
    finitas = estados[np.isfinite(estados).all(axis=1)]
    bordes = AgenteTabular.bordes_de_entrenamiento(finitas, cortes=4)
    agente = AgenteTabular(bordes, cfg=ConfigTabular(cortes=4, semilla=0))
    # Llenar la tabla con valores distintos para que el argmax no sea trivial.
    agente.q = np.random.default_rng(0).normal(size=agente.q.shape)

    tabla = agente.tabla_q(finitas, EXPOSICIONES)
    rng = np.random.default_rng(1)
    for t in rng.integers(0, len(finitas), size=40):
        for i, e in enumerate(EXPOSICIONES):
            obs = np.concatenate([finitas[t], [e - 0.5]])
            assert int(np.argmax(tabla[t, i])) == agente.actuar(obs, deterministico=True)


def test_el_tabular_aprende_una_recompensa_trivial() -> None:
    """Prueba de cordura del aprendizaje: si una acción siempre paga más, la
    tabla tiene que acabar prefiriéndola."""
    bordes = np.zeros((len(INDICADORES), 3))
    agente = AgenteTabular(bordes, cfg=ConfigTabular(cortes=4, lr=0.5, semilla=0))
    obs = np.zeros(len(features.NOMBRES) + 1)
    for _ in range(200):
        agente.aprender(obs, 2, 1.0, obs, True)
        agente.aprender(obs, 0, -1.0, obs, True)
    assert agente.actuar(obs, deterministico=True) == 2


def test_el_comite_tabular_promedia_las_tablas(montaje) -> None:
    df, esc = montaje
    estados = esc.aplicar(features.construir(df["precio"]))
    finitas = estados[np.isfinite(estados).all(axis=1)]
    bordes = AgenteTabular.bordes_de_entrenamiento(finitas, cortes=4)

    agentes = []
    for s in range(3):
        a = AgenteTabular(bordes, cfg=ConfigTabular(cortes=4, semilla=s))
        a.q = np.random.default_rng(s).normal(size=a.q.shape)
        agentes.append(a)

    comite = ComiteTabular(agentes)
    esperado = np.mean([a.tabla_q(finitas, EXPOSICIONES) for a in agentes], axis=0)
    np.testing.assert_allclose(comite.tabla_q(finitas, EXPOSICIONES), esperado)


@pytest.mark.parametrize("algoritmo", ["tabular", "ppo"])
def test_los_tres_agentes_se_ruedan_con_el_mismo_codigo(montaje, algoritmo, request) -> None:
    """La prueba que hace comparable la tabla del informe: el mismo `rodar`,
    la misma contabilidad, para un tabular y para PPO."""
    df, esc = montaje
    cfg = ConfigEntorno(largo_episodio=0, aleatorizar_inicio=False)

    if algoritmo == "tabular":
        estados = esc.aplicar(features.construir(df["precio"]))
        finitas = estados[np.isfinite(estados).all(axis=1)]
        bordes = AgenteTabular.bordes_de_entrenamiento(finitas, cortes=4)
        agente = AgenteTabular(bordes, cfg=ConfigTabular(cortes=4, semilla=0))
    else:
        request.getfixturevalue("requiere_sb3")
        from ppo import AgentePPO, ConfigPPO

        env_base = EntornoInversion(df, esc, ConfigEntorno(largo_episodio=360))
        agente = AgentePPO.nuevo(env_base, ConfigPPO(n_steps=180, lote=60, semilla=0))

    env = recorrido_completo(df, esc, cfg)
    bitacora = rodar(env, politica_tabulada(env, agente))
    assert len(bitacora) > 100
    assert np.isfinite(bitacora["patrimonio"]).all()
    assert bitacora["patrimonio"].iloc[-1] > 0
