"""El agente y, sobre todo, el manejo de la truncación."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features
from agent import AgenteDQN, Config, Replay
from environment import ConfigEntorno, EntornoInversion
from test_entorno import serie_sintetica


def test_la_truncacion_bootstrapea_con_el_estado_siguiente_real() -> None:
    """El error que un compañero encontró en el proyecto de CarRacing: al
    truncar, guardar `s` en lugar de `s'`. Aquí truncan TODOS los episodios,
    así que el error afectaría a cada transición final."""
    df = serie_sintetica(n=700)
    esc = features.Escalador.ajustar(features.construir(df["precio"]))
    env = EntornoInversion(df, esc, ConfigEntorno(largo_episodio=20, aleatorizar_inicio=False))

    obs, _ = env.reset(seed=0)
    truncado = False
    pasos = 0
    while not truncado:
        anterior = obs.copy()
        indice_anterior = env.t
        obs, _, terminado, truncado, _ = env.step(2)
        pasos += 1
        # Nunca hay terminación: el mercado no tiene estado absorbente.
        assert terminado is False
        if truncado:
            # El estado siguiente tiene que ser el de la barra siguiente, no el
            # de la barra actual.
            assert env.t == indice_anterior + 1
            assert not np.array_equal(obs, anterior), (
                "al truncar se devolvió el estado anterior; Bellman "
                "bootstrapearía desde el estado equivocado"
            )
    assert pasos == 20


def test_el_objetivo_de_bellman_bootstrapea_cuando_no_hay_terminacion() -> None:
    """Con `terminado=False` el objetivo debe ser r + gamma·maxQ, no r."""
    cfg = Config(gamma=0.9, calentamiento=4, lote=4, semilla=0)
    agente = AgenteDQN(dim_estado=3, n_acciones=3, cfg=cfg)

    s = np.zeros(3, dtype=np.float32)
    s2 = np.ones(3, dtype=np.float32)
    for _ in range(8):
        agente.recordar(s, 1, 1.0, s2, False)

    with torch.no_grad():
        q_s2 = agente.objetivo(torch.as_tensor(s2).unsqueeze(0)).max().item()
    # Un paso de gradiente mueve Q(s,1) hacia 1.0 + 0.9·q_s2, que con pesos
    # inicializados no es 1.0 salvo coincidencia.
    objetivo_esperado = 1.0 + cfg.gamma * q_s2
    antes = agente.valores_q(s)[1]
    for _ in range(300):
        agente.aprender()
    despues = agente.valores_q(s)[1]
    assert abs(despues - objetivo_esperado) < abs(antes - objetivo_esperado)


def test_el_buffer_rechaza_estados_corruptos() -> None:
    buf = Replay(10, 3)
    malo = np.array([1.0, np.nan, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN"):
        buf.guardar(malo, 0, 0.0, np.zeros(3, dtype=np.float32), False)
    with pytest.raises(ValueError, match="forma"):
        buf.guardar(np.zeros(2, dtype=np.float32), 0, 0.0, np.zeros(2, dtype=np.float32), False)


def test_el_buffer_es_circular() -> None:
    buf = Replay(5, 2, semilla=0)
    for i in range(12):
        buf.guardar(
            np.full(2, i, dtype=np.float32),
            i % 3,
            float(i),
            np.full(2, i + 1, dtype=np.float32),
            False,
        )
    assert len(buf) == 5
    s, _a, r, s2, _fin = buf.muestrear(50)
    # Solo pueden aparecer las 5 últimas transiciones (i = 7..11).
    assert set(np.unique(r.astype(int))) <= {7, 8, 9, 10, 11}
    np.testing.assert_array_equal(s2[:, 0], s[:, 0] + 1)


def test_epsilon_decae_de_forma_lineal_y_se_estabiliza() -> None:
    cfg = Config(epsilon_inicial=1.0, epsilon_final=0.1, epsilon_pasos=100)
    agente = AgenteDQN(2, 3, cfg)
    assert agente.epsilon == pytest.approx(1.0)
    agente.pasos = 50
    assert agente.epsilon == pytest.approx(0.55)
    agente.pasos = 500
    assert agente.epsilon == pytest.approx(0.1)


def test_la_politica_greedy_es_determinista() -> None:
    agente = AgenteDQN(5, 3, Config(semilla=3))
    obs = np.linspace(-1, 1, 5).astype(np.float32)
    elecciones = {agente.actuar(obs, deterministico=True) for _ in range(20)}
    assert len(elecciones) == 1


def test_guardar_y_cargar_preserva_la_politica(tmp_path: Path) -> None:
    agente = AgenteDQN(5, 3, Config(semilla=1))
    obs = np.linspace(-1, 1, 5).astype(np.float32)
    antes = agente.valores_q(obs)
    ruta = tmp_path / "m.pt"
    agente.guardar(ruta)
    otro = AgenteDQN.cargar(ruta)
    np.testing.assert_allclose(antes, otro.valores_q(obs), rtol=1e-6)


def test_dos_agentes_con_la_misma_semilla_son_identicos() -> None:
    df = serie_sintetica(n=600)
    esc = features.Escalador.ajustar(features.construir(df["precio"]))
    cfg_env = ConfigEntorno(largo_episodio=30, semilla=4)

    def correr() -> list[int]:
        env = EntornoInversion(df, esc, cfg_env)
        ag = AgenteDQN(env.observation_space.shape[0], 3, Config(semilla=9, calentamiento=10))
        obs, _ = env.reset(seed=4)
        acciones = []
        for _ in range(60):
            a = ag.actuar(obs)
            acciones.append(a)
            obs2, r, term, trunc, _ = env.step(a)
            ag.recordar(obs, a, r, obs2, term)
            ag.aprender()
            obs = obs2
            if trunc:
                obs, _ = env.reset(seed=4)
        return acciones

    assert correr() == correr()
