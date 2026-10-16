"""El remuestreo por bloques tiene que preservar la estadística y destruir el orden.

Si preservara el orden, no sirve de nada (el agente seguiría memorizando). Si no
preservara la estadística, el agente aprendería de un mercado que no existe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features
from environment import ConfigEntorno, EntornoBootstrap
from test_entorno import serie_sintetica


@pytest.fixture(scope="module")
def montaje():
    df = serie_sintetica(n=3000, deriva=0.0003, sigma=0.02)
    esc = features.Escalador.ajustar(features.construir(df["precio"]))
    cfg = ConfigEntorno(largo_episodio=252, semilla=0)
    return df, esc, cfg


def test_cada_episodio_es_una_serie_distinta(montaje) -> None:
    df, esc, cfg = montaje
    env = EntornoBootstrap(df, esc, cfg, largo_bloque=21)
    env.reset()
    primera = env.precio.copy()
    env.reset()
    segunda = env.precio.copy()
    assert not np.allclose(primera, segunda), "dos episodios con la misma trayectoria"


def test_conserva_la_media_y_la_volatilidad_de_la_fuente(montaje) -> None:
    """Promediando muchos episodios, los momentos de los log-retornos
    sintéticos deben coincidir con los de la serie fuente."""
    df, esc, cfg = montaje
    env = EntornoBootstrap(df, esc, cfg, largo_bloque=21)
    fuente = np.diff(np.log(df["precio"].to_numpy()))

    muestras = []
    for _ in range(40):
        env.reset()
        muestras.append(np.diff(np.log(env.precio)))
    sintetico = np.concatenate(muestras)

    assert sintetico.std() == pytest.approx(fuente.std(), rel=0.05)
    # La media por barra es pequeña; se compara en unidades de la desviación.
    assert abs(sintetico.mean() - fuente.mean()) < 0.1 * fuente.std()


def test_preserva_el_agrupamiento_de_volatilidad_dentro_del_bloque(montaje) -> None:
    """La autocorrelación de |r| a un rezago debe sobrevivir al remuestreo:
    es lo que hace que el estado tenga información útil."""
    _, _, cfg = montaje
    rng = np.random.default_rng(0)
    # Serie fuente con agrupamiento explícito: volatilidad en dos regímenes.
    n = 3000
    vol = np.where(np.arange(n) % 500 < 250, 0.005, 0.03)
    r = rng.normal(0.0, vol)
    precio = 50 * np.exp(np.cumsum(r))
    import pandas as pd

    fuente = pd.DataFrame({"precio": precio}, index=pd.RangeIndex(n))
    fuente.attrs["primer_paso"] = features.CALENTAMIENTO
    esc2 = features.Escalador.ajustar(features.construir(fuente["precio"]))

    env = EntornoBootstrap(fuente, esc2, cfg, largo_bloque=63)
    env.reset()
    rs = np.abs(np.diff(np.log(env.precio)))
    acf1 = np.corrcoef(rs[:-1], rs[1:])[0, 1]
    assert acf1 > 0.1, f"se perdió el agrupamiento de volatilidad (acf1={acf1:.3f})"


def test_el_bloque_de_una_barra_destruye_el_agrupamiento(montaje) -> None:
    """Control: con bloques de 1 barra el remuestreo es i.i.d. y la
    autocorrelación de |r| desaparece. Confirma que el test anterior mide algo."""
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 3000
    vol = np.where(np.arange(n) % 500 < 250, 0.005, 0.03)
    r = rng.normal(0.0, vol)
    precio = 50 * np.exp(np.cumsum(r))
    fuente = pd.DataFrame({"precio": precio}, index=pd.RangeIndex(n))
    fuente.attrs["primer_paso"] = features.CALENTAMIENTO
    esc = features.Escalador.ajustar(features.construir(fuente["precio"]))

    env = EntornoBootstrap(fuente, esc, ConfigEntorno(largo_episodio=252), largo_bloque=1)
    env.reset()
    rs = np.abs(np.diff(np.log(env.precio)))
    acf1 = np.corrcoef(rs[:-1], rs[1:])[0, 1]
    assert abs(acf1) < 0.1, f"con bloque 1 no debería haber agrupamiento (acf1={acf1:.3f})"


def test_el_episodio_dura_lo_declarado(montaje) -> None:
    df, esc, cfg = montaje
    env = EntornoBootstrap(df, esc, cfg, largo_bloque=21)
    env.reset()
    pasos = 0
    truncado = False
    while not truncado:
        _, _, _, truncado, info = env.step(2)
        pasos += 1
    assert pasos == cfg.largo_episodio == info["pasos"]


def test_exige_episodios_de_largo_fijo(montaje) -> None:
    df, esc, _ = montaje
    with pytest.raises(ValueError, match="largo fijo"):
        EntornoBootstrap(df, esc, ConfigEntorno(largo_episodio=0))


def test_la_deriva_aleatoria_cambia_el_regimen_entre_episodios(montaje) -> None:
    """Sin deriva aleatoria todos los episodios heredan la tendencia de la
    fuente; con ella, unos suben y otros bajan. Es lo que impide que una
    exposición fija sea la respuesta óptima."""
    df, esc, cfg = montaje

    sin = EntornoBootstrap(df, esc, cfg, largo_bloque=21, deriva_aleatoria=0.0)
    con = EntornoBootstrap(df, esc, cfg, largo_bloque=21, deriva_aleatoria=0.25)

    def retornos_de_episodios(env, n=40):
        salida = []
        for s in range(n):
            env.reset(seed=1000 + s)
            p = env.precio[env.primer_paso : env.ultimo_paso + 1]
            salida.append(p[-1] / p[0] - 1.0)
        return np.array(salida)

    r_sin = retornos_de_episodios(sin)
    r_con = retornos_de_episodios(con)

    # La dispersión entre episodios tiene que crecer de forma clara.
    assert r_con.std() > 1.5 * r_sin.std(), (r_sin.std(), r_con.std())
    # Y tiene que haber episodios de ambos signos en cantidad apreciable.
    proporcion_alcista = float((r_con > 0).mean())
    assert 0.25 < proporcion_alcista < 0.75, proporcion_alcista


def test_la_deriva_aleatoria_quita_el_sesgo_de_la_fuente(montaje) -> None:
    """La serie fuente tiene deriva positiva; promediando episodios con deriva
    aleatoria, el retorno esperado debe quedar centrado en cero."""
    df, esc, cfg = montaje
    env = EntornoBootstrap(df, esc, cfg, largo_bloque=21, deriva_aleatoria=0.25)
    medias = []
    for s in range(60):
        env.reset(seed=2000 + s)
        medias.append(np.diff(np.log(env.precio)).mean())
    medias = np.array(medias)
    error_estandar = medias.std(ddof=1) / np.sqrt(len(medias))
    assert abs(medias.mean()) < 3 * error_estandar, (medias.mean(), error_estandar)
