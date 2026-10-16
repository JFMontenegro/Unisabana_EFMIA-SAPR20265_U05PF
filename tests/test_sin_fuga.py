"""La prueba que sostiene la honestidad del experimento.

Si cualquiera de estos tests falla, los resultados del proyecto no valen nada:
significa que el agente está viendo el futuro.

Se corren sobre **los dos mercados**, con sus respectivas ventanas: `btc`
(horario, ventanas de 1 h / 1 d / 1 semana / 1 mes) y `brent` (diario, ventanas
de 1 semana / 1 mes / 1 trimestre / 1 año). Una fuga puede depender de la
longitud de la ventana, así que probar solo uno no bastaría.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data
import features
from environment import ConfigEntorno, EntornoInversion

MERCADOS = ("btc", "brent")


@pytest.fixture(scope="module")
def series() -> dict[str, pd.Series]:
    return {m: data.cargar(m).precios["precio"] for m in MERCADOS}


def _montar(series: dict[str, pd.Series], mercado: str, tramo: str = "entrenamiento"):
    v = data.VENTANAS[mercado]
    df = series[mercado].to_frame("precio")
    t = data.partir(df, mercado, tramo)
    esc = features.Escalador.ajustar(features.construir(t["precio"], v), v)
    return t, esc, v


@pytest.mark.parametrize("mercado", MERCADOS)
def test_indicadores_no_dependen_del_futuro(series, mercado: str) -> None:
    """El indicador en `t` calculado con la serie truncada en `t` debe ser
    idéntico al calculado con la serie completa.

    Es la definición operativa de «sin lookahead». Un `rolling(center=True)`,
    un `bfill()` o un `pct_change(-1)` rompen exactamente esta igualdad.
    """
    v = data.VENTANAS[mercado]
    precios = series[mercado]
    completos = features.construir(precios, v)
    rng = np.random.default_rng(0)
    posiciones = rng.integers(v.calentamiento, len(precios), size=25)

    for t in posiciones:
        truncados = features.construir(precios.iloc[: t + 1], v)
        np.testing.assert_allclose(
            truncados.iloc[-1].to_numpy(dtype=np.float64),
            completos.iloc[t].to_numpy(dtype=np.float64),
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"fuga de información en {mercado} t={t} ({precios.index[t]})",
        )


@pytest.mark.parametrize("mercado", MERCADOS)
def test_el_escalador_solo_ve_entrenamiento(series, mercado: str) -> None:
    """Las estadísticas de normalización no pueden moverse al añadir datos
    futuros: se estiman en entrenamiento y se congelan."""
    v = data.VENTANAS[mercado]
    _, esc_a, _ = _montar(series, mercado)

    # Volver a ajustarlo con el tramo de prueba incluido debe dar OTRA cosa:
    # si diera lo mismo, el test no estaría probando nada.
    esc_b = features.Escalador.ajustar(features.construir(series[mercado], v), v)
    assert not np.allclose(esc_a.medias, esc_b.medias), (
        "las estadísticas de entrenamiento y de la serie completa coinciden; "
        "el test no discrimina"
    )

    # Y el escalador que usa el proyecto es el primero, reproducible exacto.
    _, esc_c, _ = _montar(series, mercado)
    np.testing.assert_array_equal(esc_a.medias, esc_c.medias)
    np.testing.assert_array_equal(esc_a.desviaciones, esc_c.desviaciones)


@pytest.mark.parametrize("mercado", MERCADOS)
def test_la_observacion_no_contiene_el_retorno_que_paga(series, mercado: str) -> None:
    """Prueba de contaminación: si se altera `P_{t+1}` hacia adelante, la
    observación en `t` no puede cambiar — solo la recompensa."""
    tr, esc, _ = _montar(series, mercado)
    cfg = ConfigEntorno(largo_episodio=50, aleatorizar_inicio=False, semilla=0)

    env_a = EntornoInversion(tr, esc, cfg)
    obs_a, _ = env_a.reset(seed=0)

    # Duplicar todos los precios desde el segundo paso operable en adelante.
    corte = env_a.primer_paso + 1
    alterado = tr.copy()
    alterado.iloc[corte:, alterado.columns.get_loc("precio")] *= 2.0
    alterado.attrs.update(tr.attrs)
    env_b = EntornoInversion(alterado, esc, cfg)
    obs_b, _ = env_b.reset(seed=0)

    np.testing.assert_array_equal(
        obs_a,
        obs_b,
        err_msg="la observación inicial cambió al alterar precios futuros",
    )
    # La recompensa, en cambio, sí debe cambiar: ahí es donde entra P_{t+1}.
    _, r_a, *_ = env_a.step(2)
    _, r_b, *_ = env_b.step(2)
    assert not np.isclose(r_a, r_b), "la recompensa ignoró el precio futuro"


@pytest.mark.parametrize("mercado", MERCADOS)
def test_las_barras_de_calentamiento_no_son_operables(series, mercado: str) -> None:
    """El primer paso operable debe tener indicadores completos."""
    for tramo in ("entrenamiento", "validacion", "prueba"):
        t, esc, v = _montar(series, mercado, tramo)
        env = EntornoInversion(t, esc, ConfigEntorno(largo_episodio=10))
        obs, _ = env.reset(seed=0)
        assert np.isfinite(obs).all(), f"{mercado}/{tramo}: observación con NaN/inf"
        assert env.primer_paso >= v.calentamiento


@pytest.mark.parametrize("mercado", MERCADOS)
def test_los_tramos_no_se_solapan_en_fechas_operables(mercado: str) -> None:
    """El calentamiento puede solaparse (solo calcula indicadores), pero
    ninguna barra se opera en dos tramos."""
    df = data.cargar(mercado).precios
    operables = {}
    for tramo in ("entrenamiento", "validacion", "prueba"):
        t = data.partir(df, mercado, tramo)
        operables[tramo] = set(t.index[t.attrs["primer_paso"] : -1])

    assert not operables["entrenamiento"] & operables["validacion"]
    assert not operables["validacion"] & operables["prueba"]
    assert not operables["entrenamiento"] & operables["prueba"]


def test_el_entorno_usa_las_ventanas_del_escalador(series) -> None:
    """Un escalador horario no puede acabar alimentando indicadores diarios.

    El entorno toma las ventanas del escalador precisamente para que este
    desajuste sea imposible; el test lo fija.
    """
    tr, esc, v = _montar(series, "btc")
    env = EntornoInversion(tr, esc, ConfigEntorno(largo_episodio=10))
    assert env.ventanas.nombre == "horario"
    assert env.ventanas is v
    assert env.primer_paso >= features.HORARIO.calentamiento


def test_el_volumen_intercambiado_del_dataset_se_detecta_y_corrige() -> None:
    """Defecto real del dataset recomendado: en `BTC_USD-Hourly.csv` las
    columnas `volume` y `Volume USD` están al revés hasta finales de 2020.

    Se verifica contra la identidad que tiene que cumplirse siempre,
    `Volume USD ≈ volume × precio`, y que el cargador deja constancia.
    """
    m = data.cargar("btc")
    assert any("volumen" in a for a in m.avisos), m.avisos

    df = m.precios
    razon = df["Volume USD"] / df["volume"].replace(0, np.nan)
    error = np.abs(np.log(razon / df["precio"])).dropna()

    # Tras la corrección, la razón debe ser el precio de cierre. No coincide
    # al 100 % porque el volumen en dólares se calcula con un precio medio de
    # la hora, no con el cierre: en horas de mucho movimiento (el 13-mar-2020,
    # por ejemplo) la diferencia llega al 9 %. Por eso se verifica la mediana y
    # una cota, no igualdad exacta.
    assert error.median() < 0.002, float(error.median())
    assert error.max() < 0.15, float(error.max())
    assert (error < 0.02).mean() > 0.99, float((error < 0.02).mean())
