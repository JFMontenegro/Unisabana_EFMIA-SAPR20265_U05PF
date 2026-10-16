"""Líneas base contra las que se mide el agente.

Las dos que pide el enunciado:

- **comprar_y_mantener**: 100 % invertido desde la primera barra hasta la
  última. Paga un solo costo de entrada. Es la línea base dura: en un activo
  con deriva positiva es muy difícil de superar, y cualquier agente que la
  iguale con menos drawdown ya está aportando algo.
- **aleatoria**: acciones uniformes en {0, 1, 2}. Se corre con 30 semillas y se
  reporta media ± desviación, porque una sola corrida aleatoria no es una línea
  base, es una anécdota. Su rotación esperada (~1 vez cada 1.5 barras) la hace
  perder dinero solo en costos: con 10 bps y 252 barras al año eso es ~11 %
  anual de fricción. Eso es parte del punto — muestra cuánto cuesta operar sin
  información.

Y dos que se añaden para que la comparación no sea contra espantapájaros:

- **mitad_fija**: 50 % permanente. Tiene la mitad de la volatilidad del activo
  sin ninguna inteligencia. Si el agente no la supera en Sharpe, lo que aprendió
  es a diluir exposición, no a elegir momentos.
- **cruce_de_medias**: regla técnica clásica — invertido cuando la media de 21
  barras está por encima de la de 63, en efectivo si no. Es la estrategia que
  un practicante probaría en diez minutos sin aprendizaje por refuerzo. Es la
  vara que de verdad importa.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

import features
from environment import (
    COSTO_POR_DEFECTO,
    ConfigEntorno,
    EntornoInversion,
    recorrido_completo,
    rodar,
)

Politica = Callable[[np.ndarray], int]

# Índices de los indicadores en el vector de estado, para la regla técnica.
IDX_SMA_CORTA = features.NOMBRES.index("dist_sma_corta")
IDX_SMA_LARGA = features.NOMBRES.index("dist_sma_larga")


def comprar_y_mantener(_: np.ndarray) -> int:
    return 2


def efectivo(_: np.ndarray) -> int:
    return 0


def mitad_fija(_: np.ndarray) -> int:
    return 1


def cruce_de_medias(escalador: features.Escalador) -> Politica:
    """Invertido cuando SMA(21) > SMA(63) en escala logarítmica, efectivo si no.

    La observación llega normalizada y cada indicador con su propia media y
    desviación, así que comparar los valores estandarizados NO equivale a
    comparar las medias: hay que deshacer la normalización primero.

    El sentido de la desigualdad se invierte, y conviene escribirlo:

        dist_sma_k = log P − log SMA_k
        SMA_21 > SMA_63  ⇔  log SMA_21 > log SMA_63  ⇔  dist_21 < dist_63

    Es decir, la señal alcista clásica («la media corta cruza por encima de la
    larga») corresponde a `dist_21 < dist_63`. Se verifica numéricamente contra
    un cálculo directo sobre los precios en `tests/test_baselines.py`.

    Salvedad: `Escalador.aplicar` recorta a ±5 σ, así que en los días más
    extremos la des-normalización no recupera el valor exacto. El test mide
    cuántas barras se ven afectadas; en el Brent son unas pocas de 9 000.
    """
    mc, sc = escalador.medias[IDX_SMA_CORTA], escalador.desviaciones[IDX_SMA_CORTA]
    ml, sl = escalador.medias[IDX_SMA_LARGA], escalador.desviaciones[IDX_SMA_LARGA]

    def politica(obs: np.ndarray) -> int:
        dist_corta = obs[IDX_SMA_CORTA] * sc + mc
        dist_larga = obs[IDX_SMA_LARGA] * sl + ml
        # SMA_corta > SMA_larga  ⇔  dist_corta < dist_larga
        return 2 if dist_corta < dist_larga else 0

    return politica


def aleatoria(semilla: int) -> Politica:
    rng = np.random.default_rng(semilla)

    def politica(_: np.ndarray) -> int:
        return int(rng.integers(3))

    return politica


def politicas_fijas(escalador: features.Escalador) -> dict[str, Politica]:
    """Las líneas base deterministas, en el orden en que se reportan."""
    return {
        "Comprar y mantener": comprar_y_mantener,
        "Cruce de medias": cruce_de_medias(escalador),
        "50 % fijo": mitad_fija,
        "Todo en efectivo": efectivo,
    }


def correr_todas(
    precios: pd.DataFrame,
    escalador: features.Escalador,
    cfg: ConfigEntorno,
    *,
    semillas_aleatorias: int = 30,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Devuelve (bitácoras de las políticas fijas, bitácoras aleatorias apiladas).

    Las aleatorias se devuelven aparte porque se resumen como distribución, no
    como una curva única.
    """
    bitacoras: dict[str, pd.DataFrame] = {}
    for nombre, politica in politicas_fijas(escalador).items():
        env = recorrido_completo(precios, escalador, cfg)
        bitacoras[nombre] = rodar(env, politica)

    aleatorias = []
    for s in range(semillas_aleatorias):
        env = recorrido_completo(precios, escalador, cfg)
        b = rodar(env, aleatoria(1000 + s))
        b["semilla"] = s
        aleatorias.append(b)
    return bitacoras, pd.concat(aleatorias)


def resumir_aleatorias(apiladas: pd.DataFrame, barras_anio: int = 252) -> dict[str, float]:
    """Media y desviación entre semillas de cada métrica."""
    import metrics

    por_semilla = [
        metrics.resumir(g.drop(columns="semilla"), barras_anio)
        for _, g in apiladas.groupby("semilla")
    ]
    df = pd.DataFrame(por_semilla)
    resumen = df.mean().to_dict()
    resumen.update({f"{k}_desv": v for k, v in df.std(ddof=1).to_dict().items()})
    resumen["semillas"] = float(len(por_semilla))
    return resumen


def _construir_entorno(precios: pd.DataFrame, cfg: ConfigEntorno) -> EntornoInversion:
    """Atajo de conveniencia para pruebas rápidas desde el notebook."""
    esc = features.Escalador.ajustar(features.construir(precios["precio"]))  # ventanas diarias
    return recorrido_completo(precios, esc, cfg)


def main() -> None:
    """Ejecuta las líneas base sobre un tramo y reporta sus métricas.

    El enunciado pide que el README explique cómo «ejecutar los baselines»
    como un paso propio, así que este módulo es ejecutable:

        python src/baselines.py --mercado btc --tramo prueba
    """
    import argparse

    import data
    import metrics
    from train import montar

    p = argparse.ArgumentParser(description="Corre las líneas base y reporta sus métricas")
    p.add_argument("--mercado", default="btc", choices=list(data.FUENTES))
    p.add_argument("--tramo", default="prueba", choices=["entrenamiento", "validacion", "prueba"])
    p.add_argument("--semillas-aleatorias", type=int, default=30)
    p.add_argument("--costo", type=float, default=COSTO_POR_DEFECTO)
    p.add_argument("--capital", type=float, default=100_000.0)
    a = p.parse_args()

    tramos, escalador = montar(a.mercado, incluir_prueba=(a.tramo == "prueba"))
    precios = tramos[a.tramo]
    barras_anio = data.BARRAS_POR_ANIO[a.mercado]
    cfg = ConfigEntorno(costo=a.costo, largo_episodio=0, aleatorizar_inicio=False)

    fijas, aleatorias = correr_todas(
        precios, escalador, cfg, semillas_aleatorias=a.semillas_aleatorias
    )
    resumenes = {n: metrics.resumir(b, barras_anio) for n, b in fijas.items()}
    resumenes[f"Aleatoria ({a.semillas_aleatorias} semillas)"] = resumir_aleatorias(
        aleatorias, barras_anio
    )

    primer = precios.index[precios.attrs["primer_paso"]]
    print(f"Líneas base · {a.mercado} · tramo de {a.tramo} · {primer:%Y-%m-%d} → {precios.index[-1]:%Y-%m-%d}")
    print(f"Costo por operar: {10_000 * a.costo:.0f} bps sobre |cambio de exposición|\n")
    tabla = metrics.tabla(resumenes)
    print(metrics.formatear(tabla).to_string())


if __name__ == "__main__":
    main()
