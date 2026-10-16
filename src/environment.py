"""Entorno de inversión compatible con Gymnasium.

Formulación del problema de decisión secuencial
-----------------------------------------------
En cada barra `t` el agente observa el estado `s_t`, construido **solo** con
precios hasta `P_t`, y elige la fracción de capital expuesta al activo para el
intervalo `t -> t+1`:

    a_t ∈ {0 %, 50 %, 100 %}

El resto del capital queda en efectivo, que se remunera al 0 %. El retorno del
portafolio en ese intervalo es

    r^port_{t+1} = a_t · r_{t+1} − c · |a_t − a_{t-1}|      con  r_{t+1} = P_{t+1}/P_t − 1

y el patrimonio evoluciona como `E_{t+1} = E_t · (1 + r^port_{t+1})`.

Tres decisiones de diseño que vale la pena hacer explícitas
-----------------------------------------------------------
1. **El costo se cobra sobre la rotación, no sobre la posición.** `c · |Δa|`
   con `c = 10 bps`. Cobrarlo sobre `a_t` sería un impuesto a estar invertido y
   el agente aprendería a no invertir; cobrarlo solo al abrir y no al cerrar
   dejaría media operación gratis. Pasar de 0 % a 100 % cuesta 10 bps del
   capital, y volver cuesta otros 10.

2. **La exposición actual forma parte del estado.** Sin ella el problema deja
   de ser markoviano: el costo de elegir 100 % depende de si ya se está en
   100 %, y un agente que no observa su propia posición no puede distinguir
   «mantener gratis» de «entrar pagando».

3. **El retorno que paga la acción es el de `t+1`, que el estado no contiene.**
   Es la frontera de la causalidad. El estado se construye con `P_0..P_t` y la
   recompensa con `P_{t+1}`; `tests/test_sin_fuga.py` lo verifica recalculando
   los indicadores sobre la serie truncada.

Funciones de recompensa
-----------------------
Tres, para poder comparar e iterar (ver `README.md` §5):

- `log_neto`      : log(1 + r^port). Sumada sobre el episodio da exactamente el
                    log del patrimonio final. Es la recompensa «correcta» en el
                    sentido de que optimiza el objetivo declarado.
- `drawdown`      : log_neto − λ · incremento del drawdown del portafolio.
                    Penaliza el daño nuevo, no el drawdown acumulado, para que
                    la penalización no se cobre dos veces.
- `sharpe_dif`    : Sharpe diferencial de Moody & Saffell (1998): la derivada
                    del Sharpe móvil respecto al retorno del período, calculada
                    con medias exponenciales. Recompensa densa y ajustada por
                    riesgo sin necesitar el episodio completo.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

import features

EXPOSICIONES = (0.0, 0.5, 1.0)
COSTO_POR_DEFECTO = 0.0010  # 10 puntos básicos sobre |Δ exposición|


@dataclass
class ConfigEntorno:
    """Parámetros del entorno. Todo lo que cambia el problema vive aquí."""

    costo: float = COSTO_POR_DEFECTO
    recompensa: str = "log_neto"
    lambda_drawdown: float = 10.0
    eta_sharpe: float = 0.02  # tasa de la media exponencial del Sharpe diferencial
    # Barras que transcurren entre dos decisiones. Con 1 el agente decide cada
    # día; con 5, cada semana, manteniendo la exposición en el intermedio.
    # No es un detalle de eficiencia: con 10 bps de costo y una razón
    # señal/ruido por barra de ~0.02, decidir todos los días hace que la
    # fricción domine cualquier ventaja informativa. Se mide en el README §6.
    frecuencia_decision: int = 1
    largo_episodio: int = 252  # un año de barras; 0 = recorrer todo el tramo
    aleatorizar_inicio: bool = True
    exposicion_inicial: float = 0.0
    escala_recompensa: float = 100.0  # log-retornos diarios ~1e-2; la red ve ~1e0
    semilla: int | None = None
    _validas: tuple[str, ...] = field(
        default=("log_neto", "drawdown", "sharpe_dif"), repr=False
    )

    def __post_init__(self) -> None:
        if self.recompensa not in self._validas:
            raise ValueError(f"recompensa '{self.recompensa}' no está en {self._validas}")
        if not 0.0 <= self.costo < 0.1:
            raise ValueError(f"costo fuera de rango: {self.costo}")


class EntornoInversion(gym.Env):
    """Un activo, efectivo al 0 %, tres niveles de exposición."""

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        precios: pd.DataFrame,
        escalador: features.Escalador,
        cfg: ConfigEntorno | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or ConfigEntorno()
        self.escalador = escalador
        # Las ventanas no se pasan aparte: las lleva el escalador. Así el
        # entorno no puede construir indicadores con una longitud de ventana
        # distinta de la que se usó para estimar medias y desviaciones, que
        # sería un desajuste silencioso y difícil de ver.
        self.ventanas = features.PRESETS[escalador.ventanas]
        self._preparar(precios)

        self.action_space = spaces.Discrete(len(EXPOSICIONES))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.estados.shape[1] + 1,),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng(self.cfg.semilla)
        self.t = self.primer_paso
        self.exposicion = self.cfg.exposicion_inicial

    def _preparar(self, precios: pd.DataFrame) -> None:
        """Precalcula precios, indicadores y retornos alineados.

        Está separado de `__init__` porque `EntornoBootstrap` lo vuelve a
        llamar en cada `reset` con una trayectoria sintética distinta.
        """
        if "precio" not in precios.columns:
            raise ValueError("se esperaba una columna 'precio'")
        self.fechas = precios.index
        self.precio = precios["precio"].to_numpy(dtype=np.float64)
        self.n = self.precio.size

        self.estados = self.escalador.aplicar(
            features.construir(precios["precio"], self.ventanas)
        )

        self.primer_paso = int(
            precios.attrs.get("primer_paso", self.ventanas.calentamiento)
        )
        # Último índice operable: necesita P_{t+1} para pagar la acción.
        self.ultimo_paso = self.n - 2
        if self.ultimo_paso <= self.primer_paso:
            raise ValueError(
                f"tramo demasiado corto: {self.n} barras, primer paso {self.primer_paso}"
            )
        if np.isnan(self.estados[self.primer_paso]).any():
            raise ValueError(
                f"el estado del primer paso ({self.fechas[self.primer_paso]}) "
                "tiene indicadores incompletos; falta calentamiento"
            )

        # Retornos simples alineados: self.ret[t] = P_{t+1}/P_t − 1, el retorno
        # que paga la decisión tomada en t.
        self.ret = np.empty(self.n, dtype=np.float64)
        self.ret[:-1] = self.precio[1:] / self.precio[:-1] - 1.0
        self.ret[-1] = np.nan

    # ------------------------------------------------------------------ gym
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        largo = self.cfg.largo_episodio or (self.ultimo_paso - self.primer_paso)
        if self.cfg.aleatorizar_inicio and largo < (self.ultimo_paso - self.primer_paso):
            self.t = int(self._rng.integers(self.primer_paso, self.ultimo_paso - largo + 1))
        else:
            self.t = self.primer_paso
        self.fin = min(self.t + largo, self.ultimo_paso)

        self.exposicion = self.cfg.exposicion_inicial
        self.patrimonio = 1.0
        self.pico = 1.0
        self.drawdown = 0.0
        self.costos_pagados = 0.0
        self.rotacion = 0.0
        self.pasos = 0
        # Medias exponenciales del Sharpe diferencial.
        self._m1 = 0.0
        self._m2 = 0.0

        return self._observar(), self._info()

    def step(self, accion: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        accion = int(accion)
        if not 0 <= accion < len(EXPOSICIONES):
            raise ValueError(f"acción inválida: {accion}")

        nueva = EXPOSICIONES[accion]
        recompensa = 0.0
        self.sub_barras = []
        # El costo se cobra una sola vez, en la barra de la decisión; las barras
        # intermedias solo acumulan el retorno de la exposición ya tomada.
        for i in range(self.cfg.frecuencia_decision):
            if self.t > self.ultimo_paso:
                break
            recompensa += self._avanzar(nueva, cobrar_costo=(i == 0))
            if self.t >= self.fin:
                break

        truncado = self.t >= self.fin or self.t > self.ultimo_paso
        terminado = False  # no hay estado absorbente: el mercado no «termina»
        obs = self._observar(ultimo=truncado)
        return obs, recompensa, terminado, truncado, self._info(r_activo=self.sub_barras[-1][1])

    def _avanzar(self, nueva: float, *, cobrar_costo: bool) -> float:
        """Aplica la exposición durante una barra y devuelve su recompensa."""
        rotacion = abs(nueva - self.exposicion) if cobrar_costo else 0.0
        costo = self.cfg.costo * rotacion

        r_activo = float(self.ret[self.t])
        # El costo se paga en el momento de operar, ANTES de que el retorno se
        # acumule sobre el capital restante:
        #     E_{t+1} = E_t · (1 − c·|Δa|) · (1 + a_t·r_{t+1})
        # Es un orden distinto al habitual `E_t·(1 + a·r − c)` y la diferencia
        # es de segundo orden (c·r ≈ 1e-6), pero deja un invariante exacto que
        # se puede testear: comprar y mantener da justo (P_T/P_0)·(1 − c).
        r_port = (1.0 - costo) * (1.0 + nueva * r_activo) - 1.0

        # El patrimonio no puede volverse negativo con exposición ≤ 100 % salvo
        # por un retorno simple < −100 %, imposible en precios positivos.
        anterior = self.patrimonio
        self.patrimonio = anterior * (1.0 + r_port)
        self.pico = max(self.pico, self.patrimonio)
        dd_anterior = self.drawdown
        self.drawdown = self.patrimonio / self.pico - 1.0  # ≤ 0

        recompensa = self._recompensar(r_port, dd_anterior)

        self.exposicion = nueva
        # Costo en unidades de capital inicial (no en fracción de rotación):
        # es la cifra que el informe tiene que reportar.
        self.costos_pagados += anterior * costo
        self.rotacion += rotacion
        self.pasos += 1
        # Registro por barra, no por decisión: la contabilidad y las métricas
        # siempre se calculan sobre barras, cambie o no la frecuencia con la
        # que el agente decide.
        self.sub_barras.append((self.fechas[self.t], r_activo, self.patrimonio))
        self.t += 1
        return recompensa

    # -------------------------------------------------------------- interno
    def _recompensar(self, r_port: float, dd_anterior: float) -> float:
        base = float(np.log1p(r_port))
        k = self.cfg.escala_recompensa

        if self.cfg.recompensa == "log_neto":
            return k * base

        if self.cfg.recompensa == "drawdown":
            # Solo el daño NUEVO. Penalizar el nivel del drawdown cobraría el
            # mismo pozo en cada barra hasta recuperarse y el agente aprendería
            # a cerrar posiciones para «resetear» el pico, que no es lo que se
            # quiere incentivar.
            empeora = max(0.0, dd_anterior - self.drawdown)
            return k * (base - self.cfg.lambda_drawdown * empeora)

        # Sharpe diferencial (Moody & Saffell, 1998). Con A y B las medias
        # exponenciales del retorno y del retorno al cuadrado:
        #   D_t = (B_{t-1}·ΔA − ½·A_{t-1}·ΔB) / (B_{t-1} − A_{t-1}²)^{3/2}
        eta = self.cfg.eta_sharpe
        a_ant, b_ant = self._m1, self._m2
        d_a = r_port - a_ant
        d_b = r_port * r_port - b_ant
        varianza = b_ant - a_ant * a_ant
        if varianza <= 1e-12 or self.pasos < 2:
            # Al arranque la varianza es ~0 y el cociente explota. Devolver el
            # retorno crudo evita que las primeras barras dominen el gradiente.
            dif = base
        else:
            dif = (b_ant * d_a - 0.5 * a_ant * d_b) / varianza**1.5
            dif = float(np.clip(dif, -10.0, 10.0))
        self._m1 = a_ant + eta * d_a
        self._m2 = b_ant + eta * d_b
        return dif

    def _observar(self, ultimo: bool = False) -> np.ndarray:
        idx = min(self.t, self.n - 1) if ultimo else self.t
        base = self.estados[idx]
        return np.concatenate([base, [self.exposicion - 0.5]]).astype(np.float32)

    def _info(self, r_activo: float | None = None) -> dict:
        info = {
            "fecha": self.fechas[min(self.t, self.n - 1)],
            "patrimonio": self.patrimonio,
            "exposicion": self.exposicion,
            "drawdown": self.drawdown,
            "costos": self.costos_pagados,
            "rotacion": self.rotacion,
            "pasos": self.pasos,
        }
        if r_activo is not None:
            info["ret_activo"] = r_activo
        return info


class EntornoBootstrap(EntornoInversion):
    """Entrena sobre trayectorias sintéticas remuestreadas por bloques.

    El problema que resuelve
    ------------------------
    Con 11 indicadores continuos y ~5 800 barras, el estado identifica
    prácticamente de forma única cada fecha del tramo de entrenamiento: una red
    de 128×128 aprende la secuencia del Brent de memoria en pocos miles de
    pasos. En la primera iteración de este proyecto eso dio un retorno de
    entrenamiento del orden de 10⁹ % con validación NEGATIVA. El agente no
    había aprendido una estrategia; había aprendido un calendario.

    Cómo funciona el remuestreo
    ---------------------------
    En cada episodio se construye una serie de precios nueva concatenando
    bloques de log-retornos tomados al azar del tramo de entrenamiento:

        r_sintético = [ bloque_{i1}, bloque_{i2}, ... ]        |bloque| = L
        P_sintético = P_0 · exp(cumsum(r_sintético))

    Con L ≈ 21 barras se preserva lo que importa de la estructura temporal —
    agrupamiento de volatilidad, colas gruesas, autocorrelación de corto
    plazo — y se destruye lo único que el agente estaba explotando: el orden
    concreto en que ocurrió la historia. El agente ya no puede memorizar porque
    nunca ve dos veces la misma serie.

    Lo que el remuestreo NO preserva, y hay que decirlo: la tendencia de largo
    plazo y las dependencias a más de L barras. Un bloque de 21 días rompe
    cualquier regularidad trimestral. Es un sesgo conocido del bootstrap por
    bloques y es el precio que se paga por eliminar la memorización.

    El segundo problema, y la razón de `deriva_aleatoria`
    ------------------------------------------------------
    El remuestreo conserva la deriva **incondicional** del tramo fuente. El
    Brent subió 490 % entre 1987 y 2013, así que en el mundo sintético el activo
    tiene retorno esperado positivo y la política óptima es *estar siempre
    invertido*: el agente no aprende a decidir, aprende que invertir paga. Eso
    se vio en la fase 2 — la memorización desapareció pero el agente salió con
    70-80 % de exposición permanente y perdió en un tramo de validación bajista.

    `deriva_aleatoria` lo corrige sorteando una deriva por episodio:

        μ ~ Normal(0, σ_μ)   anualizada,  añadida a cada barra del episodio

    Con σ_μ ≈ 25 % anual, unos episodios son alcistas y otros bajistas. El
    agente ya no puede acertar con una exposición fija, porque ninguna lo es en
    promedio; la única forma de ganar es **inferir el régimen** de lo que sí
    observa —momentum, distancia a las medias, volatilidad— que es exactamente
    la habilidad que se le pide en el período de prueba.
    """

    def __init__(
        self,
        precios: pd.DataFrame,
        escalador: features.Escalador,
        cfg: ConfigEntorno | None = None,
        *,
        largo_bloque: int = 21,
        deriva_aleatoria: float = 0.0,
    ) -> None:
        cfg = cfg or ConfigEntorno()
        if not cfg.largo_episodio:
            raise ValueError("el entorno con bootstrap necesita episodios de largo fijo")
        self.largo_bloque = int(largo_bloque)
        self.deriva_aleatoria = float(deriva_aleatoria)
        serie = precios["precio"].to_numpy(dtype=np.float64)
        self.retornos_fuente = np.diff(np.log(serie))
        self.precio_inicial = float(serie[0])
        self.ventanas = features.PRESETS[escalador.ventanas]
        self._largo_sintetico = self.ventanas.calentamiento + cfg.largo_episodio + 2
        if self.retornos_fuente.size < self._largo_sintetico:
            raise ValueError("el tramo fuente es más corto que un episodio sintético")
        self._semilla_base = cfg.semilla if cfg.semilla is not None else 0
        self._contador = 0
        super().__init__(self._sintetizar(self._semilla_base), escalador, cfg)

    def _sintetizar(self, semilla: int) -> pd.DataFrame:
        rng = np.random.default_rng(semilla)
        n_bloques = int(np.ceil(self._largo_sintetico / self.largo_bloque))
        maximo = self.retornos_fuente.size - self.largo_bloque
        inicios = rng.integers(0, maximo + 1, size=n_bloques)
        trozos = [self.retornos_fuente[i : i + self.largo_bloque] for i in inicios]
        r = np.concatenate(trozos)[: self._largo_sintetico]
        if self.deriva_aleatoria > 0.0:
            # Se quita la deriva incondicional del tramo fuente y se reemplaza
            # por una sorteada para este episodio. Así el régimen cambia de
            # episodio a episodio y ninguna exposición fija es óptima.
            mu_episodio = rng.normal(0.0, self.deriva_aleatoria) / 252.0
            r = r - self.retornos_fuente.mean() + mu_episodio
        precio = self.precio_inicial * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
        # Fechas ficticias: el entorno nunca las usa para decidir, solo para
        # indexar. Se marcan con un año imposible para que ninguna gráfica las
        # confunda con fechas reales.
        idx = pd.RangeIndex(precio.size, name="barra")
        df = pd.DataFrame({"precio": precio}, index=idx)
        df.attrs["primer_paso"] = self.ventanas.calentamiento
        df.attrs["sintetico"] = True
        return df

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        self._contador += 1
        semilla = seed if seed is not None else self._semilla_base * 1_000_003 + self._contador
        self._preparar(self._sintetizar(semilla))
        # La llamada al padre reinicia la contabilidad (patrimonio, pico,
        # costos, medias del Sharpe); su observación se descarta porque el
        # inicio aquí no se sortea: la serie ya es distinta en cada episodio,
        # así que se recorre completa desde el primer paso operable.
        super().reset(seed=None, options=options)
        self.t = self.primer_paso
        self.fin = min(self.primer_paso + self.cfg.largo_episodio, self.ultimo_paso)
        return self._observar(), self._info()


def politica_tabulada(env: EntornoInversion, valores):
    """Política greedy que lee una tabla Q precalculada en vez de la red.

    `valores` es cualquier objeto con `tabla_q(estados, exposiciones)` — un
    agente o un comité. La tabla se calcula una sola vez para todas las barras
    del tramo y las tres exposiciones posibles; después, decidir es indexar.

    Es exactamente la misma política que `agente.politica_greedy()`: misma red,
    mismos estados, mismo `argmax`. Lo único que cambia es cuándo se evalúa la
    red. `tests/test_evaluacion_rapida.py` verifica que las dos produzcan la
    misma secuencia de acciones.
    """
    tabla = valores.tabla_q(env.estados, EXPOSICIONES)
    indice_exposicion = {e: i for i, e in enumerate(EXPOSICIONES)}

    def politica(_obs: np.ndarray) -> int:
        valores = tabla[env.t, indice_exposicion[env.exposicion]]
        if not np.isfinite(valores).all():
            # Solo puede pasar si se intenta decidir sobre una barra de
            # calentamiento, donde los indicadores aún no están completos.
            # Elegir el argmax de eso daría una acción arbitraria sin avisar.
            raise ValueError(
                f"estado incompleto en la barra {env.t} ({env.fechas[env.t]}): "
                "no se puede decidir sobre indicadores con NaN"
            )
        return int(np.argmax(valores))

    return politica


def rodar(env: EntornoInversion, politica) -> pd.DataFrame:
    """Recorre el entorno con una política y devuelve la bitácora por barra.

    `politica` es cualquier `callable(obs) -> int`. Sirve igual para el DQN,
    para comprar y mantener y para la política aleatoria: así las tres se
    evalúan con exactamente el mismo código de contabilidad, que es la única
    forma de que la comparación signifique algo.
    """
    obs, _ = env.reset(seed=0)
    filas = []
    truncado = terminado = False
    while not (truncado or terminado):
        accion = int(politica(obs))
        obs, recompensa, terminado, truncado, info = env.step(accion)
        # `step` puede abarcar varias barras si `frecuencia_decision > 1`. La
        # bitácora siempre lleva una fila por barra: la recompensa se atribuye
        # a la barra de la decisión y las demás llevan 0, de modo que la suma
        # de la columna sigue siendo el retorno del episodio.
        for i, (fecha, r_activo, patrimonio) in enumerate(env.sub_barras):
            filas.append(
                {
                    "fecha": fecha,
                    "accion": accion,
                    "exposicion": info["exposicion"],
                    "patrimonio": patrimonio,
                    "ret_activo": r_activo,
                    "recompensa": recompensa if i == 0 else 0.0,
                }
            )
        filas[-1]["costos"] = info["costos"]
        filas[-1]["rotacion"] = info["rotacion"]

    bitacora = pd.DataFrame(filas).set_index("fecha")
    # Costos y rotación son acumulados: se propagan hacia adelante a las barras
    # intermedias, donde no hubo decisión.
    bitacora[["costos", "rotacion"]] = bitacora[["costos", "rotacion"]].ffill().fillna(0.0)
    equity = np.concatenate([[1.0], bitacora["patrimonio"].to_numpy()])
    bitacora["drawdown"] = (equity / np.maximum.accumulate(equity) - 1.0)[1:]
    return bitacora


def recorrido_completo(
    precios: pd.DataFrame, escalador: features.Escalador, cfg: ConfigEntorno
) -> EntornoInversion:
    """Entorno para evaluar: una sola pasada determinista por todo el tramo.

    Copia la configuración de entrenamiento y cambia **solo** el recorrido
    (una pasada completa, sin inicio aleatorio). Se construye con
    `dataclasses.replace` a propósito: la primera versión enumeraba los campos
    a mano y se dejó fuera `frecuencia_decision`, de modo que un agente
    entrenado para decidir cada semana se evaluaba decidiendo cada hora. El
    resultado parecía razonable —números plausibles, ninguna excepción— y
    arruinaba justamente el experimento que esa opción existía para hacer.
    Enumerar campos a mano en un constructor es una lista que se desactualiza;
    `replace` no puede olvidarse de uno.
    """
    cfg_eval = replace(
        cfg,
        largo_episodio=0,
        aleatorizar_inicio=False,
        semilla=0,
    )
    return EntornoInversion(precios, escalador, cfg_eval)
