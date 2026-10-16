"""Q-Learning tabular — el método clásico de la comparación.

Por qué está, y por qué no puede ganar
---------------------------------------
El indicador de desempeño del curso pide entrenar agentes «desde métodos
clásicos hasta Deep Reinforcement Learning». Este módulo es el extremo
clásico, y su papel en el informe no es competir: es **hacer visible el motivo
por el que existe la aproximación de funciones**.

El estado del problema tiene 11 componentes continuas. Una tabla sobre ese
espacio es imposible: con apenas 10 cortes por dimensión serían 10¹¹ celdas,
y el tramo de entrenamiento tiene 20 875 barras. La fracción de la tabla que
se visitaría alguna vez es del orden de 10⁻⁷. Eso es la maldición de la
dimensionalidad, y no se arregla con más episodios.

Así que el agente tabular hace lo único que puede hacerse: **renunciar a casi
todo el estado**. Se queda con tres indicadores y la exposición, discretizados
en pocos cortes. La comparación con DQN y PPO deja de ser «tres algoritmos
sobre el mismo problema» y pasa a ser «qué se pierde al tener que resumir el
estado» — que es exactamente la transición que el curso quiere mostrar.

Qué tres indicadores, y por qué
-------------------------------
- `razon_vol`      : régimen de volatilidad. El EDA mostró que es lo único
                     predecible de la serie (autocorrelación de |r| de +0.29).
- `dist_sma_larga` : posición frente a la tendencia, el resumen de régimen más
                     usado en la práctica.
- `ret_medio`      : momentum a una semana.

Los cortes se calculan **solo con los cuantiles del tramo de entrenamiento** y
se congelan, igual que el escalador: usar los cuantiles de la serie completa
sería la misma fuga que normalizar con ella.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import features

# Los tres indicadores que sobreviven a la discretización, por su índice en
# `features.NOMBRES`.
INDICADORES = ("razon_vol", "dist_sma_larga", "ret_medio")
IDX = tuple(features.NOMBRES.index(n) for n in INDICADORES)


@dataclass
class ConfigTabular:
    """Hiperparámetros del agente tabular."""

    cortes: int = 4  # cuantiles por indicador → cortes**3 × 3 exposiciones estados
    gamma: float = 0.99
    lr: float = 0.1
    epsilon_inicial: float = 1.0
    epsilon_final: float = 0.05
    epsilon_pasos: int = 60_000
    optimista: float = 0.0  # valor inicial de la tabla
    semilla: int = 0

    def a_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


class AgenteTabular:
    """Q-Learning con tabla sobre un estado discretizado.

    La tabla se indexa por `(bin1, bin2, bin3, exposicion)` y guarda un valor
    por acción. Es Q-Learning fuera de política clásico:

        Q(s,a) ← Q(s,a) + α · [ r + γ·max_a' Q(s',a') − Q(s,a) ]

    Sin replay buffer y sin red objetivo: con una tabla no hacen falta, porque
    cada celda se actualiza de forma independiente y no hay aproximación que
    pueda divergir por interferencia entre estados.
    """

    def __init__(
        self,
        bordes: np.ndarray,
        n_acciones: int = 3,
        n_exposiciones: int = 3,
        cfg: ConfigTabular | None = None,
    ) -> None:
        self.cfg = cfg or ConfigTabular()
        self.bordes = np.asarray(bordes, dtype=np.float64)  # (3, cortes-1)
        self.n_acciones = n_acciones
        self.n_exposiciones = n_exposiciones
        forma = (self.cfg.cortes,) * len(IDX) + (n_exposiciones, n_acciones)
        self.q = np.full(forma, self.cfg.optimista, dtype=np.float64)
        self.visitas = np.zeros(forma[:-1], dtype=np.int64)
        self.rng = np.random.default_rng(self.cfg.semilla)
        self.pasos = 0

    # ------------------------------------------------------------ discretizar
    @classmethod
    def bordes_de_entrenamiento(
        cls, estados: np.ndarray, cortes: int = 4
    ) -> np.ndarray:
        """Cuantiles del tramo de entrenamiento, congelados.

        `estados` son los indicadores YA normalizados por el escalador, así que
        esto es consistente con lo que ve el resto del proyecto.
        """
        limpio = estados[np.isfinite(estados).all(axis=1)]
        probabilidades = np.linspace(0.0, 1.0, cortes + 1)[1:-1]
        return np.stack([np.quantile(limpio[:, i], probabilidades) for i in IDX])

    def _indice(self, obs: np.ndarray) -> tuple[int, ...]:
        celdas = tuple(
            int(np.digitize(obs[idx], self.bordes[k])) for k, idx in enumerate(IDX)
        )
        # La exposición es el último componente de la observación, desplazada
        # en −0.5 por el entorno: {0, 0.5, 1} → {−0.5, 0, 0.5} → {0, 1, 2}.
        exposicion = round((float(obs[-1]) + 0.5) * (self.n_exposiciones - 1))
        exposicion = min(max(exposicion, 0), self.n_exposiciones - 1)
        return (*celdas, exposicion)

    # --------------------------------------------------------------- política
    @property
    def epsilon(self) -> float:
        c = self.cfg
        if self.pasos >= c.epsilon_pasos:
            return c.epsilon_final
        avance = self.pasos / c.epsilon_pasos
        return c.epsilon_inicial + avance * (c.epsilon_final - c.epsilon_inicial)

    def actuar(self, obs: np.ndarray, *, deterministico: bool = False) -> int:
        if not deterministico and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_acciones))
        return int(np.argmax(self.q[self._indice(obs)]))

    def politica_greedy(self):
        return lambda obs: self.actuar(obs, deterministico=True)

    # ------------------------------------------------------------ aprendizaje
    def aprender(
        self,
        obs: np.ndarray,
        accion: int,
        recompensa: float,
        obs_siguiente: np.ndarray,
        terminado: bool,
    ) -> None:
        i = self._indice(obs)
        j = self._indice(obs_siguiente)
        self.visitas[i] += 1
        mejor_siguiente = 0.0 if terminado else float(np.max(self.q[j]))
        objetivo = recompensa + self.cfg.gamma * mejor_siguiente
        self.q[(*i, accion)] += self.cfg.lr * (objetivo - self.q[(*i, accion)])
        self.pasos += 1

    # ------------------------------------------------------- tabla para rodar
    def tabla_q(self, estados: np.ndarray, exposiciones: tuple[float, ...]) -> np.ndarray:
        """Misma interfaz que `AgenteDQN.tabla_q`, para que `politica_tabulada`
        y el comité funcionen igual con los tres algoritmos."""
        n = len(estados)
        salida = np.zeros((n, len(exposiciones), self.n_acciones), dtype=np.float64)
        celdas = np.stack(
            [np.digitize(estados[:, idx], self.bordes[k]) for k, idx in enumerate(IDX)]
        )
        for e_i in range(len(exposiciones)):
            salida[:, e_i, :] = self.q[(*tuple(celdas), e_i)]
        return salida

    # ------------------------------------------------------------ diagnóstico
    def cobertura(self) -> dict[str, float]:
        """Qué fracción de la tabla llegó a visitarse. Es el número que
        justifica por qué el método tabular no escala."""
        total = int(self.visitas.size)
        vistas = int((self.visitas > 0).sum())
        return {
            "celdas": float(total),
            "visitadas": float(vistas),
            "cobertura": vistas / total if total else 0.0,
            "visitas_medianas": float(np.median(self.visitas[self.visitas > 0]))
            if vistas
            else 0.0,
        }

    # ----------------------------------------------------------- persistencia
    def guardar(self, ruta: Path | str) -> None:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            ruta,
            q=self.q,
            visitas=self.visitas,
            bordes=self.bordes,
            cfg=json.dumps(asdict(self.cfg)),
        )

    @classmethod
    def cargar(cls, ruta: Path | str) -> AgenteTabular:
        d = np.load(Path(ruta), allow_pickle=False)
        cfg = ConfigTabular(**json.loads(str(d["cfg"])))
        agente = cls(d["bordes"], cfg=cfg)
        agente.q = d["q"]
        agente.visitas = d["visitas"]
        return agente


class ComiteTabular:
    """Promedio de las tablas de varias semillas, igual que el comité de DQN."""

    def __init__(self, agentes: list[AgenteTabular]) -> None:
        if not agentes:
            raise ValueError("un comité necesita al menos un agente")
        self.agentes = agentes

    def tabla_q(self, estados: np.ndarray, exposiciones: tuple[float, ...]) -> np.ndarray:
        return np.mean([a.tabla_q(estados, exposiciones) for a in self.agentes], axis=0)
