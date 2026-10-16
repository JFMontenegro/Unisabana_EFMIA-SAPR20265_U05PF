"""DQN propio en PyTorch, adaptado del agente que resolvió CarRacing-v3.

Qué se conserva del proyecto anterior y qué cambia
--------------------------------------------------
Se conserva la maquinaria, que es la parte difícil y ya está probada: replay
buffer, red objetivo, Double DQN, pérdida de Huber, ε-greedy con decaimiento
lineal, semillas fijas en todas las fuentes de aleatoriedad.

Cambia la red: allá eran tres capas convolucionales sobre 4 fotogramas de
84×84; aquí el estado es un vector de 11 números, así que es un perceptrón de
dos capas ocultas. 11 → 128 → 128 → 3. Meter una red grande sobre 11 entradas
y ~5 800 barras de entrenamiento es la forma más rápida de memorizar el pasado
del Brent, que es exactamente lo que el período de prueba va a castigar.

Un detalle que en el proyecto anterior fue un error real
--------------------------------------------------------
En CarRacing, al truncar un episodio el buffer guardaba `s` como `s'` con
`terminal = 0`, así que Bellman bootstrapeaba desde el estado equivocado. Lo
encontró un compañero del grupo revisando el repo. Aquí el problema es más
agudo porque **todos** los episodios truncan: el mercado no tiene estado
absorbente, un episodio termina porque se acabó la ventana de 252 barras, no
porque la historia haya concluido. Por eso:

- `terminado` siempre es `False` y el objetivo siempre bootstrapea;
- el estado siguiente que se guarda es el real, no el actual;
- `tests/test_agente.py` lo verifica explícitamente.

Tratar la truncación como terminal le enseñaría al agente que el mundo se
acaba cada 252 barras y que por tanto no importa en qué posición lo deja.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class Config:
    """Hiperparámetros. Todo lo que se busca en validación está aquí."""

    # red
    oculta: int = 128
    capas: int = 2
    # aprendizaje
    gamma: float = 0.99
    lr: float = 5e-4
    # Decaimiento de pesos desacoplado (AdamW). No es higiene opcional en este
    # problema: con 11 indicadores y ~5 800 barras de entrenamiento, los
    # indicadores forman una huella casi única de cada fecha y la red memoriza
    # el pasado del Brent en pocos miles de pasos. El efecto se mide en el
    # README §6.
    decaimiento_pesos: float = 1e-4
    lote: int = 64
    # replay
    capacidad: int = 200_000
    calentamiento: int = 5_000
    actualizar_cada: int = 1  # pasos del entorno por paso de gradiente
    objetivo_cada: int = 1_000  # pasos de gradiente entre copias a la red objetivo
    # exploración
    epsilon_inicial: float = 1.0
    epsilon_final: float = 0.05
    epsilon_pasos: int = 60_000
    # estabilidad
    doble_dqn: bool = True
    recorte_gradiente: float = 10.0
    semilla: int = 0

    def a_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


class RedQ(nn.Module):
    """Perceptrón que estima Q(s, ·) para las tres exposiciones."""

    def __init__(self, dim_estado: int, n_acciones: int, oculta: int, capas: int) -> None:
        super().__init__()
        bloques: list[nn.Module] = []
        entrada = dim_estado
        for _ in range(capas):
            bloques += [nn.Linear(entrada, oculta), nn.ReLU()]
            entrada = oculta
        bloques.append(nn.Linear(entrada, n_acciones))
        self.net = nn.Sequential(*bloques)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Replay:
    """Buffer circular de transiciones. Vectores, no fotogramas: cabe entero."""

    def __init__(self, capacidad: int, dim_estado: int, semilla: int = 0) -> None:
        self.capacidad = capacidad
        self.estados = np.zeros((capacidad, dim_estado), dtype=np.float32)
        self.siguientes = np.zeros((capacidad, dim_estado), dtype=np.float32)
        self.acciones = np.zeros(capacidad, dtype=np.int64)
        self.recompensas = np.zeros(capacidad, dtype=np.float32)
        self.terminales = np.zeros(capacidad, dtype=np.float32)
        self.pos = 0
        self.llenos = 0
        self.rng = np.random.default_rng(semilla)

    def guardar(
        self,
        estado: np.ndarray,
        accion: int,
        recompensa: float,
        siguiente: np.ndarray,
        terminado: bool,
    ) -> None:
        if estado.shape != self.estados.shape[1:]:
            raise ValueError(f"estado con forma {estado.shape}, se esperaba {self.estados.shape[1:]}")
        if siguiente.shape != estado.shape:
            raise ValueError("el estado siguiente no tiene la forma del estado")
        if not np.isfinite(estado).all() or not np.isfinite(siguiente).all():
            raise ValueError("estado con NaN o infinito; el entorno está mal")
        i = self.pos
        self.estados[i] = estado
        self.siguientes[i] = siguiente
        self.acciones[i] = accion
        self.recompensas[i] = recompensa
        self.terminales[i] = float(terminado)
        self.pos = (self.pos + 1) % self.capacidad
        self.llenos = min(self.llenos + 1, self.capacidad)

    def muestrear(self, lote: int) -> tuple[np.ndarray, ...]:
        idx = self.rng.integers(0, self.llenos, size=lote)
        return (
            self.estados[idx],
            self.acciones[idx],
            self.recompensas[idx],
            self.siguientes[idx],
            self.terminales[idx],
        )

    def __len__(self) -> int:
        return self.llenos


class AgenteDQN:
    """DQN con red objetivo y Double DQN opcional."""

    def __init__(self, dim_estado: int, n_acciones: int, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.dim_estado = dim_estado
        self.n_acciones = n_acciones

        torch.manual_seed(self.cfg.semilla)
        self.rng = np.random.default_rng(self.cfg.semilla)

        self.q = RedQ(dim_estado, n_acciones, self.cfg.oculta, self.cfg.capas)
        self.objetivo = RedQ(dim_estado, n_acciones, self.cfg.oculta, self.cfg.capas)
        self.objetivo.load_state_dict(self.q.state_dict())
        self.objetivo.eval()

        self.opt = torch.optim.AdamW(
            self.q.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.decaimiento_pesos
        )
        self.perdida_fn = nn.SmoothL1Loss()
        self.replay = Replay(self.cfg.capacidad, dim_estado, self.cfg.semilla)

        self.pasos = 0
        self.actualizaciones = 0

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
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            return int(self.q(x).argmax(dim=1).item())

    def politica_greedy(self):
        """Devuelve un `callable(obs) -> int` para `environment.rodar`."""
        return lambda obs: self.actuar(obs, deterministico=True)

    # ------------------------------------------------------------ aprendizaje
    def recordar(self, *args) -> None:
        self.replay.guardar(*args)
        self.pasos += 1

    def aprender(self) -> float | None:
        if len(self.replay) < self.cfg.calentamiento:
            return None
        if self.pasos % self.cfg.actualizar_cada != 0:
            return None

        s, a, r, s2, fin = self.replay.muestrear(self.cfg.lote)
        s_t = torch.as_tensor(s)
        s2_t = torch.as_tensor(s2)
        a_t = torch.as_tensor(a).unsqueeze(1)
        r_t = torch.as_tensor(r)
        fin_t = torch.as_tensor(fin)

        q_actual = self.q(s_t).gather(1, a_t).squeeze(1)

        with torch.no_grad():
            if self.cfg.doble_dqn:
                # La red en línea elige la acción, la objetivo la valora. Esto
                # corta el sesgo de maximización: el max de estimaciones
                # ruidosas sobreestima el valor verdadero (desigualdad de
                # Jensen), y el DQN clásico usa la misma red para elegir y
                # para valorar, así que se cree su propio ruido.
                mejor = self.q(s2_t).argmax(dim=1, keepdim=True)
                q_siguiente = self.objetivo(s2_t).gather(1, mejor).squeeze(1)
            else:
                q_siguiente = self.objetivo(s2_t).max(dim=1).values
            objetivo = r_t + self.cfg.gamma * (1.0 - fin_t) * q_siguiente

        perdida = self.perdida_fn(q_actual, objetivo)
        self.opt.zero_grad(set_to_none=True)
        perdida.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.recorte_gradiente)
        self.opt.step()

        self.actualizaciones += 1
        if self.actualizaciones % self.cfg.objetivo_cada == 0:
            self.objetivo.load_state_dict(self.q.state_dict())
        return float(perdida.item())

    # ------------------------------------------------------------ persistencia
    def guardar(self, ruta: Path | str) -> None:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q": self.q.state_dict(),
                "cfg": asdict(self.cfg),
                "dim_estado": self.dim_estado,
                "n_acciones": self.n_acciones,
                "pasos": self.pasos,
            },
            ruta,
        )

    @classmethod
    def cargar(cls, ruta: Path | str) -> AgenteDQN:
        punto = torch.load(ruta, map_location="cpu", weights_only=False)
        cfg = Config(**punto["cfg"])
        agente = cls(punto["dim_estado"], punto["n_acciones"], cfg)
        agente.q.load_state_dict(punto["q"])
        agente.objetivo.load_state_dict(punto["q"])
        agente.q.eval()
        agente.pasos = punto.get("pasos", 0)
        return agente

    def valores_q(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            return self.q(x).squeeze(0).numpy()

    def tabla_q(self, estados: np.ndarray, exposiciones: tuple[float, ...]) -> np.ndarray:
        """Q(s, ·) para cada barra y cada exposición posible, de una pasada.

        Evaluar la política barra por barra con `batch=1` es correcto pero
        absurdamente lento en una red de 243 parámetros: el costo es la llamada
        a PyTorch, no la multiplicación. Como el estado es
        `(indicadores_t, exposición)` y la exposición solo puede tomar tres
        valores, toda la tabla cabe en un solo lote de `3·n` filas.

        Devuelve un arreglo `(n, |exposiciones|, n_acciones)`. El resultado es
        idéntico al de llamar `valores_q` barra por barra —
        `test_la_tabla_q_coincide_con_la_evaluacion_barra_a_barra` lo fija— pero
        cuesta dos órdenes de magnitud menos.
        """
        n = len(estados)
        filas = []
        with torch.no_grad():
            for e in exposiciones:
                col = np.full((n, 1), e - 0.5, dtype=np.float64)
                x = np.concatenate([estados, col], axis=1).astype(np.float32)
                filas.append(self.q(torch.as_tensor(x)).numpy())
        return np.stack(filas, axis=1)
