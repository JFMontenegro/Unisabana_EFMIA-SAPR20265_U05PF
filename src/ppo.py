"""PPO con Stable-Baselines3 — el segundo algoritmo profundo de la comparación.

Por qué PPO y no otra cosa
---------------------------
El temario del curso lo señala como «algoritmo de referencia en la industria y
base del entrenamiento de los grandes modelos de lenguaje actuales», y es la
contraparte natural de DQN: **aprende la política directamente** en vez de
estimar valores y tomar el `argmax`.

Esa diferencia importa aquí más de lo habitual. El diagnóstico de §7.4 del
informe es que los valores Q de las tres acciones caben en un rango del 6 %, y
que por tanto el `argmax` de DQN es un amplificador de ruido. Un método de
política no tiene ese problema: no hay `argmax` sobre estimaciones parecidas,
hay una distribución que se mueve por gradiente. Si la fragilidad del DQN
viniera de ahí, PPO debería notarse.

Se usa Stable-Baselines3 a propósito: el indicador del curso pide entrenar
«con frameworks del estado del arte», y además hace la comparación más
honesta — el DQN es implementación propia, así que si PPO ganara con una
implementación de referencia, la diferencia no podría atribuirse a un error
nuestro.

El entorno no necesitó ningún cambio: `check_env` de SB3 lo acepta tal cual
(`tests/test_ppo.py` lo verifica).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[1]


@dataclass
class ConfigPPO:
    """Hiperparámetros de PPO.

    Los valores son los de referencia de SB3 salvo dos: `n_steps` se alinea con
    el largo del episodio (720 barras) para que cada actualización vea
    trayectorias completas, y `ent_coef` sube un poco porque con tres acciones
    cuyos valores están muy cerca, una política que colapsa temprano a una sola
    acción no vuelve a explorar.
    """

    n_steps: int = 720
    lote: int = 180
    n_epocas: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    oculta: int = 32
    capas: int = 1
    recorte_gradiente: float = 0.5
    semilla: int = 0

    def a_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


class AgentePPO:
    """Envoltura de un modelo PPO de SB3 con la misma interfaz del resto.

    La clave es `tabla_q`: devuelve las **probabilidades** de cada acción para
    cada barra y cada exposición posible, con la misma forma que la tabla de
    valores del DQN. El `argmax` de esas probabilidades es exactamente la
    acción determinista de `model.predict(deterministic=True)` —
    `test_ppo.py` lo fija— así que `environment.politica_tabulada` y el comité
    funcionan igual con los tres algoritmos sin ninguna rama especial.
    """

    def __init__(self, modelo) -> None:
        self.modelo = modelo

    @classmethod
    def nuevo(cls, env, cfg: ConfigPPO) -> AgentePPO:
        from stable_baselines3 import PPO

        torch.set_num_threads(1)
        modelo = PPO(
            "MlpPolicy",
            env,
            n_steps=cfg.n_steps,
            batch_size=cfg.lote,
            n_epochs=cfg.n_epocas,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            learning_rate=cfg.lr,
            max_grad_norm=cfg.recorte_gradiente,
            policy_kwargs={"net_arch": [cfg.oculta] * cfg.capas},
            seed=cfg.semilla,
            verbose=0,
            device="cpu",
        )
        return cls(modelo)

    def tabla_q(self, estados: np.ndarray, exposiciones: tuple[float, ...]) -> np.ndarray:
        """Probabilidades de acción por barra y exposición, en una pasada.

        Las primeras barras de cada tramo son de calentamiento: sus
        indicadores todavía tienen NaN y nunca se operan. El DQN las atravesaba
        sin quejarse —una capa lineal propaga NaN en silencio— pero la
        `Categorical` de SB3 valida sus logits y lanza excepción. Eso es
        preferible: aquí se excluyen esas filas explícitamente y se devuelven
        como NaN, de modo que si alguna vez se intentara decidir sobre una,
        `environment.politica_tabulada` lo detecta en vez de elegir una acción
        a partir de basura.
        """
        n = len(estados)
        finitas = np.isfinite(estados).all(axis=1)
        politica = self.modelo.policy
        politica.set_training_mode(False)

        salida = np.full((n, len(exposiciones), self.modelo.action_space.n), np.nan)
        if not finitas.any():
            return salida

        with torch.no_grad():
            for i, e in enumerate(exposiciones):
                col = np.full((int(finitas.sum()), 1), e - 0.5, dtype=np.float64)
                x = np.concatenate([estados[finitas], col], axis=1).astype(np.float32)
                obs = torch.as_tensor(x, device=politica.device)
                distribucion = politica.get_distribution(obs)
                salida[finitas, i, :] = distribucion.distribution.probs.cpu().numpy()
        return salida

    def politica_greedy(self):
        def politica(obs: np.ndarray) -> int:
            accion, _ = self.modelo.predict(obs, deterministic=True)
            return int(accion)

        return politica

    def guardar(self, ruta: Path | str) -> None:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        self.modelo.save(ruta)

    @classmethod
    def cargar(cls, ruta: Path | str) -> AgentePPO:
        from stable_baselines3 import PPO

        return cls(PPO.load(Path(ruta), device="cpu"))


class ComitePPO:
    """Promedio de las distribuciones de política de varias semillas.

    Es el análogo exacto del comité de DQN: allá se promedian valores y aquí
    probabilidades. En los dos casos la decisión es única y declarada de
    antemano, no una búsqueda entre semillas.
    """

    def __init__(self, agentes: list[AgentePPO]) -> None:
        if not agentes:
            raise ValueError("un comité necesita al menos un agente")
        self.agentes = agentes

    def tabla_q(self, estados: np.ndarray, exposiciones: tuple[float, ...]) -> np.ndarray:
        return np.mean([a.tabla_q(estados, exposiciones) for a in self.agentes], axis=0)
