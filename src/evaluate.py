"""Evaluación final en el período held-out. Se corre UNA vez.

Protocolo, en este orden y sin vuelta atrás:

1. Entre todas las corridas candidatas, se elige **una** por su retorno en
   validación. Esa elección se imprime y se guarda, con el número que la
   justificó, antes de tocar el tramo de prueba.
2. Ese único modelo se evalúa en modo greedy, de una sola pasada, sobre el
   tramo de prueba.
3. Se corren las líneas base sobre el mismo tramo, con el mismo código de
   contabilidad.
4. Se reportan retorno acumulado, máximo drawdown y costos de transacción, más
   las métricas de contexto, y se dibuja la curva de patrimonio.

Nada de lo que salga de aquí puede volver a `train.py` ni a `experiments.py`.
Si el resultado no gusta, el resultado es el que es: eso es lo que significa
tener un conjunto de prueba.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

import baselines
import data
import metrics
import plots
from agent import AgenteDQN
from environment import ConfigEntorno
from train import directorio_resultados, montar


class Comite:
    """Promedia los valores Q de varios agentes y actúa con el argmax.

    Por qué el comité es el modelo principal del informe
    ----------------------------------------------------
    En las fases 3 a 7 la dispersión entre semillas fue enorme: con idénticos
    hiperparámetros, dos semillas dieron +28 % y −76 % de retorno en validación.
    Con esa varianza, «escoger el mejor checkpoint de la mejor semilla» equivale
    a elegir el ganador de una lotería de ~75 candidatos (5 semillas × 15
    evaluaciones), y el ganador de una lotería no generaliza: es sobreajuste al
    conjunto de validación, aunque el de prueba siga intacto.

    El comité evita esa selección. Promedia Q(s,·) sobre las cinco semillas y
    actúa con el argmax del promedio, que es una decisión **única y tomada de
    antemano**, no una búsqueda. Es la versión de valores del promediado de
    modelos y reduce la varianza del estimador de Q sin tocar nada más.

    DECLARADO ANTES DE LEER EL TRAMO DE PRUEBA: el agente que el informe reporta
    como «DQN» es el comité. El mejor individuo por validación se reporta también,
    en su propia fila, para que se vea la diferencia — pero no es el resultado
    principal.
    """

    def __init__(self, agentes: list[AgenteDQN]) -> None:
        if not agentes:
            raise ValueError("un comité necesita al menos un agente")
        self.agentes = agentes

    def politica_greedy(self):
        def politica(obs: np.ndarray) -> int:
            q = np.mean([a.valores_q(obs) for a in self.agentes], axis=0)
            return int(np.argmax(q))

        return politica

    def tabla_q(self, estados: np.ndarray, exposiciones: tuple[float, ...]) -> np.ndarray:
        """El promedio de las tablas de los miembros — el comité, tabulado."""
        return np.mean([a.tabla_q(estados, exposiciones) for a in self.agentes], axis=0)


def cargar_comite(patron: str, mercado: str = "btc") -> tuple[Comite, list[str]]:
    """Carga el comité desde las corridas o los pesos incluidos del mercado.

    Los pesos de las corridas intermedias no se versionan (son muchos y pesan);
    los cinco del comité sí, en `comite/` dentro de la carpeta del mercado.
    Así alguien que clone el repositorio puede evaluar sin volver a entrenar.
    """
    resultados = directorio_resultados(mercado)
    corridas = resultados / "corridas"
    carpetas = [c for c in sorted(corridas.glob(patron)) if (c / "mejor.pt").exists()]
    if carpetas:
        agentes = [AgenteDQN.cargar(c / "mejor.pt") for c in carpetas]
        return Comite(agentes), [c.name for c in carpetas]

    guardados = sorted((resultados / "comite").glob("semilla_*.pt"))
    if not guardados:
        raise FileNotFoundError(
            f"ninguna corrida casa con '{patron}' en {corridas} "
            f"y {resultados / 'comite'} no contiene modelos de {mercado}"
        )
    return Comite([AgenteDQN.cargar(p) for p in guardados]), [p.stem for p in guardados]


def cargar_individual(carpeta: Path, cfg_corrida: dict, mejor_val: dict) -> AgenteDQN:
    """Recupera el individuo elegido, usando su copia del comité si hace falta."""
    ruta_corrida = carpeta / "mejor.pt"
    if ruta_corrida.exists():
        return AgenteDQN.cargar(ruta_corrida)

    resultados = directorio_resultados(cfg_corrida["mercado"])
    ruta_comite = resultados / "comite" / f"semilla_{cfg_corrida['semilla']}.pt"
    ruta_config = ruta_comite.with_suffix(".config.json")
    if not ruta_comite.exists() or not ruta_config.exists():
        raise FileNotFoundError(
            f"no existe {ruta_corrida}; para recuperar el individuo del comité "
            f"se necesitan {ruta_comite} y {ruta_config}"
        )
    cfg_comite = json.loads(ruta_config.read_text(encoding="utf-8"))
    if cfg_comite != cfg_corrida:
        raise ValueError(
            f"la configuración de {ruta_comite} no coincide con la corrida {carpeta.name}"
        )

    agente = AgenteDQN.cargar(ruta_comite)
    if asdict(agente.cfg) != cfg_corrida["agente"]:
        raise ValueError(f"los hiperparámetros de los pesos en {ruta_comite} no coinciden")
    if agente.pasos != mejor_val["paso"]:
        raise ValueError(
            f"el checkpoint de {ruta_comite} tiene {agente.pasos} pasos; "
            f"se esperaba el paso {mejor_val['paso']}"
        )
    print(f"Modelo individual recuperado desde {ruta_comite}")
    return agente


def seleccionar(patron: str, tramo: str = "prueba", mercado: str = "btc") -> tuple[Path, dict]:
    """Elige la corrida con mejor retorno en validación. Único criterio."""
    candidatos = []
    corridas = directorio_resultados(mercado) / "corridas"
    for carpeta in sorted(corridas.glob(patron)):
        archivo = carpeta / "mejor.json"
        if archivo.exists():
            d = json.loads(archivo.read_text(encoding="utf-8"))
            candidatos.append((carpeta, d))
    if not candidatos:
        raise FileNotFoundError(f"ninguna corrida casa con '{patron}' en {corridas}")

    candidatos.sort(key=lambda par: par[1]["retorno_acumulado"], reverse=True)
    print(f"Candidatos ({len(candidatos)}), ordenados por retorno en VALIDACIÓN:")
    for carpeta, d in candidatos:
        print(
            f"  {carpeta.name:24s} val {100 * d['retorno_acumulado']:+8.1f} %  "
            f"sharpe {d['sharpe']:+.2f}  dd {100 * d['max_drawdown']:6.1f} %  "
            f"(paso {d['paso']:,})"
        )
    elegido, mejor = candidatos[0]
    print(f"\n→ Seleccionado: {elegido.name}  (val {100 * mejor['retorno_acumulado']:+.1f} %)")
    if tramo == "prueba":
        print("  A partir de aquí se lee el tramo de PRUEBA por primera y única vez.\n")
    else:
        print(f"  Evaluación sobre el tramo de {tramo} (ensayo; no es la evaluación final).\n")
    return elegido, mejor


def evaluar(
    patron: str = "f7_final_s*",
    mercado: str = "btc",
    tramo: str = "prueba",
    semillas_aleatorias: int = 30,
    etiqueta_salida: str = "prueba",
) -> pd.DataFrame:
    resultados = directorio_resultados(mercado)
    carpeta, mejor_val = seleccionar(patron, tramo, mercado)
    cfg_corrida = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
    if cfg_corrida["mercado"] != mercado:
        raise ValueError(
            f"la corrida {carpeta.name} pertenece al mercado {cfg_corrida['mercado']}, "
            f"pero se solicitó evaluar {mercado}"
        )

    tramos, escalador = montar(mercado, incluir_prueba=True)
    precios = tramos[tramo]
    barras_anio = data.BARRAS_POR_ANIO[mercado]

    # Se reconstruye el entorno con la MISMA configuración con la que se
    # entrenó el modelo elegido —costo, recompensa, frecuencia de decisión—
    # salvo el recorrido, que aquí es la pasada completa y determinista.
    entorno_corrida = cfg_corrida["entorno"]
    cfg_env = ConfigEntorno(
        costo=entorno_corrida["costo"],
        recompensa=cfg_corrida["recompensa"],
        lambda_drawdown=entorno_corrida["lambda_drawdown"],
        eta_sharpe=entorno_corrida.get("eta_sharpe", 0.02),
        frecuencia_decision=entorno_corrida.get("frecuencia_decision", 1),
        largo_episodio=0,
        aleatorizar_inicio=False,
    )

    from environment import politica_tabulada, recorrido_completo, rodar

    comite, miembros = cargar_comite(patron, mercado)
    print(f"Comité de {len(miembros)} agentes: {', '.join(miembros)}")
    print("El 'DQN' que reporta el informe es el comité (decidido antes de leer prueba).\n")

    env = recorrido_completo(precios, escalador, cfg_env)
    bitacora_dqn = rodar(env, politica_tabulada(env, comite))

    agente_individual = cargar_individual(carpeta, cfg_corrida, mejor_val)
    env_ind = recorrido_completo(precios, escalador, cfg_env)
    bitacora_individual = rodar(env_ind, politica_tabulada(env_ind, agente_individual))

    # Las líneas base corren a SU frecuencia natural, no a la del agente.
    #
    # Comprar y mantener nunca cambia, así que le da igual. Pero el cruce de
    # medias es una regla que se evalúa en cada barra, y la política aleatoria
    # se define como actuar en cada barra: heredarles la frecuencia que el
    # agente eligió en validación las convertiría en estrategias distintas de
    # las que dicen ser, y le regalaría al agente una ventaja de costos que no
    # viene de su información sino de un parámetro que las líneas base no
    # tienen. Cada estrategia se evalúa como está definida.
    cfg_base = replace(cfg_env, frecuencia_decision=1)
    fijas, aleatorias = baselines.correr_todas(
        precios, escalador, cfg_base, semillas_aleatorias=semillas_aleatorias
    )

    resumenes = {
        "DQN": metrics.resumir(bitacora_dqn, barras_anio),
        "DQN (mejor semilla)": metrics.resumir(bitacora_individual, barras_anio),
    }
    for nombre, b in fijas.items():
        resumenes[nombre] = metrics.resumir(b, barras_anio)
    resumenes[f"Aleatoria ({semillas_aleatorias} semillas)"] = baselines.resumir_aleatorias(
        aleatorias, barras_anio
    )

    # Y, cuando el agente decide con menos frecuencia, se añade una aleatoria
    # igualada a su frecuencia. Separa las dos cosas que podrían explicar un
    # buen resultado: la información del agente, o simplemente operar menos.
    if cfg_env.frecuencia_decision != 1:
        _, aleatorias_igualadas = baselines.correr_todas(
            precios, escalador, cfg_env, semillas_aleatorias=semillas_aleatorias
        )
        resumenes[f"Aleatoria (cada {cfg_env.frecuencia_decision} barras)"] = (
            baselines.resumir_aleatorias(aleatorias_igualadas, barras_anio)
        )

    tabla = metrics.tabla(resumenes)
    legible = metrics.formatear(tabla)

    resultados.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(resultados / f"evaluacion_{etiqueta_salida}.csv")
    bitacora_dqn.to_csv(resultados / f"bitacora_dqn_{etiqueta_salida}.csv")
    (resultados / f"evaluacion_{etiqueta_salida}.json").write_text(
        json.dumps(
            {
                "mercado": mercado,
                "modelo_principal": "comité de los agentes de la fase final",
                "miembros_del_comite": miembros,
                "corrida_elegida_individual": carpeta.name,
                "criterio": "retorno acumulado en validación",
                "valor_en_validacion": mejor_val["retorno_acumulado"],
                "paso_del_checkpoint": mejor_val["paso"],
                "tramo_evaluado": tramo,
                "rango": [
                    str(precios.index[env.primer_paso].date()),
                    str(precios.index[-1].date()),
                ],
                "config": cfg_corrida["agente"],
                "recompensa": cfg_corrida["recompensa"],
                "bootstrap": cfg_corrida.get("bootstrap"),
                "metricas": resumenes,
            },
            indent=2,
            ensure_ascii=False,
            default=float,
        ),
        encoding="utf-8",
    )

    curvas = {"DQN": bitacora_dqn, "DQN (mejor semilla)": bitacora_individual, **fijas}
    sufijo_figura = etiqueta_salida if mercado == "btc" else f"{mercado}_{etiqueta_salida}"
    plots.curvas_de_patrimonio(
        {k: v for k, v in curvas.items() if k != "Todo en efectivo"},
        aleatorias,
        titulo=f"Patrimonio en el período de prueba ({precios.index[env.primer_paso].date()} → {precios.index[-1].date()})",
        nombre=f"equity_{sufijo_figura}.png",
    )
    plots.exposicion_y_precio(
        bitacora_dqn,
        precios["precio"],
        nombre=f"exposicion_{sufijo_figura}.png",
    )

    print(legible.to_string())
    print()
    reparto = bitacora_dqn["exposicion"].value_counts(normalize=True).sort_index()
    print("Reparto de exposición del DQN:")
    for nivel, frac in reparto.items():
        print(f"  {100 * nivel:5.0f} %  →  {100 * frac:5.1f} % de las barras")
    print(
        f"\nValores Q medios (mejor semilla): {np.round(valores_q_medios(agente_individual, env), 3)}"
    )
    return tabla


def valores_q_medios(agente: AgenteDQN, env) -> np.ndarray:
    """Q(s,·) promediado sobre todos los estados del tramo.

    Si los tres valores están pegados, la política es frágil: una perturbación
    pequeña cambia el argmax. Es el mismo diagnóstico que apareció en el
    proyecto de CarRacing.
    """
    estados = env.estados[env.primer_paso : env.ultimo_paso]
    filas = []
    for exposicion in (0.0, 0.5, 1.0):
        col = np.full((len(estados), 1), exposicion - 0.5)
        x = np.concatenate([estados, col], axis=1).astype(np.float32)
        import torch

        with torch.no_grad():
            filas.append(agente.q(torch.as_tensor(x)).numpy().mean(axis=0))
    return np.array(filas)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluación final en el período held-out")
    p.add_argument("--patron", default="f7_final_s*", help="glob de corridas candidatas")
    p.add_argument("--mercado", default="btc", choices=list(data.FUENTES))
    p.add_argument("--tramo", default="prueba", choices=["validacion", "prueba"])
    p.add_argument("--semillas-aleatorias", type=int, default=30)
    p.add_argument("--etiqueta", default=None)
    a = p.parse_args()
    evaluar(
        patron=a.patron,
        mercado=a.mercado,
        tramo=a.tramo,
        semillas_aleatorias=a.semillas_aleatorias,
        etiqueta_salida=a.etiqueta or a.tramo,
    )


if __name__ == "__main__":
    main()
