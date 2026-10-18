"""Encadena las fases 3 a 7 tomando cada decisión con el tramo de validación.

Cada fase elige su ganador por el **retorno medio en validación entre semillas**,
no por el máximo: el máximo entre semillas premia la suerte de inicialización y
es la forma más común de inventarse una mejora que no existe.

Al terminar deja en la carpeta del mercado la cadena completa de decisiones,
con los números que las justificaron, para que cualquiera pueda auditar que la
configuración final se eligió sin mirar el tramo de prueba.
"""

from __future__ import annotations

import json

import pandas as pd

import data
import experiments
import report
from train import directorio_resultados

CAPACIDADES = {
    "grande": {"oculta": 128, "capas": 2, "decaimiento_pesos": 1e-4},
    "mediana": {"oculta": 64, "capas": 2, "decaimiento_pesos": 1e-3},
    "pequena": {"oculta": 32, "capas": 1, "decaimiento_pesos": 1e-2},
    "minima": {"oculta": 16, "capas": 1, "decaimiento_pesos": 1e-2},
}


def _ganador(df: pd.DataFrame, por: str) -> tuple[str, pd.DataFrame]:
    agregado = (
        df.groupby(por)["val_retorno"]
        .agg(["mean", "std", "count", "max"])
        .sort_values("mean", ascending=False)
    )
    return str(agregado.index[0]), agregado


def correr(pasos: int = 150_000, mercado: str = "btc") -> dict:
    resultados = directorio_resultados(mercado)
    decisiones: dict = {"pasos_por_corrida": pasos, "mercado": mercado}

    print("\n########## FASE 3 — remuestreo + deriva aleatoria por episodio\n", flush=True)
    experiments.fase3_deriva(pasos=pasos, mercado=mercado)
    f3 = report.resumen_de_fase("f3_*", mercado=mercado)
    recompensa, tabla_r = _ganador(f3, "recompensa")
    print("\nRetorno medio en validación por recompensa:")
    print(tabla_r.to_string())
    print(f"\n→ Recompensa elegida: {recompensa}\n", flush=True)
    decisiones["fase3"] = {
        "elegido": recompensa,
        "criterio": "retorno medio en validación entre semillas",
        "tabla": json.loads(tabla_r.to_json(orient="index")),
    }

    print("\n########## FASE 4 — capacidad de la red\n", flush=True)
    experiments.fase4_capacidad(recompensa, pasos=pasos, mercado=mercado)
    f4 = report.resumen_de_fase("f4_*", mercado=mercado)
    f4 = f4.assign(variante=f4["corrida"].str.split("_").str[1])
    variante, tabla_c = _ganador(f4, "variante")
    print("\nRetorno medio en validación por capacidad:")
    print(tabla_c.to_string())
    print(f"\n→ Capacidad elegida: {variante} = {CAPACIDADES[variante]}\n", flush=True)
    decisiones["fase4"] = {
        "elegido": variante,
        "config": CAPACIDADES[variante],
        "criterio": "retorno medio en validación entre semillas",
        "tabla": json.loads(tabla_c.to_json(orient="index")),
    }

    print("\n########## FASE 5 — cinco semillas, decidiendo en cada barra\n", flush=True)
    experiments.fase5_final(recompensa, CAPACIDADES[variante], pasos=pasos, mercado=mercado)
    f5 = report.resumen_de_fase("f5_*", mercado=mercado)
    print(f5.to_string(index=False))
    decisiones["fase5"] = {
        "recompensa": recompensa,
        "config": CAPACIDADES[variante],
        "frecuencia_decision": 1,
        "semillas": f5["semilla"].tolist(),
        "val_retorno": f5["val_retorno"].tolist(),
        "val_retorno_medio": float(f5["val_retorno"].mean()),
        "val_rotacion_media": float(f5["val_rotacion"].mean()),
    }

    print("\n########## FASE 6 — cada cuánto decidir\n", flush=True)
    experiments.fase6_frecuencia(recompensa, CAPACIDADES[variante], pasos=pasos, mercado=mercado)
    f6 = report.resumen_de_fase("f6_*", mercado=mercado)
    f6 = f6.assign(frecuencia=f6["corrida"].str.extract(r"cada(\d+)")[0].astype(int))
    frecuencia, tabla_f = _ganador(f6, "frecuencia")
    frecuencia = int(frecuencia)
    print("\nRetorno medio en validación por frecuencia de decisión:")
    print(tabla_f.to_string())
    print(f"\n→ Frecuencia elegida: decidir cada {frecuencia} barra(s)\n", flush=True)
    decisiones["fase6"] = {
        "elegido": frecuencia,
        "criterio": "retorno medio en validación entre semillas",
        "tabla": json.loads(tabla_f.to_json(orient="index")),
    }

    print("\n########## FASE 7 — configuración final, cinco semillas\n", flush=True)
    experiments.fase7_final(
        recompensa, CAPACIDADES[variante], frecuencia, pasos=pasos, mercado=mercado
    )
    f7 = report.resumen_de_fase("f7_*", mercado=mercado)
    print(f7.to_string(index=False))
    decisiones["fase7"] = {
        "recompensa": recompensa,
        "config": CAPACIDADES[variante],
        "frecuencia_decision": frecuencia,
        "semillas": f7["semilla"].tolist(),
        "val_retorno": f7["val_retorno"].tolist(),
        "val_retorno_medio": float(f7["val_retorno"].mean()),
        "val_rotacion_media": float(f7["val_rotacion"].mean()),
    }

    resultados.mkdir(parents=True, exist_ok=True)
    (resultados / "decisiones.json").write_text(
        json.dumps(decisiones, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n→ {resultados / 'decisiones.json'}")
    print(
        f"\nEl tramo de prueba sigue sin leerse. Lo lee `python src/evaluate.py --mercado {mercado}`."
    )
    return decisiones


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Corre las fases 3 a 7 encadenadas")
    p.add_argument("--pasos", type=int, default=150_000)
    p.add_argument("--mercado", default="btc", choices=list(data.FUENTES))
    a = p.parse_args()
    correr(pasos=a.pasos, mercado=a.mercado)
