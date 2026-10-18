# Agente Inversor

**Grupo 8** · Proyecto final · Simulación y Aprendizaje por Refuerzo · Maestría en Inteligencia Artificial, Universidad de La Sabana

> ¿Puede un agente de aprendizaje por refuerzo aprender una estrategia de inversión que se comporte mejor que estrategias simples cuando se enfrenta a datos que no ha visto durante su entrenamiento?

**Respuesta corta: en BTC no, en Brent sí — y la diferencia entre los dos casos resultó ser el hallazgo más útil del proyecto.**

| Período held-out | DQN (comité) | Comprar y mantener | Diferencia |
|---|---|---|---|
| **BTC/USD horario**, jun-2021 → mar-2022 | **−10.7 %** (Sharpe −0.05) | +14.1 % (Sharpe 0.60) | pierde |
| **Brent diario**, 2019 → 2026 (apéndice A) | **+147.5 %** (Sharpe 0.52, drawdown −57.5 %) | +131.9 % (Sharpe 0.48, drawdown −87.8 %) | gana |

Mismo código, mismo protocolo, misma arquitectura. Lo único que cambia es el mercado — y, con él, **la frecuencia de decisión que el criterio de selección eligió**: cada 5 barras en Brent (12 operaciones al año, 1.4 % anual en costos) y cada barra en BTC (155 operaciones al año, **14 % anual en costos**).

Los cuatro hallazgos que sostienen el informe:

1. **El agente memoriza la serie histórica.** Entrenando sobre los precios reales alcanzó un patrimonio de entrenamiento de **6 319 veces el capital** en BTC y **10¹⁴ veces** en Brent, con validación que no lo acompañaba. No aprendió a invertir: aprendió un calendario. El remuestreo por bloques lo devolvió a menos de 1×.
2. **El criterio de selección que declaré es ciego al costo cuando el tramo de validación tiene deriva fuerte.** La validación de BTC subió 172 % en siete meses; ahí, 45 % anual en comisiones es invisible. El criterio eligió decidir cada hora frente a cada día **por 0.6 puntos porcentuales**, con cinco veces más varianza y **ocho veces más rotación**. En el período de prueba, que sube 14 %, esa decisión se cobró sola.
3. **La política no tiene señal, en ninguno de los dos mercados.** En BTC la correlación entre exposición y retorno siguiente fue **−0.003**, y la volatilidad del activo cuando el agente estaba invertido (0.00752) fue indistinguible de cuando estaba en efectivo (0.00746): **no hizo timing de volatilidad**, que es justamente lo único predecible de la serie. El Brent gana por tener menos costos y menos exposición al peor tramo, no por acertar.
4. **La dispersión entre semillas supera el efecto de toda decisión de diseño.** Por eso, antes de leer el test, se declaró que el agente reportado sería un comité de cinco semillas. En Brent el comité rindió **+141 puntos** más que el individuo elegido por validación; en BTC rindió **6 puntos menos**. La conclusión honesta no es «el comité es mejor» sino que **con esta varianza ningún método de selección es confiable**, y reportar un solo número sin su dispersión es engañarse.

### Entregables

| Qué | Dónde |
|---|---|
| **Informe** (documenta los 11 puntos del §17 y responde las 34 preguntas numeradas del enunciado) | `Informe_Agente_Inversor_Grupo8.docx` |
| **Código reproducible** del agente y del entorno, con semillas fijas e instrucciones | este repositorio — §10 |
| Análisis exploratorio ejecutado | `notebooks/eda.ipynb` |
| Bitácoras, decisiones y modelos del comité | `results/` y `results_brent/` |

Este README es la documentación técnica del repositorio. **Las 34 preguntas numeradas del enunciado se responden en el informe**, señaladas con su número.

---

## 1. El problema, formulado como un MDP

| Elemento | Definición |
|---|---|
| **Estado** `s_t` | 10 indicadores causales calculados con precios hasta `P_t`, estandarizados con medias y desviaciones del tramo de entrenamiento, más la exposición vigente. 11 componentes. |
| **Acción** `a_t` | Fracción del capital expuesta al activo: **{0 %, 50 %, 100 %}**. El resto queda en efectivo al 0 %. |
| **Transición** | Determinista dada la serie: `t → t+1`. El agente no mueve el mercado. |
| **Recompensa** `r_{t+1}` | Tres variantes, comparadas en §5. La principal es `log(1 + r^port)`. |
| **Retorno del portafolio** | `r^port_{t+1} = (1 − c·\|a_t − a_{t−1}\|)·(1 + a_t·r_{t+1}) − 1`, con `c = 10 bps` |
| **Horizonte** | Episodios de 720 barras (un mes) durante el entrenamiento; una pasada completa en evaluación. |

### Los 10 indicadores del estado

Los mismos en los dos mercados; lo que cambia son las ventanas, porque «21 barras» es un mes en datos diarios y menos de un día en datos horarios.

| Indicador | Qué mide | Ventana (BTC horario) | Ventana (Brent diario) |
|---|---|---|---|
| `ret_1`, `ret_corto`, `ret_medio` | Log-retornos a tres horizontes | 1 h · 1 día · 1 semana | 1 d · 1 semana · 1 mes |
| `vol_corta` | Volatilidad realizada reciente | 24 h | 21 d |
| `razon_vol` | `log(vol_corta / vol_larga)` — régimen relativo | 24 h / 720 h | 21 d / 252 d |
| `dist_sma_corta`, `dist_sma_larga` | `log P − log SMA` — posición frente a la tendencia | 24 h · 168 h | 21 d · 63 d |
| `rsi` | RSI de Wilder, centrado | 14 barras | 14 barras |
| `drawdown` | `log(P / máximo móvil)` | 720 h | 252 d |
| `pos_rango` | Posición dentro del rango de la ventana larga | 720 h | 252 d |

Todos son adimensionales o están en unidades de retorno: el BTC pasó de 3 100 a 68 600 USD y un estado que dependiera del nivel no se podría comparar entre épocas.

**La exposición vigente es parte del estado y no es opcional.** Sin ella el problema deja de ser markoviano: el costo de elegir 100 % depende de si ya se está en 100 %, y un agente que no observa su propia posición no puede distinguir «mantener gratis» de «entrar pagando».

**El volumen no entra en el estado.** Es deliberado: mantiene el estado idéntico entre los dos mercados, de modo que las conclusiones no dependan de qué información extra tenga uno de ellos. El cargador sí deja el volumen disponible, corregido (§2.1).

---

## 2. Los datos

**Mercado principal: BTC/USD horario**, el archivo `examples/data/BTC_USD-Hourly.csv` del repositorio de [**Gym Trading Env**](https://gym-trading-env.readthedocs.io/en/latest/download.html) — la librería que el profesor señaló como fuente de datos — y el mismo archivo que usa su ejemplo oficial.

- **33 259 barras horarias**, del 15-may-2018 al 1-mar-2022
- Volatilidad anualizada en entrenamiento: **72.5 %**; curtosis en exceso **75.4**
- Precio entre 3 140 y 68 627 USD

### Por qué este archivo y no el descargador

El profesor señaló `gym_trading_env.downloader.download()`, que baja OHLCV de Binance, Bitfinex o Huobi vía ccxt. **Esas APIs no son alcanzables desde el entorno donde se corrió este proyecto**: la red de egreso las bloquea, igual que a Yahoo Finance o stooq. El CSV que la propia librería distribuye es la misma clase de dato —OHLCV horario de BTC, del mismo proyecto— y se baja con un script reproducible.

`data.cargar_pickle_gte()` lee los pickles que produce `download()`, así que cambiar de fuente es una línea:

```python
from data import cargar_pickle_gte
mercado = cargar_pickle_gte("data/binance-BTCUSDT-1h.pkl")
```

### 2.1 Un defecto real del dataset

**En `BTC_USD-Hourly.csv`, hasta el 12-oct-2020 las columnas `volume` y `Volume USD` están intercambiadas**: la que dice «volume» trae los dólares y la que dice «Volume USD» trae los bitcoins. Son **20 143 de 33 259 filas, el 60.6 %**.

Se detecta con una identidad que tiene que cumplirse siempre:

```
Volume USD  ≈  volume × precio
```

El ejemplo oficial de la librería construye su variable de volumen a partir de `Volume USD`, así que un agente entrenado tal cual recibe una señal que **cambia de significado a mitad de la serie** — y en este proyecto la frontera cae casi exactamente entre el tramo de entrenamiento (termina el 31-oct-2020) y el de validación. Sería una fuente de resultados inexplicables muy difícil de rastrear.

`data._revisar_volumen()` lo corrige fila por fila y deja constancia en los avisos del cargador; `tests/test_sin_fuga.py` lo verifica. Tras corregir, la identidad se cumple con una mediana de error del **0.04 %** y el 99.8 % de las barras dentro del 2 %. El resto son horas de mucho movimiento, donde el volumen en dólares se calcula con un precio medio de la hora y no con el cierre — eso no es un defecto, es la definición de la columna.

### Partición temporal — congelada antes de mirar nada

```python
# src/data.py
PARTICION["btc"] = {
    "entrenamiento": ("2018-05-15", "2020-10-31"),   # 20 875 barras operables · 65 %
    "validacion":    ("2020-11-01", "2021-05-31"),   #  5 065 barras operables · 15 %
    "prueba":        ("2021-06-01", "2022-12-31"),   #  6 553 barras operables · 20 %
}
```

Está escrita en el módulo de datos, no en un script de entrenamiento, precisamente para que ningún experimento pueda moverla «solo esta vez». Cada tramo arranca 720 barras antes de su fecha de inicio para calentar los indicadores; esas barras **no se operan**, y `tests/test_sin_fuga.py` verifica que ninguna barra operable aparezca en dos tramos.

### Los tres tramos son tres mercados distintos

| Tramo | Barras | Precio | Retorno | Vol. anualizada | Curtosis |
|---|---|---|---|---|---|
| Entrenamiento (jun-2018 → oct-2020) | 20 875 | 8 741 → 13 816 | +58 % | 72.1 % | 74.8 |
| Validación (nov-2020 → may-2021) | 5 065 | 13 734 → 37 341 | **+172 %** | 98.6 % | 13.6 |
| **Prueba (jun-2021 → mar-2022)** | 6 553 | 37 821 → 43 312 | **+14.5 %** | 70.1 % | 6.5 |

Dos cosas que hay que declarar antes de leer cualquier resultado:

1. **El tramo de validación es un mercado alcista extremo** (+172 % en siete meses). Seleccionar modelo ahí **premia estar invertido**. No se eligió por eso —se eligió por orden cronológico, antes de mirar nada— pero el sesgo existe y condiciona la interpretación de §6.
2. **El tramo de prueba es un viaje de ida y vuelta**: de 37 800 a 69 000 USD y de vuelta a 35 000, para cerrar en 43 300. Termina en +14.5 %, así que **comprar y mantener es batible**. Es lo que hace que el período de prueba sea una prueba de verdad y no un trámite — al contrario de lo que pasó en la réplica sobre el Brent (apéndice A), donde el tramo de prueba subió 132 % y la línea base era casi imbatible.

### Réplica de robustez

El repositorio soporta dos mercados más, con el mismo código: **Brent diario** (EIA, 9 102 barras desde 1987) y **S&P 500 mensual** (Shiller, 1871-2026). El experimento completo se corrió también sobre el Brent; los resultados están en el **apéndice A** y sirven para separar lo que es propio de este activo de lo que es propio del método.

---

## 3. El entorno, y la fuga de información

`src/environment.py` implementa la API de Gymnasium (`reset`, `step`, `observation_space`, `action_space`).

### La frontera de la causalidad

El estado en `t` se construye con `P_0..P_t`. La recompensa usa `P_{t+1}`. Esa línea es todo el proyecto: si se cruza, los resultados no valen nada y el error no aparece en ninguna gráfica — solo en un rendimiento sospechosamente bueno.

Por eso la regla no se declara, **se prueba**:

```python
# tests/test_sin_fuga.py — se corre sobre los dos mercados
truncados = features.construir(precios.iloc[:t + 1], ventanas)   # serie cortada en t
np.testing.assert_allclose(truncados.iloc[-1], completos.iloc[t])
```

Si alguien introduce un `rolling(center=True)`, un `bfill()` o un `pct_change(-1)`, esa igualdad se rompe y el test falla. Hay además una prueba de contaminación: duplicar todos los precios **futuros** no puede cambiar la observación presente, solo la recompensa.

**La normalización también puede filtrar.** Las medias y desviaciones se estiman solo en el tramo de entrenamiento y se congelan; estimarlas sobre la serie completa metería la volatilidad de 2021 dentro del estado de 2019. `test_el_escalador_solo_ve_entrenamiento` lo verifica en los dos mercados.

**Y las ventanas pueden desajustarse.** Un escalador ajustado con ventanas horarias alimentando indicadores diarios sería un error silencioso. El entorno toma las ventanas **del escalador**, no de un parámetro aparte, para que el desajuste sea imposible por construcción.

### Costos de transacción

`c = 10 bps` sobre la **rotación**, `|a_t − a_{t−1}|`, no sobre la posición. Es además la comisión *taker* estándar de Binance en spot (0.1 %), así que la cifra del enunciado resulta ser la realista para este activo.

- Cobrarlo sobre `a_t` sería un impuesto a estar invertido, y el agente aprendería a no invertir.
- Cobrarlo solo al abrir y no al cerrar dejaría media operación gratis.
- Pasar de 0 % a 100 % cuesta 10 bps del capital; volver cuesta otros 10.

El costo se paga **antes** de que el retorno se acumule: `E_{t+1} = E_t·(1 − c·|Δa|)·(1 + a_t·r_{t+1})`. La diferencia con la formulación habitual `E_t·(1 + a·r − c)` es de segundo orden (≈1e-6) pero deja un invariante exacto que se puede testear: comprar y mantener da justo `(P_T/P_0)·(1 − c)`.

### En datos horarios el costo no es un detalle, es el problema

Esta cuenta está hecha en el notebook de EDA **antes de entrenar**, y es la que motiva la fase 6:

| Decidir | Operaciones/año si cambia la mitad de las veces | Costo anual |
|---|---|---|
| Cada hora | 4 380 | **438 % del capital** |
| Cada día | 182 | 18.2 % |
| Cada semana | 26 | 2.6 % |

El BTC rindió 37 % anualizado en el tramo de entrenamiento. Un agente que decida cada hora y cambie la mitad de las veces pierde **doce veces** ese retorno en comisiones antes de acertar una sola operación.

### La señal existe, y es más pequeña que el costo

| | |
|---|---|
| Autocorrelación de `r` a 1 hora | **−0.0365** (banda de ruido ±0.0138) |
| Ventaja esperada por operación (`\|ρ\|·σ`) | **2.83 bps** |
| Costo de ida y vuelta | **20 bps** |
| Razón costo / ventaja | **7.1×** |

La reversión a una hora es estadísticamente real —está fuera de la banda de ruido— y es **siete veces más pequeña que lo que cuesta usarla**. Ese es el problema del proyecto en una línea.

Lo que sí es grande es la persistencia de la **volatilidad**: autocorrelación de `|r|` de **+0.29 a una hora** y todavía **+0.16 un día después**. Por eso el espacio de acciones es de exposición y no de dirección: lo predecible es cuánto riesgo hay, no hacia dónde va el precio.

### Truncación, y un error heredado

Ningún episodio termina: el mercado no tiene estado absorbente. Un episodio acaba porque se agotó la ventana de 720 barras, no porque la historia haya concluido. Por lo tanto `terminado` siempre es `False` y el objetivo de Bellman **siempre** bootstrapea.

Esto importa porque en el proyecto anterior del grupo (DQN sobre CarRacing-v3) había exactamente ese error: al truncar, el buffer guardaba `s` como `s'`. Lo encontró un compañero revisando el repositorio. Aquí el error sería peor, porque afectaría a *todas* las transiciones finales. `test_la_truncacion_bootstrapea_con_el_estado_siguiente_real` lo cubre.

---

## 4. El agente

DQN propio en PyTorch, sin Stable-Baselines3. Reutiliza la maquinaria del proyecto de CarRacing —replay buffer, red objetivo, Double DQN, pérdida de Huber, ε-greedy lineal, semillas fijas— y cambia la red: allá eran tres capas convolucionales sobre fotogramas de 84×84, aquí es un perceptrón sobre un vector de 11 números.

| | |
|---|---|
| Arquitectura | MLP `11 → h → h → 3` |
| Optimizador | AdamW (decaimiento de pesos desacoplado) |
| Double DQN | sí — la red en línea elige la acción, la objetivo la valora |
| Pérdida | Huber (`SmoothL1`) |
| γ | 0.99 |
| Replay | 200 000 transiciones, calentamiento 5 000 |
| Red objetivo | copia cada 1 000 actualizaciones |
| ε | 1.0 → 0.05 en 60 000 pasos |
| Escala de recompensa | ×100 |
| Pasos por corrida | 120 000 |
| **Criterio de parada** | Presupuesto fijo de 120 000 pasos, conservando el checkpoint con mejor retorno **en validación** (evaluado cada 10 000 pasos). Equivale a parada temprana sobre validación, nunca sobre prueba. |
| Semillas | 0, 1, 2, 3, 4 |

**Por qué Double DQN y no DQN clásico:** el `max` sobre estimaciones ruidosas sobreestima el valor verdadero (desigualdad de Jensen), y el DQN clásico usa la misma red para elegir y para valorar, así que se cree su propio ruido. Con tres acciones cuyos valores verdaderos están muy cerca entre sí —que es el caso aquí, ver §7.4— ese sesgo es suficiente para invertir el orden del `argmax`.

### Una optimización que había que probar

Evaluar la política barra por barra con lotes de tamaño 1 es correcto y absurdamente lento: en una red de cientos de parámetros el costo es la llamada a PyTorch, no la multiplicación. Como el estado es `(indicadores_t, exposición)` y la exposición solo toma tres valores, **toda la tabla Q cabe en un solo lote de `3n` filas**: `agent.tabla_q()` la calcula de una pasada y `environment.politica_tabulada()` decide indexando.

Una optimización que cambiara la política en silencio invalidaría todos los números del informe, así que `tests/test_evaluacion_rapida.py` fija la equivalencia: misma secuencia de acciones y mismo patrimonio, barra por barra, con decisión horaria y semanal. Y como la diferencia numérica en float32 llega a 2e-5 relativo, un test aparte verifica que el margen entre la mejor y la segunda acción sea mil veces mayor que ese error.

---

## 5. Iteración de la función de recompensa

Se diseñaron y compararon **tres** recompensas, con idénticos hiperparámetros y semillas: lo único que cambia es la señal.

### 5.1 `log_neto` — el objetivo declarado

```
r_t = log(1 + r^port_t)
```

Tiene una propiedad que la hace la recompensa «correcta»: **la suma de recompensas del episodio es exactamente el log del patrimonio final**, así que maximizar el retorno descontado con γ→1 es maximizar el objetivo del problema. `test_la_recompensa_log_suma_al_log_del_patrimonio` lo verifica numéricamente.

### 5.2 `drawdown` — penalizar el daño nuevo

```
r_t = log(1 + r^port_t) − λ · max(0, DD_{t−1} − DD_t)      λ = 10
```

Nace del EDA: la curtosis del BTC horario es **75**, así que el riesgo está concentrado en pocas barras y una recompensa que solo mire el retorno medio no lo captura.

Detalle de diseño que importa: se penaliza el **incremento** del drawdown, no su nivel. Penalizar el nivel cobraría el mismo pozo en cada barra hasta recuperarse, y el agente aprendería a **cerrar la posición para resetear el pico** — reward hacking puro: la política óptima bajo esa recompensa es liquidar en cada caída y nunca volver. `test_la_penalizacion_de_drawdown_solo_castiga_el_dano_nuevo` fija la versión correcta.

### 5.3 `sharpe_dif` — Sharpe diferencial (Moody & Saffell, 1998)

```
D_t = (B_{t−1}·ΔA − ½·A_{t−1}·ΔB) / (B_{t−1} − A²_{t−1})^{3/2}
```

con `A` y `B` medias exponenciales del retorno y del retorno al cuadrado. Da una recompensa densa y ajustada por riesgo sin necesitar el episodio completo. **Su modo de fallo es conocido y se observó:** al arranque la varianza es ~0 y el cociente explota, así que las primeras barras dominarían el gradiente; está acotado a ±10 y las dos primeras barras usan el log-retorno crudo.

### 5.4 Qué pasó

![Retorno en validación por recompensa](figuras/recompensas_f3.png)

Retorno acumulado en validación, media ± desviación entre 2 semillas (fase 3: remuestreo + deriva, que es la comparación limpia):

| Recompensa | Validación | Sharpe | Máx. drawdown | Rotación/año | % invertido |
|---|---|---|---|---|---|
| **`sharpe_dif`** | **+76.4 % ± 31.6** | 1.74 | −34.6 % | 1 161 | 71 % |
| `log_neto` | +66.4 % ± 11.3 | 1.64 | −38.0 % | 1 241 | 81 % |
| `drawdown` | +30.5 % ± 42.4 | 1.65 | **−14.7 %** | **394** | **5 %** |

**El criterio declarado de antemano era el retorno medio en validación, y eligió `sharpe_dif`.** Ganó por **10 puntos porcentuales** sobre `log_neto`, con desviaciones entre semillas de 32 y 11 puntos: la diferencia no es distinguible de ruido.

Lo que el criterio dejó fuera merece quedar escrito. `drawdown` produjo sistemáticamente un agente radicalmente distinto: **5 % del tiempo invertido**, un tercio de la rotación y **menos de la mitad del drawdown**. No es un agente peor, es un agente que resuelve otro problema — el de no perder. Bajo un criterio ajustado por riesgo habría ganado. Cambiar el criterio después de ver los resultados habría sido hacer trampa; registrarlo, no.

### 5.5 El reward hacking que sí ocurrió

No fue el que esperábamos. Las tres recompensas, entrenadas sobre la serie real, **explotaron la identificabilidad del estado**: con 10 indicadores continuos sobre 20 875 barras, el estado localiza la hora, y la política óptima bajo cualquiera de las tres es memorizar «qué pasó ese día». El patrimonio de entrenamiento llegó a **6 319 veces el capital inicial** — un número que no describe una estrategia, describe un índice.

Es reward hacking en el sentido estricto: el agente maximizó la recompensa declarada sin resolver el problema que la recompensa pretendía representar. Y la recompensa no tenía nada de malo; el problema estaba en el **entorno**, que le ofrecía un atajo. Es la lección más transferible del proyecto: cuando un agente obtiene un resultado absurdo, revisar primero qué le permite el entorno, no cómo está escrita la recompensa.

---

## 6. El camino experimental

Siete fases. Cada una responde una pregunta y se decide **solo con el tramo de validación**; `results/decisiones.json` guarda la cadena completa con los números que la justificaron.

| Fase | Pregunta | Corridas | Patrimonio máx. en entrenamiento | Validación (media ± desv) | Rotación/año |
|---|---|---|---|---|---|
| 1 | ¿Qué recompensa, sobre la serie real? | 6 | **6 319 ×** | +53.0 % ± 29.4 | 1 108 |
| 2 | ¿Y si no puede memorizar? (remuestreo por bloques) | 6 | **0.60 ×** | +26.7 % ± 18.1 | 914 |
| 3 | ¿Y si tampoco hereda la deriva? | 6 | 0.50 × | +57.8 % ± 32.4 | 932 |
| 4 | ¿Cuánta red hace falta? | 8 | 3.07 × | +142.0 % ± 74.8 | 662 |
| 5 | Configuración elegida, 5 semillas | 5 | 4.35 × | +190.5 % ± 84.5 | 343 |
| 6 | ¿Cada cuánto decidir? | 6 | 3.07 × | +174.9 % ± 87.4 | 173 |
| 7 | Configuración final, 5 semillas | 5 | 4.35 × | +190.5 % ± 84.5 | 343 |

![Memorización](figuras/brecha_memorizacion.png)

### 6.1 El estado identifica la barra

El diagnóstico está en el notebook de EDA: en el **63 % de las barras**, el estado más parecido de toda la serie de entrenamiento es el de la barra temporalmente contigua, frente al **0.029 %** que cabría esperar si el vecino más cercano pudiera ser cualquiera. El estado no es un identificador perfecto de la hora, pero la localiza en una vecindad estrechísima — y eso basta para memorizar.

### 6.2 Remuestreo por bloques (fase 2)

Cada episodio se construye concatenando bloques de 72 barras (tres días) tomados al azar del tramo de entrenamiento. Con ese largo se preserva el agrupamiento de volatilidad y las colas gruesas, y se destruye el orden concreto en que ocurrió la historia. El agente nunca ve dos veces la misma serie.

**Funcionó, y el número es inequívoco:** el patrimonio de entrenamiento cayó de **6 319 ×** a **0.60 ×**. Los tests `test_conserva_la_media_y_la_volatilidad_de_la_fuente` y `test_preserva_el_agrupamiento_de_volatilidad_dentro_del_bloque` verifican que lo que se preserva se preserva — con un control explícito: con bloques de **1** barra el agrupamiento desaparece, lo que confirma que el test mide algo.

El bloque tiene que ser bastante más corto que el episodio. Con un bloque igual al episodio, cada episodio sería un trozo contiguo de la serie real y la memorización volvería; se usa ~1/10 del episodio en los dos mercados.

### 6.3 La deriva heredada (fase 3)

El remuestreo arrastra un problema que no es obvio: conserva la deriva **incondicional** del tramo fuente. En el mundo sintético el activo tiene retorno esperado positivo y la política óptima vuelve a ser trivial — *estar siempre invertido*. La corrección es sortear una deriva por episodio, `μ ~ N(0, 25 % anual)`, tras quitar la de la fuente. Unos episodios son alcistas y otros bajistas; ninguna exposición fija es óptima en promedio.

### 6.4 Capacidad (fase 4)

| Capacidad | Parámetros | Validación media | Desv. |
|---|---|---|---|
| **32 × 1 capa, decaimiento 1e-2** | **483** | **+220.6 %** | 109.4 |
| 16 × 1 capa, 1e-2 | 243 | +167.6 % | 30.6 |
| 64 × 2 capas, 1e-3 | 5 123 | +103.2 % | 0.2 |
| 128 × 2 capas, 1e-4 | 18 435 | +76.4 % | 31.6 |

El orden es monótono en el tamaño de la red, al revés de lo que diría la intuición: **cuanto más pequeña, mejor generaliza**. Con 20 875 barras de entrenamiento y una razón señal/ruido por barra de ~0.01, no hay material para alimentar más capacidad.

### 6.5 Cada cuánto decidir (fase 6) — y por qué esta es la fase que decidió el resultado

| Decidir cada | Validación media | Desv. | Rotación/año | Costo anual implícito | Sharpe en validación |
|---|---|---|---|---|---|
| **1 barra (hora)** | **+220.6 %** | **109.4** | **412.7** | **41 %** | 2.85 |
| 24 barras (día) | +220.0 % | **20.7** | 73.5 | 7.4 % | 2.67 |
| 168 barras (semana) | +84.1 % | 32.1 | 32.4 | 3.2 % | 1.91 |

**El criterio declarado eligió decidir cada hora por 0.6 puntos porcentuales de diferencia** — sobre desviaciones de 109 y 21 puntos. Es, estadísticamente, un empate; y el ganador tiene cinco veces más varianza y **ocho veces más rotación**.

La razón por la que el criterio no vio el problema está en §2: **el tramo de validación sube 172 % en siete meses**. En ese régimen, pagar 41 % anual en comisiones se nota poco, porque el mercado regala mucho más. En el tramo de prueba, que sube 14.5 %, no hay nada que regale.

Elegir ahora la frecuencia diaria y evaluarla sería una segunda lectura del conjunto de prueba, así que **no se hace ni se reporta**. Lo que sí se puede decir es que cualquier criterio que hubiera incorporado la dispersión entre semillas, o el costo por unidad de retorno, habría elegido la frecuencia diaria — y que en la réplica sobre el Brent, donde el criterio sí eligió la frecuencia barata, el agente ganó (apéndice A).

### 6.6 Una verificación de reproducibilidad que salió gratis

La fase 6 eligió **decidir en cada barra**, que es exactamente la frecuencia con la que ya se había corrido la fase 5. Las corridas `f5_final_s*` y `f7_final_s*` son por tanto la misma configuración con las mismas semillas — y dan los mismos números **hasta el último decimal** (`val_retorno` 1.432029, 2.979734, 1.918793, 0.794938, 2.399996 en las dos). Lo mismo ocurrió en la réplica sobre el Brent entre `f4_minima_s0`, `f5_final_s0` y `f6_cada1_s0`.

No estaba planeado, y es la comprobación de que el control de semillas cubre de verdad todas las fuentes de aleatoriedad: PyTorch, el replay buffer, el entorno, el remuestreo por bloques, la deriva sorteada y la política ε-greedy.

### 6.7 El problema que ninguna fase resolvió

La desviación entre semillas nunca bajó de ~75 puntos porcentuales en validación, mientras que la diferencia entre la mejor y la peor **configuración** de la fase 4 fue de 144 puntos. **El ruido de inicialización es del mismo orden que el efecto de todo lo que se buscó.** Con dos semillas por celda —limitación real de presupuesto de cómputo, declarada— la búsqueda de hiperparámetros selecciona ruido tanto como señal.

Por eso, **antes de leer el tramo de prueba**, se declaró que el agente reportado sería un **comité**: el promedio de Q(s,·) sobre las cinco semillas de la fase 7, con acción `argmax`. Es una decisión única tomada de antemano, no una búsqueda entre 60 candidatos (5 semillas × 12 evaluaciones).

---

## 7. Evaluación en el período held-out

**2021-06-01 → 2022-03-01. Una sola pasada, modo greedy, sin re-ajustar nada.** El agente y toda su configuración quedaron fijados en §6 usando exclusivamente el tramo de validación.

![Patrimonio en el período de prueba](figuras/equity_prueba.png)

| | Retorno acum. | Anualizado | Máx. drawdown | Costos | Sharpe | Vol. anual | Rotación/año | % invertido |
|---|---|---|---|---|---|---|---|---|
| **DQN (comité de 5)** | **−10.7 %** | −13.98 % | **−33.1 %** | 10.68 % | −0.05 | 49.9 % | 155.1 | 53 % |
| DQN (mejor semilla en validación) | −4.4 % | −5.87 % | −32.1 % | 16.73 % | 0.11 | 46.9 % | 240.7 | 51 % |
| Comprar y mantener | **+14.1 %** | +19.22 % | −51.8 % | 0.10 % | **0.60** | 70.1 % | 1.3 | 100 % |
| 50 % fijo | +11.8 % | +16.10 % | −29.8 % | 0.05 % | **0.60** | 35.0 % | 0.7 | 100 % |
| Cruce de medias | −8.5 % | −11.23 % | −40.0 % | 5.92 % | −0.03 | 46.2 % | 81.6 | 50 % |
| Todo en efectivo | 0.0 % | 0.00 % | 0.0 % | 0.00 % | 0.00 | 0.0 % | 0.0 | 0 % |
| Aleatoria (30 semillas) | −93.9 % | −97.57 % | −94.4 % | **102.64 %** | −8.08 | 45.4 % | 3 893.6 | 67 % |

Las tres métricas que pedía el enunciado: **retorno acumulado −10.7 %, máximo drawdown −33.1 %, costos de transacción 10.7 % del capital** en nueve meses.

El agente pierde contra comprar y mantener y contra el 50 % fijo. Le gana a la política aleatoria por 83 puntos, pero eso **no significa nada aquí**: la aleatoria horaria rota 3 894 veces al año y se gasta **el 103 % del capital en comisiones**. Superarla es superar a una máquina de quemar dinero, no a una estrategia.

Lo único que el agente hizo mejor que comprar y mantener es el riesgo: **drawdown de −33 % frente a −52 %** y volatilidad de 50 % frente a 70 %. Redujo el riesgo a la mitad. Pero el 50 % fijo —una estrategia sin ninguna inteligencia— logra lo mismo y además gana dinero.

### 7.1 Descomposición: ¿costos o política?

El mismo comité, la misma secuencia de acciones, solo cambiando la contabilidad:

| | Retorno acum. | Sharpe |
|---|---|---|
| DQN con costo real (10 bps) | −10.7 % | −0.05 |
| DQN **sin costos de transacción** | +0.3 % | 0.26 |
| Comprar y mantener | +14.1 % | 0.60 |

Los costos explican **11 de los 24.8 puntos** que lo separan de comprar y mantener: un 44 %. El resto es la política.

### 7.2 ¿Hay alguna señal en la política?

![Exposición frente al precio](figuras/exposicion_prueba.png)

No — y el dato decisivo no es el de retorno:

| | |
|---|---|
| Correlación exposición(t) ↔ retorno(t+1) | **−0.0028** |
| Retorno medio del activo **invertido** | +0.00199 % / hora |
| Retorno medio del activo **en efectivo** | +0.00808 % / hora |
| Retorno medio incondicional | +0.00483 % / hora |
| **Volatilidad del activo invertido** | **0.00752** |
| **Volatilidad del activo en efectivo** | **0.00746** |

El agente estuvo fuera del mercado precisamente en las horas mejores. Pero lo que condena la política es la última fila: **la volatilidad del activo mientras el agente estaba invertido es indistinguible de la que había cuando estaba en efectivo**. La volatilidad es lo único que el EDA mostró predecible (autocorrelación de `|r|` de +0.29 a una hora), el espacio de acciones se diseñó para explotarla, y el agente **no la explotó en absoluto**. No es que la señal fuera débil y los costos se la comieran: la señal estaba ahí y el agente no la encontró.

### 7.3 El comité, y una predicción que no se replicó

| | Brent (apéndice A) | BTC |
|---|---|---|
| DQN comité | **+147.5 %** | −10.7 % |
| DQN mejor semilla por validación | +6.7 % | **−4.4 %** |
| Diferencia a favor del comité | **+141 pp** | −6 pp |

En Brent el comité fue dramáticamente mejor, tal como se predijo por adelantado. En BTC fue ligeramente peor. **La predicción no se replicó**, y la conclusión correcta no es la que yo quería escribir. No es «el comité es mejor»: es que **con una dispersión entre semillas de ±85 puntos, ningún procedimiento de selección entre cinco modelos es confiable**, y el comité es preferible por reducir varianza, no porque garantice mejor resultado.

### 7.4 El diagnóstico de los valores Q

```
Q medios sobre todos los estados de prueba (filas: exposición actual; columnas: acción)
          0 %     50 %    100 %
 0 %    5.624   5.511   5.380
50 %    5.520   5.544   5.473
100 %   5.300   5.525   5.595
```

La diagonal domina —mantener la exposición actual vale más que cambiarla, que es exactamente lo que el costo debería enseñar— pero los nueve valores caben en un rango del **6 %**. La política se decide por diferencias pequeñas frente al ruido de estimación de la propia red, que es la razón de la inestabilidad entre semillas.

---

## 7.5 Tres algoritmos, el mismo protocolo

El enunciado recomienda DQN y permite PPO si se justifica. Se entrenaron ambos y además un **Q-Learning tabular**, con idéntico entorno, recompensa, partición, criterio de selección y comité de cinco semillas. Lo único distinto es qué aprende cada uno.

![Comparación de algoritmos](figuras/comparacion_algoritmos.png)

| | Validación (media ± desv) | Prueba | Costos | Rotación/año | % invertido |
|---|---|---|---|---|---|
| **DQN** | **+190.5 % ± 84.5** | **−10.7 %** | **10.68 %** | **155.1** | 53 % |
| PPO (Stable-Baselines3) | +106.0 % ± 67.4 | −29.5 % | 24.30 % | 367.1 | 75 % |
| Q-Learning tabular | +24.5 % ± 35.2 | −32.1 % | 37.38 % | 606.4 | 94 % |

Tres cosas que vale la pena señalar:

1. **El orden en validación se conserva en prueba.** No era obvio con esta varianza, y es la única parte del protocolo que se comportó como uno esperaría.
2. **El orden de los resultados es exactamente el orden de los costos.** DQN paga 10.7 %, PPO 24.3 %, el tabular 37.4 %. En un problema donde la ventaja informativa por operación es de 2.8 bps y el costo de ida y vuelta es de 20, lo que más predice el resultado es cuánto opera cada uno — no qué tan sofisticado es el algoritmo.
3. **Los tres fallan igual.** Correlación entre exposición y retorno siguiente de −0.003 a −0.008, y ninguno hace *timing* de volatilidad: la volatilidad del activo estando invertido es indistinguible de la que hay estando en efectivo. Que tres familias de algoritmos distintas fallen de la misma forma es una conclusión mucho más robusta que el fallo de una sola.

**Por qué el tabular está.** No para competir: para hacer visible por qué existe la aproximación de funciones. El estado tiene once componentes continuas y una tabla sobre ese espacio es imposible —con diez cortes por dimensión serían 10¹¹ celdas frente a 20 875 barras—, así que el agente tabular renuncia a casi todo el estado y se queda con tres indicadores discretizados en cuartiles: 192 celdas, 99-100 % visitadas. La comparación deja de ser «tres algoritmos sobre el mismo problema» y pasa a ser **qué se pierde al tener que resumir el estado**.

**Por qué PPO está.** Si los valores Q de las tres acciones caben en un rango del 6 % (§7.4), el `argmax` del DQN es un amplificador de ruido; un método de política no toma `argmax` sobre estimaciones parecidas. Si la fragilidad viniera de ahí, PPO debería notarse. No se notó: rotó más del doble y rindió peor. Se usó un framework de referencia a propósito, para que una eventual diferencia no pudiera atribuirse a un error de nuestra implementación. El entorno pasa `check_env` de SB3 sin modificaciones.

---

## 8. Reflexión individual

**William Mauricio Torres Ríos**

**Dos errores míos definieron este proyecto, y ninguno de los dos lanzó una excepción.**

El primero fue de diseño y lo encontré a tiempo: la fase 1 dio un patrimonio de entrenamiento de 6 319 veces el capital. En el contexto en el que trabajo —gerencia de tecnología, donde un modelo con métricas espectaculares es lo que la gente quiere ver— ese número habría pasado como un éxito en más de una presentación. Era basura: el agente había aprendido un calendario. Lo que lo delató no fue la intuición sino el procedimiento: haber separado la validación antes de empezar y haberla mirado al lado del entrenamiento en la misma gráfica.

El segundo lo encontré tarde y por casualidad. `recorrido_completo` construía la configuración de evaluación enumerando campos a mano, y se me quedó fuera `frecuencia_decision`. Resultado: **todos los agentes se evaluaban decidiendo en cada barra, sin importar con qué frecuencia se hubieran entrenado**, y la fase que comparaba frecuencias estaba midiendo siempre lo mismo. No falló nada. Los números eran plausibles. Lo vi porque un agente «semanal» reportó 2 136 operaciones al año y el máximo posible son 52 — es decir, lo vi porque una cifra violaba una cota que yo podía calcular de cabeza, no porque ningún test fallara. Al corregirlo, el resultado del Brent pasó de −3.3 % a **+147.5 %**: el error estaba haciendo que un agente que ganaba pareciera perder. Lo que me llevo es el hábito, no el arreglo: enumerar campos a mano en un constructor es una lista que se desactualiza en silencio, y toda métrica debería tener una cota conocida contra la cual chocar.

**La tercera lección es sobre el criterio de selección, y es la que más me costó aceptar.** Declaré de antemano que elegiría por retorno acumulado en validación, precisamente para no poder ajustarlo a conveniencia. Funcionó: no lo cambié. Pero resultó estar **ciego al costo**, porque el tramo de validación de BTC sube 172 % en siete meses y en ese régimen pagar 41 % anual en comisiones es invisible. El criterio eligió decidir cada hora frente a cada día por 0.6 puntos porcentuales, sobre desviaciones de 109 y 21 — un empate que el criterio no podía ver como empate. Y esa decisión, sola, explica buena parte de la diferencia entre el BTC que pierde y el Brent que gana. **Un criterio declarado de antemano te protege de engañarte, pero no te protege de haber elegido mal el criterio.** Lo segundo solo se arregla pensando antes en qué régimen vas a seleccionar.

**Sobre la pregunta del enunciado, la respuesta honesta es «depende, y el depende es instructivo».** En Brent el agente superó a comprar y mantener en retorno (+147.5 % frente a +131.9 %), en Sharpe (0.52 frente a 0.48) y sobre todo en drawdown (−57 % frente a −88 %). En BTC perdió. No tengo evidencia de que haya aprendido a invertir en ninguno de los dos: en BTC la correlación entre su exposición y el retorno siguiente es −0.003, y —el dato que más me incomoda— **la volatilidad del activo mientras estaba invertido es la misma que cuando estaba en efectivo**. El EDA había mostrado que la volatilidad es lo único predecible de la serie, diseñé el espacio de acciones para explotarla, y el agente no la explotó. Lo que el Brent muestra probablemente no es habilidad sino una exposición media más baja con pocos costos, en un tramo donde eso bastaba.

**Lo que haría distinto, declarado como lo que es — hipótesis, no resultado.** Un criterio de selección que divida retorno entre costos o que penalice la dispersión entre semillas, fijado antes de la fase 1. Diez semillas por celda en vez de dos, aunque cueste menos exploración: más vale medir bien tres opciones que medir mal ocho. Un tramo de validación que contenga al menos un régimen bajista, aunque eso signifique romper el orden cronológico con una partición por bloques alternados. Y probablemente un espacio de acciones continuo con PPO en vez de tres niveles discretos con DQN, porque con márgenes entre acciones del 6 % el `argmax` es un amplificador de ruido.

Ninguna de estas cosas la puedo probar con estos datos sin invalidar el experimento. Esa restricción —no poder usar al final lo que aprendí sobre datos que ya miré— es la parte del método que más me costó respetar, y la que separa un resultado de una anécdota.

---
**Joan Montenegro**  
**reproducir un resultado también forma parte importante de un proyecto** 

La revisión del repositorio me permitió entender que la reproducibilidad depende de más elementos que las semillas aleatorias. También requiere que otra persona pueda instalar las dependencias, localizar los modelos y ejecutar la evaluación siguiendo las instrucciones disponibles.

Un caso que me llamó la atención fue que las pruebas podían terminar sin errores aunque algunas no se hubieran ejecutado por falta de una dependencia. Esto me llevó a distinguir entre una ejecución exitosa y una verificación completa. También encontramos modelos incluidos en el repositorio que un evaluador no podía recuperar porque los buscaba en otra ubicación.

Estos problemas no cambiaban lo aprendido por los agentes, pero dificultaban comprobar sus resultados. Si repitiera el proyecto, incorporaría una revisión desde un entorno limpio antes de entregar: instalar el proyecto, ejecutar las pruebas y regenerar la evaluación únicamente con los archivos publicados. Considero que esa comprobación debería acompañar la evaluación del modelo, porque permite que el trabajo pueda ser utilizado y revisado por otras personas.

## 9. Limitaciones

Declaradas, no escondidas. Cada una acota hasta dónde llega la conclusión de §7.

1. **Dos semillas por celda en las fases de búsqueda.** Es poco, y se nota: la desviación entre semillas domina las comparaciones de las fases 3, 4 y 6. Fue una restricción de cómputo (dos núcleos, ~6 minutos por corrida de 120 000 pasos), no una decisión metodológica.
2. **El tramo de validación de BTC es un mercado alcista extremo.** Está declarado en §2 y sus consecuencias medidas en §6.5, pero sigue siendo la limitación más seria del diseño: un tramo de validación con un solo régimen no puede seleccionar un agente para otro.
3. **Un solo período de prueba por mercado.** Un *walk-forward* con varias ventanas rodantes daría una distribución de resultados en vez de un número, que es lo que corresponde con esta varianza. No cabía en el presupuesto de cómputo.
4. **Nueve meses de held-out en BTC.** 6 553 barras horarias son muchas observaciones pero un solo episodio de mercado. Las conclusiones sobre BTC son más frágiles que las del Brent, que cubre 7.8 años.
5. **Selección por validación sobre ~12 evaluaciones por corrida.** El checkpoint de cada miembro del comité se eligió por su mejor punto en validación. Eso introduce optimismo en las cifras de validación de §6 — no en las de prueba, que son limpias.
6. **Costos simplificados.** 10 bps fijos sobre la rotación, sin modelar deslizamiento, impacto de mercado ni el diferencial compra-venta, y con el efectivo remunerado al 0 %. En BTC la cifra coincide con la comisión *taker* de Binance, así que es realista para el tamaño de orden pequeño; para órdenes grandes sería optimista.
7. **El estado no usa volumen ni datos de libro de órdenes.** Fue deliberado (§1), pero significa que el agente opera con menos información de la que tendría un sistema real.

### Mejora pendiente para próximos entrenamientos: escala de la deriva

Una revisión posterior identificó un problema en la deriva, la tendencia media
añadida a las trayectorias sintéticas de entrenamiento. EntornoBootstrap
convierte la deriva anual a cada barra dividiendo entre 252. Esa conversión
corresponde al Brent diario, pero BTC horario tiene 8 760 barras al año. Por
ello, el aporte de la deriva por hora queda amplificado unas 34,8 veces.

**La corrección está pendiente:** utilizar las barras anuales de cada mercado y
añadir una prueba que verifique la conversión. Para medir su efecto será
necesario entrenar nuevos modelos y compararlos manteniendo las demás
condiciones. Los resultados publicados corresponden a la implementación
histórica; el impacto de este problema sobre su desempeño todavía no se ha
cuantificado.

---

## 10. Reproducir estos resultados

### Instalación y pruebas con uv

Se necesita `uv` instalado. El proyecto requiere Python 3.11 o posterior; los
comandos siguientes usan Python 3.12, con el que se verificaron las pruebas.
Desde la raíz del repositorio:

```bash
uv sync --locked --python 3.12 --extra dev
uv run --locked --extra dev pytest -ra
```

`uv sync` crea o actualiza `.venv`. `--locked` exige que `uv.lock` coincida con
`pyproject.toml` y conserva las versiones resueltas. El extra `dev` incluye las
herramientas de pruebas y notebooks. DQN, el agente tabular, los experimentos,
el pipeline y los reportes funcionan sin instalar PPO. En un entorno sin
`stable-baselines3`, **94 pruebas pasan y se omiten 7** que requieren esa
dependencia; las pruebas tabulares se ejecutan siempre.

Para entrenar PPO o ejecutar todas las pruebas, se añade su extra opcional:

```bash
uv sync --locked --python 3.12 --extra dev --extra ppo
uv run --locked --extra dev --extra ppo pytest -ra
```

Con ambos extras, **las 101 pruebas pasan sin omisiones**. `--extra ppo` instala
`stable-baselines3`; no selecciona ni ejecuta ese algoritmo. PPO se entrena
explícitamente con `src/train_alt.py ppo`.

### Datos, análisis y entrenamiento

```bash
uv run --locked python src/data.py  # baja y valida las tres series
uv run --locked --extra dev jupyter execute notebooks/eda.ipynb

uv run --locked python src/experiments.py 1 --pasos 120000  # serie real
uv run --locked python src/experiments.py 2 --pasos 120000  # remuestreo por bloques
uv run --locked python src/pipeline.py --pasos 120000  # fases 3 a 7
uv run --locked --extra ppo python src/train_alt.py ppo --semilla 0
uv run --locked python src/train_alt.py tabular --semilla 0
uv run --locked python src/report.py
```

Los YAML de `configs/` documentan las configuraciones de referencia y final;
los scripts no los cargan automáticamente. Los valores efectivos de cada
corrida se guardan en `config.json` dentro de su carpeta: `results/corridas/<etiqueta>/`
para BTC y `results_brent/corridas/<etiqueta>/` para Brent. DQN (`train.py`),
PPO y el tabular (`train_alt.py`) eligen esa carpeta con `--mercado`.
El pipeline usa
150 000 pasos por defecto; `--pasos 120000` selecciona el presupuesto reportado
para BTC. Los ejemplos de `train_alt.py` entrenan una semilla; para obtener los
cinco miembros de cada comité se repiten con `--semilla` de 0 a 4.

Para entrenar y evaluar una nueva corrida DQN de Brent sin mover archivos:

```bash
uv run --locked python src/train.py --mercado brent --pasos 120000 --etiqueta nueva_s0
uv run --locked python src/evaluate.py --mercado brent --patron nueva_s0 --etiqueta nueva_s0
```

Este ejemplo usa los parámetros por defecto del entrenador. `--patron` selecciona
la corrida nueva y `--etiqueta` en el evaluador guarda sus resultados con un
nombre propio, conservando la evaluación publicada.

`experiments.py`, `pipeline.py` y `report.py` también aceptan `--mercado`
(por defecto, `btc`). Para ejecutar la búsqueda y generar reportes de Brent:

```bash
uv run --locked python src/experiments.py 1 --mercado brent --pasos 150000
uv run --locked python src/experiments.py 2 --mercado brent --pasos 150000
uv run --locked python src/pipeline.py --mercado brent --pasos 150000
uv run --locked python src/report.py --mercado brent
```

Cada fase entrena el mercado solicitado y guarda su resumen en la carpeta de
ese mercado. El pipeline lee sus corridas para seleccionar la recompensa,
la capacidad y la frecuencia, y guarda `decisiones.json` en la misma carpeta.
Los reportes generan allí `resumen_experimentos.csv`; las figuras de Brent
usan nombres como `recompensas_brent_f3.png` y `brecha_memorizacion_brent.png`.
Estos comandos ejecutan nuevas corridas; para evaluar directamente los modelos
incluidos se usan los comandos de la sección siguiente.

`Escalador.de_dict()` acepta los nombres antiguos de los indicadores diarios
guardados de Brent (por ejemplo, `ret_5` y `vol_21`) y los traduce a los nombres
actuales. Conserva el orden, las medias y las desviaciones; los archivos
guardados permanecen intactos. Esos nombres antiguos requieren ventanas diarias.

### Evaluación final por mercado

```bash
uv run --locked python src/baselines.py --tramo prueba
uv run --locked python src/evaluate.py
uv run --locked python src/evaluate.py --mercado brent
```

La evaluación lee el tramo de prueba: la configuración y los modelos deben
quedar fijados antes de ejecutarla. En `evaluate.py`, `--mercado` selecciona
conjuntamente los datos, las corridas candidatas, el comité y la carpeta de salida:

| Mercado | Corridas | Comité incluido | Tablas, JSON y bitácoras |
|---|---|---|---|
| `btc` (por defecto) | `results/corridas/` | `results/comite/` | `results/` |
| `brent` | `results_brent/corridas/` | `results_brent/comite/` | `results_brent/` |

Para el mejor individuo, el evaluador usa `corridas/<corrida>/mejor.pt` dentro
de la carpeta del mercado si existe; si falta, recupera
`comite/semilla_<semilla>.pt` y comprueba que su configuración y paso de
entrenamiento correspondan al individuo elegido. Esa recuperación requiere
también el archivo `semilla_<semilla>.config.json` y falla con un mensaje
explícito si el respaldo falta o no corresponde. La evaluación de ambos mercados
puede completarse con los modelos incluidos, sin volver a entrenar. Si faltan
los modelos del mercado solicitado, se informa del error sin usar los de otro.

Con la etiqueta predeterminada `prueba`, las salidas se llaman
`evaluacion_prueba.csv`, `evaluacion_prueba.json` y `bitacora_dqn_prueba.csv`
dentro de la carpeta del mercado. Las figuras quedan en `figuras/`: BTC genera
`equity_prueba.png` y `exposicion_prueba.png`; Brent genera
`equity_brent_prueba.png` y `exposicion_brent_prueba.png`.

Los entrenadores y el evaluador comparten la misma selección de carpetas por
mercado. Las corridas nuevas de Brent quedan directamente en
`results_brent/corridas/`, disponibles para `evaluate.py --mercado brent`.

### El dataset, identificado

| Archivo | Contenido | Fuente |
|---|---|---|
| `data/btc_crudo.csv` | BTC/USD horario, 33 259 barras, may-2018 → mar-2022 | [`ClementPerroud/Gym-Trading-Env`](https://github.com/ClementPerroud/Gym-Trading-Env), `examples/data/BTC_USD-Hourly.csv` |
| `data/brent_crudo.csv` | Brent spot diario, 9 102 barras desde 1987 | `datasets/oil-prices` (EIA) |
| `data/sp500m_crudo.csv` | S&P 500 mensual, 1871-2026 | `datasets/s-and-p-500` (Shiller) |

Los CSV no se versionan: los descarga `src/data.py` mediante el comando de
`uv run` indicado arriba. Ver `data/README.md`.

Todas las fuentes de aleatoriedad están sembradas: `torch.manual_seed`, el generador del replay buffer, el del entorno, el del remuestreo y el de la política ε-greedy reciben la misma semilla de `Config.semilla`. `test_dos_agentes_con_la_misma_semilla_son_identicos` verifica que dos corridas con la misma semilla produzcan la misma secuencia de acciones.

**Cómputo:** 42 corridas de 120 000 pasos en BTC más 42 de 150 000 en Brent, en dos núcleos de CPU. Unas 4 horas de reloj en total.

### Estructura del repositorio

```
agente-inversor/
├── data/                      # series de precios (las baja src/data.py)
├── src/
│   ├── data.py                # descarga, validación, partición congelada, loader de gym-trading-env
│   ├── features.py            # los 10 indicadores causales + presets de ventanas + escalador
│   ├── environment.py         # entorno Gymnasium, bootstrap por bloques, política tabulada
│   ├── agent.py               # DQN, replay buffer, red objetivo, tabla Q por lotes
│   ├── ppo.py                 # PPO sobre Stable-Baselines3
│   ├── tabular.py             # Q-Learning tabular (el método clásico)
│   ├── train_alt.py           # entrena PPO y tabular con el mismo protocolo
│   ├── baselines.py           # comprar y mantener, aleatoria, cruce de medias, 50 % fijo
│   ├── metrics.py             # retorno, drawdown, costos, Sharpe, rotación
│   ├── train.py               # entrenamiento y selección por validación
│   ├── experiments.py         # las fases, paralelizadas
│   ├── pipeline.py            # encadena las fases y registra cada decisión
│   ├── evaluate.py            # evaluación final en el held-out, comité
│   ├── plots.py               # figuras
│   └── report.py              # tablas y figuras del informe
├── notebooks/eda.ipynb        # análisis exploratorio (solo entrenamiento)
├── configs/
│   ├── base.yaml              # configuración de referencia
│   └── final.yaml             # la del agente reportado en §7, con su justificación
├── tests/                     # 101 pruebas en 11 archivos; 7 requieren el extra ppo
│   ├── test_sin_fuga.py       # ← el que sostiene la honestidad del experimento (los dos mercados)
│   ├── test_entorno.py        # contabilidad: costos, patrimonio, recompensas, frecuencia
│   ├── test_entrenamiento_rutas.py # entrenamiento breve por mercado y evaluación sin mover archivos
│   ├── test_baselines.py      # las líneas base son lo que dicen ser
│   ├── test_agente.py         # DQN, replay, truncación
│   ├── test_bootstrap.py      # el remuestreo preserva estadística y destruye orden
│   ├── test_evaluacion_modelos.py # respaldos y evaluación de BTC y Brent por separado
│   ├── test_evaluacion_rapida.py  # la tabla Q es la misma política + regresión del error de §8
│   ├── test_escalador.py      # compatibilidad de formatos y conservación de estados
│   ├── test_experimentos_mercados.py # fases, decisiones y reportes por mercado
│   └── test_agentes_comparados.py # PPO y tabular comparten la interfaz del DQN
├── results/                   # BTC: corridas, comité, decisiones y evaluación
├── results_brent/             # la réplica del apéndice A
├── figuras/
├── pyproject.toml
├── uv.lock                    # versiones resueltas para uv sync --locked
└── README.md
```

### Cómo auditar que no hubo fuga hacia el tramo de prueba

1. **El código de entrenamiento no puede leer el tramo de prueba.** `train.montar()` no lo carga: hay que pedirlo con `incluir_prueba=True`, y el único módulo que lo hace es `evaluate.py`. Cualquier intento desde `train.py`, `experiments.py` o `pipeline.py` muere con `KeyError: 'prueba'`.
2. `results/decisiones.json` registra cada decisión con la tabla de validación que la justificó.
3. `results/corridas/*/config.json` guarda la configuración resuelta de cada corrida, con la partición incluida.
4. `uv run --locked --extra dev pytest tests/test_sin_fuga.py` verifica, en los dos mercados, la causalidad de los indicadores, el congelamiento del escalador y que ninguna barra operable aparezca en dos tramos.

---

## Apéndice A — Réplica sobre el Brent

Antes de que el profesor señalara los datos de Gym Trading Env, el experimento completo se corrió sobre el **precio spot diario del crudo Brent** (EIA, 9 102 barras desde 1987). Se conserva porque responde una pregunta que el mercado principal por sí solo no puede: **¿cuánto de lo que se concluye es propio del BTC y cuánto es propio del método?**

Es la misma base de código, con un cambio de preset de ventanas (`features.DIARIO` en vez de `features.HORARIO`) y una partición propia: entrenamiento 1987-2013, validación 2014-2018, prueba 2019-2026.

### Resultado en el período held-out (2019-2026)

| | Retorno acum. | Máx. drawdown | Costos | Sharpe | Rotación/año | % invertido |
|---|---|---|---|---|---|---|
| **DQN (comité de 5)** | **+147.5 %** | **−57.5 %** | 10.66 % | **0.52** | 12.0 | 66 % |
| DQN (mejor semilla en validación) | +6.7 % | −66.8 % | 9.85 % | 0.19 | 12.2 | 73 % |
| Comprar y mantener | +131.9 % | −87.8 % | 0.10 % | 0.48 | 0.1 | 100 % |
| 50 % fijo | +106.1 % | −60.4 % | 0.05 % | 0.48 | 0.1 | 100 % |
| Cruce de medias 21/63 | +76.1 % | −60.4 % | 4.10 % | 0.39 | 4.5 | 51 % |
| Aleatoria (30 semillas, cada barra) | −35.3 % | −73.3 % | 71.45 % | −0.04 | 111.9 | 67 % |
| Aleatoria (cada 5 barras, igualada al DQN) | +39.5 % | −69.6 % | 19.11 % | 0.25 | 22.4 | 66 % |

**El agente del Brent sí supera a comprar y mantener**: más retorno, mejor Sharpe y **30 puntos menos de drawdown**, con 12 operaciones al año. También supera a la aleatoria igualada a su misma frecuencia (+147.5 % frente a +39.5 %), que es el control que separa «el agente sabe algo» de «el agente simplemente opera poco».

![Patrimonio en el período de prueba del Brent](figuras/equity_brent.png)

### Lo que la réplica aporta

1. **La memorización no es propia de un activo.** Sobre la serie real del Brent el agente alcanzó un patrimonio de entrenamiento de **1.06 × 10¹⁴ veces el capital** con validación negativa; sobre BTC llegó a 5 816 ×. En los dos casos el remuestreo por bloques lo devolvió al orden de magnitud de comprar y mantener. El diagnóstico subyacente también se replica: el vecino más cercano en el espacio de estados es una barra contigua en el **53 %** de los casos en Brent y en el **63 %** en BTC.

2. **El comité pesó muchísimo aquí.** El individuo elegido por validación rindió **+6.7 %** y el comité **+147.5 %**: 141 puntos de diferencia, en la dirección predicha antes de leer el test. En BTC la misma predicción falló por 6 puntos en la dirección contraria. Dos mercados, dos resultados opuestos: lo que queda en pie no es «el comité gana» sino que la selección entre cinco modelos con ±85 pp de dispersión no es un procedimiento confiable en ninguno de los dos sentidos.

3. **Los dos tramos de prueba son opuestos, y eso es útil.** El del Brent subió 132 % y la línea base era casi imbatible; el del BTC sube 14.5 % con un ida y vuelta de por medio y sí es batible. Un método que solo se hubiera probado en el primero no habría podido distinguir «el agente es malo» de «la línea base era inalcanzable».

4. **La frecuencia de decisión es la variable que separa los dos resultados.** En Brent el criterio eligió decidir cada 5 barras: 12 operaciones al año, 1.4 % anual en costos. En BTC eligió decidir cada barra: 155 operaciones al año, 14 % anual. Mismo criterio, mismo código; lo que cambió fue el régimen del tramo de validación donde ese criterio tenía que discriminar (§6.5).

5. **Una fase que hubo que descartar.** La fase 6 del Brent se corrió con el error de `recorrido_completo` descrito en §8 —todas las frecuencias se evaluaban por barra, así que la comparación no medía lo que decía— y **no se volvió a correr**. Las fases 1 a 5 del Brent y su evaluación final sí son válidas: la evaluación se rehízo con el código corregido. La conclusión que esa fase pretendía sostener se apoya hoy solo en la evidencia de BTC.

Los artefactos completos de esta réplica (42 corridas, bitácoras, decisiones y la evaluación) están en `results_brent/`.

---
