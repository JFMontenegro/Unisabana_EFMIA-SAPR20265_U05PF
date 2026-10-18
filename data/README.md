# data/

Esta carpeta queda **vacía en el repositorio a propósito**. Las series de precios no se
versionan: `.gitignore` excluye `data/*.csv` porque los archivos de origen se actualizan
(el de Brent crece cada día) y versionarlos ensuciaría cada diff con ruido.

Para llenarla, desde la raíz del proyecto y después de la instalación con `uv`
descrita en el README principal:

```bash
uv run --locked python src/data.py
```

Eso baja y valida las tres series, y escribe aquí:

| Archivo | Contenido | Fuente |
|---|---|---|
| `btc_crudo.csv` | BTC/USD horario, 33 259 barras, may-2018 → mar-2022 | `ClementPerroud/Gym-Trading-Env`, `examples/data/BTC_USD-Hourly.csv` |
| `brent_crudo.csv` | Brent spot diario, 9 102 barras desde 1987 | `datasets/oil-prices` (EIA) |
| `sp500m_crudo.csv` | S&P 500 mensual, 1871-2026 | `datasets/s-and-p-500` (Shiller) |

Si en su lugar quiere usar el descargador de Gym Trading Env, deje aquí el pickle que
produce y cárguelo con `data.cargar_pickle_gte("data/binance-BTCUSDT-1h.pkl")`.
