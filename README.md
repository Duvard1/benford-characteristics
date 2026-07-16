# Benford Características Acústicas
Detección de voz generada por IA en llamadas telefónicas mediante características acústicas que intentan describir como se produce la voz usando la Ley de Benford.


## Requisitos Previos

Este proyecto necesita **ffmpeg** instalado en el sistema (no en Python), porque `pydub` lo usa por debajo para decodificar formatos comprimidos (AAC, MP3, OGG, etc). La conversión se hace siempre desde Python, nunca con comandos manuales de terminal por parte del usuario.


Verificar que esté instalado:

```bash
ffmpeg -version
```

Si no lo tienes:
- Ubuntu/Debian: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: descargar desde https://ffmpeg.org/download.html y agregarlo al PATH

---

## Instalación

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

---

## Instalación del proyecto

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecutar la API

```bash
uvicorn app.main:app --reload
```

La API quedará disponible en `http://127.0.0.1:8000`.
Documentación interactiva automática: `http://127.0.0.1:8000/docs`.

## Probar el endpoint `/benford-char`

El endpoint `/benford-char` recibe un archivo de audio, calcula sus características y realiza una **clasificación heurística** para estimar si la voz es **REAL** o generada por **IA** basándose en la métrica MAD del RMS (con un umbral definido en `config.py`).

```bash
# Obtener todas las características
curl -X POST "http://127.0.0.1:8000/benford-char" \
  -F "file=@ruta/a/tu/audio.aac"

# Filtrar para obtener solo una característica específica (ej. rms)
curl -X POST "http://127.0.0.1:8000/benford-char?caracteristica=rms" \
  -F "file=@ruta/a/tu/audio.aac"
```

Opcionalmente, se puede filtrar la respuesta mediante el parámetro de consulta `caracteristica`. Los valores contemplados y sus descripciones son:

- `rms`: Root Mean Square. Mide qué tan fuerte o qué tan alto suena la voz.
- `pitch`: Frecuencia fundamental (F0). Mide qué tan grave o aguda es la voz.
- `zcr`: Zero Crossing Rate. Cuenta cuántas veces la señal cambia de positiva a negativa.
- `silencios`: Duración (en mlseg) de los momentos donde no hay voz.
- `jitter`: Mide qué tanto cambia el Pitch de un instante al siguiente.
- `shimmer`: Mide qué tanto cambia el volumen entre un instante y otro.
- `todas`: Si no se especifica una caracteristisca se retorna la lista completa de características analizadas.

Respuesta esperada (ejemplo, filtrado por `rms`):

```json
{
  "archivo": {
    "nombre_original": "audio.aac",
    "sample_rate": 16000,
    "duracion_segundos": 12.4,
    "num_muestras": 198400,
    "wav_normalizado": "9f2a3c1b.wav"
  },
  "features": {
    "rms": {
      "resumen": {
        "num_frames": 198,
        "rms_mean": 0.097,
        "rms_std": 0.025
      },
      "benford": {
        "n_valores": 180,
        "distribucion_observada": [0.31, 0.19, 0.12, 0.10, 0.08, 0.07, 0.05, 0.05, 0.03],
        "distribucion_teorica": [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046],
        "metricas": {"mad": 0.0128, "chi2": 4.21, "kl": 0.006, "js": 0.0015}
      }
    }
  },
  "clasificacion": {
    "prediccion": "REAL",
    "confianza": "Alta",
    "feature_utilizada": "RMS MAD",
    "valor_obtenido": 0.0339,
    "umbral": 0.0272
  },
  "mensaje": "Audio normalizado y analizado exitosamente con la Ley de Benford."
}
```

Cada bloque dentro de `features` contiene el `resumen` (estadísticas descriptivas) y el análisis de `benford` (distribución observada vs teórica + métricas MAD/χ²/KL/JS). Los valores crudos de los arrays no se incluyen en la respuesta para mantenerla compacta y optimizada.


## Evaluación en Lote (Batch)

El proyecto incluye un script para evaluar de manera masiva conjuntos de audios de prueba y generar reportes comparativos de las métricas de Benford.

### 1. Preparar los datos
Coloca tus audios de prueba en las carpetas correspondientes bajo el directorio `data/`:
- Audios reales (humanos): `data/real/`
- Audios generados por IA (sintéticos): `data/ia/`

Formatos soportados: `.wav`, `.mp3`, `.aac`, `.m4a`, `.ogg`, `.opus`, `.flac`.

### 2. Ejecutar la evaluación
Ejecuta el script desde la raíz del proyecto usando:

```bash
python -m scripts.evaluar_dataset
```

### 3. Resultados
El script procesará cada archivo y realizará lo siguiente:
- Guardará un análisis individual en formato JSON en `results/real/{nombre_audio}.json` o `results/ia/{nombre_audio}.json`.
- Generará un resumen consolidado en `results/reporte_global.json` con los promedios globales de divergencia/distancia de Benford.
- Mostrará en la consola una tabla comparativa detallando las métricas de cada feature para ambos grupos y calculando su diferencia.

## Estructura del proyecto

```
voice-ai-detector/
│
├── app/
│   ├── __init__.py
│   ├── main.py             # Inicializa la API FastAPI
│   ├── routes.py           # Endpoints (POST /benford-char, GET /health)
│   ├── audio.py            # Carga, valida y normaliza audio -> WAV PCM 16kHz mono
│   ├── config.py           # Constantes centralizadas (rutas, sample rate, etc)
│   ├── utils.py            # Helpers genéricos (nombres únicos, limpieza temp)
│   ├── analysis.py         # Orquestador del cálculo de features + Benford + métricas
│   │
│   ├── features/           # Extracción de características de audio
│   │   ├── __init__.py
│   │   ├── framing.py      # Ventaneo de señales (frame size y hop size)
│   │   ├── energy.py       # RMS por frame
│   │   ├── pitch.py        # Pitch (F0) por frame (autocorrelación)
│   │   ├── zcr.py          # Zero Crossing Rate por frame
│   │   ├── silence.py      # Detección de pausas/silencios
│   │   ├── jitter.py       # Jitter (variación del período de pitch)
│   │   └── shimmer.py      # Shimmer (variación de amplitud/RMS)
│   │
│   ├── benford/            # Distribución y análisis de Benford
│   │   ├── __init__.py
│   │   └── benford.py      # Extracción del primer dígito y distribución
│   │
│   └── metrics/            # Métricas de distancia/divergencia
│       ├── __init__.py
│       └── metrics.py      # Implementación de MAD, chi2, KL, JS
│
├── scripts/
│   └── evaluar_dataset.py  # Script de evaluación en batch del dataset de prueba
│
├── uploads/                # Directorio de subidas temporales de la API
├── temp/                   # Archivos temporales de audio normalizados (.wav)
├── data/
│   ├── real/               # Dataset de audios humanos
│   └── ia/                 # Dataset de audios generados por IA
│
├── results/                # Directorio de resultados
│   ├── real/               # Reportes JSON individuales de audios reales
│   ├── ia/                 # Reportes JSON individuales de audios de IA
│   └── reporte_global.json # Resumen consolidado del dataset completo
│
└── requirements.txt        # Librerías y dependencias
```

### Nota sobre jitter y shimmer
Este MVP calcula jitter y shimmer como variación **frame a frame** (25ms), no ciclo-a-ciclo glótico real como haría Praat. Es una aproximación suficiente para el MVP y sigue siendo comparable entre voz real e IA, pero si se necesita jitter/shimmer clínico habría que implementar detección de picos pitch-síncrona (marcado de cada ciclo glótico individual). Ver docstrings de `jitter.py` y `shimmer.py` para más detalle.

Principio de diseño: cada módulo tiene una única responsabilidad, de forma que agregar una nueva característica no requiere tocar `audio.py`, `routes.py` ni ningún otro módulo existente — solo se agrega el archivo nuevo en `app/features/`, se implementa su lógica de extracción y se registra en `app/analysis.py`.
