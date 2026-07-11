"""
config.py
---------
Configuración centralizada del proyecto.

Responsabilidad única: definir constantes y rutas que usan otros módulos.
Ningún otro archivo debe "hardcodear" estos valores directamente,
así si algo cambia (ej. sample rate objetivo) se cambia en un solo lugar.
"""

from pathlib import Path

# --- Rutas base del proyecto ---
BASE_DIR = Path(__file__).resolve().parent.parent

UPLOADS_DIR = BASE_DIR / "uploads"
TEMP_DIR = BASE_DIR / "temp"
RESULTS_DIR = BASE_DIR / "results"

# Crear carpetas si no existen (evita errores en primera ejecución)
for _dir in (UPLOADS_DIR, TEMP_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- Parámetros de audio estándar del pipeline ---
TARGET_SAMPLE_RATE = 16000  # Hz
TARGET_CHANNELS = 1         # Mono
TARGET_SAMPLE_WIDTH = 2     # 16-bit PCM (2 bytes por muestra)

# --- Formatos de entrada aceptados ---
# pydub/ffmpeg soportan todos estos; se valida la extensión antes de procesar.
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".aac", ".m4a", ".ogg", ".opus", ".flac"}

# Tamaño máximo de archivo aceptado (en bytes). Ajustable según necesidad.
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

# --- Parámetros de framing (ventaneo) ---
# Usados por todas las features que trabajan frame a frame
# (RMS, pitch, jitter, shimmer, ZCR). Centralizados aquí para que
# todas las características se calculen sobre la misma segmentación temporal.
FRAME_SIZE_MS = 25   # duración de cada ventana de análisis
HOP_SIZE_MS = 10      # desplazamiento entre ventanas consecutivas (solapamiento)

# --- Parámetros de detección de Pitch (F0) ---
# Rango típico de la voz humana en telefonía (fundamental).
PITCH_FMIN_HZ = 75    # por debajo de esto ya no es voz humana típica
PITCH_FMAX_HZ = 400   # cubre voces graves y agudas con margen
# Umbral de "fuerza de periodicidad" (autocorrelación normalizada) para
# considerar un frame como voz sonora (voiced). Por debajo se considera
# silencio/ruido/consonante sorda y se descarta (no tiene pitch definido).
PITCH_VOICED_THRESHOLD = 0.3

# --- Parámetros de detección de Silencios/Pausas ---
# Un frame se considera "silencio" si su RMS cae por debajo de este umbral.
# Se reutiliza el mismo framing (FRAME_SIZE_MS / HOP_SIZE_MS) que RMS.
SILENCE_RMS_THRESHOLD = 0.02
# Pausas más cortas que esto se descartan (evita contar micro-caídas de
# energía entre sílabas como si fueran pausas reales de habla).
MIN_SILENCE_DURATION_MS = 100

# --- Umbral de clasificación heurística ---
# Punto medio de MAD para RMS (promedio REAL ~0.0339, promedio IA ~0.0198)
RMS_MAD_THRESHOLD = 0.0272

