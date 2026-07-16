# Define constantes y rutas usadas por otros módulos
from pathlib import Path

# Rutas base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
TEMP_DIR = BASE_DIR / "temp"
RESULTS_DIR = BASE_DIR / "results"
# Crear carpetas si no existen
for _dir in (UPLOADS_DIR, TEMP_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Parámetros de audio estándar
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2     # 16-bit PCM (2 bytes por muestra)

# Formatos de entrada aceptados
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".aac", ".m4a", ".ogg", ".opus", ".flac"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

# Framing o Ventaneo
FRAME_SIZE_MS = 25   # duración de cada ventana de análisis
HOP_SIZE_MS = 10      # desplazamiento entre ventanas consecutivas (solapamiento)

# Parámetros de detección de Pitch
# Rango típico de la voz humana en telefonía (fundamental).
PITCH_FMIN_HZ = 75    # por debajo de esto ya no es voz humana típica
PITCH_FMAX_HZ = 400   # cubre voces graves y agudas con margen
PITCH_VOICED_THRESHOLD = 0.3 # umbral para saber si un fragmento tienen voz

# Parámetros de detección de Silencios
SILENCE_RMS_THRESHOLD = 0.02
# minimo de duracion para que se considere silencio
MIN_SILENCE_DURATION_MS = 100

# --- Umbral de clasificación heurística ---
# Punto medio de MAD para RMS (promedio REAL ~0.0339, promedio IA ~0.0198)
RMS_MAD_THRESHOLD = 0.0272

