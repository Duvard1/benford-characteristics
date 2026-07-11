"""
audio.py

Responsabilidad única: conversión y normalización de audio.

Flujo:
    Archivo de entrada (AAC, MP3, WAV, M4A, OGG, OPUS, FLAC...)
        ↓
    Validación (extensión, tamaño)
        ↓
    Conversión a WAV PCM (vía pydub -> ffmpeg, invocado desde Python,
    NO mediante comandos manuales de terminal)
        ↓
    Normalización: mono, 16kHz, 16-bit PCM
        ↓
    Lectura de muestras -> numpy array

Este módulo NO calcula pitch, jitter, RMS, ni ninguna otra característica.
Eso corresponde a los módulos dentro de app/features/ (fases posteriores).

Requisito de sistema: ffmpeg debe estar instalado y accesible en el PATH,
ya que pydub delega en él para decodificar formatos comprimidos (AAC, MP3, etc).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from app.config import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    TARGET_SAMPLE_WIDTH,
    TEMP_DIR,
)
from app.utils import eliminar_archivo_seguro, generar_nombre_unico


# --- Excepciones específicas del módulo ---

class AudioValidationError(Exception):
    """Se lanza cuando el archivo de entrada no cumple los requisitos básicos."""
    pass


class AudioConversionError(Exception):
    """Se lanza cuando pydub/ffmpeg falla al decodificar o convertir el audio."""
    pass


# --- Estructura de salida del módulo ---

@dataclass
class AudioNormalizado:
    """
    Representa el resultado final de este módulo: audio ya normalizado
    y listo para que otros módulos (features/*) lo consuman.
    """
    samples: np.ndarray      # muestras PCM (float32, rango [-1, 1])
    sample_rate: int         # siempre TARGET_SAMPLE_RATE tras normalizar
    duration_seconds: float
    wav_path: Path           # ruta al .wav normalizado en disco (temp/)


# --- Validación ---

def validar_archivo(filename: str, contenido: bytes) -> None:
    """
    Valida extensión y tamaño del archivo recibido.
    No valida el contenido interno (eso lo hace la conversión al fallar).

    Raises:
        AudioValidationError si algo no cumple los requisitos.
    """
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise AudioValidationError(
            f"Extensión '{extension}' no soportada. "
            f"Formatos permitidos: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    if len(contenido) == 0:
        raise AudioValidationError("El archivo está vacío.")

    if len(contenido) > MAX_FILE_SIZE_BYTES:
        raise AudioValidationError(
            f"El archivo excede el tamaño máximo permitido "
            f"({MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB)."
        )


# --- Conversión ---

def _guardar_temporal(contenido: bytes, extension: str) -> Path:
    """Guarda los bytes recibidos como archivo temporal en disco."""
    nombre = generar_nombre_unico(extension)
    ruta = TEMP_DIR / nombre
    ruta.write_bytes(contenido)
    return ruta


def convertir_a_wav_pcm(ruta_entrada: Path) -> Path:
    """
    Convierte cualquier formato soportado a WAV PCM mono 16kHz 16-bit,
    usando pydub (que internamente invoca ffmpeg, pero siempre desde Python,
    nunca mediante un comando manual escrito por el usuario).

    Args:
        ruta_entrada: ruta al archivo original (temp/)

    Returns:
        Ruta al nuevo archivo .wav normalizado (temp/)

    Raises:
        AudioConversionError si ffmpeg/pydub no logran decodificar el archivo.
    """
    try:
        audio = AudioSegment.from_file(ruta_entrada)
    except Exception as exc:
        raise AudioConversionError(
            f"No se pudo decodificar el archivo '{ruta_entrada.name}'. "
            f"¿Está corrupto o el códec no es soportado? Detalle: {exc}"
        ) from exc

    # Normalización: mono, 16kHz, 16-bit PCM
    audio = audio.set_channels(TARGET_CHANNELS)
    audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
    audio = audio.set_sample_width(TARGET_SAMPLE_WIDTH)

    ruta_salida = TEMP_DIR / generar_nombre_unico(".wav")

    try:
        audio.export(ruta_salida, format="wav")
    except Exception as exc:
        raise AudioConversionError(
            f"Fallo al exportar el WAV normalizado: {exc}"
        ) from exc

    return ruta_salida


def leer_muestras(ruta_wav: Path) -> tuple[np.ndarray, int]:
    """
    Lee un WAV PCM y devuelve las muestras como float32 normalizado en [-1, 1].

    Returns:
        (samples, sample_rate)
    """
    samples, sample_rate = sf.read(ruta_wav, dtype="float32", always_2d=False)
    return samples, sample_rate


# --- Punto de entrada del módulo ---

def procesar_audio(
    filename: str,
    contenido: bytes,
    limpiar_original: bool = True,
) -> AudioNormalizado:
    """
    Función principal del módulo. Orquesta validación -> conversión -> lectura.

    Args:
        filename: nombre original del archivo (para inferir extensión)
        contenido: bytes crudos del archivo recibido (ej. desde un UploadFile)
        limpiar_original: si True, borra el archivo temporal original
                           tras convertirlo (mantiene solo el .wav normalizado)

    Returns:
        AudioNormalizado listo para ser consumido por app/features/*

    Raises:
        AudioValidationError, AudioConversionError
    """
    validar_archivo(filename, contenido)

    extension = Path(filename).suffix.lower()
    ruta_original = _guardar_temporal(contenido, extension)

    try:
        ruta_wav = convertir_a_wav_pcm(ruta_original)
    finally:
        if limpiar_original:
            eliminar_archivo_seguro(ruta_original)

    samples, sample_rate = leer_muestras(ruta_wav)
    duracion = len(samples) / sample_rate if sample_rate else 0.0

    return AudioNormalizado(
        samples=samples,
        sample_rate=sample_rate,
        duration_seconds=duracion,
        wav_path=ruta_wav,
    )


def procesar_audio_desde_ruta(ruta_archivo: Union[str, Path]) -> AudioNormalizado:
    """
    Variante de conveniencia para procesar un archivo que ya está en disco
    (útil para pruebas locales o para procesar data/real/ y data/ia/ en batch).
    """
    ruta_archivo = Path(ruta_archivo)
    contenido = ruta_archivo.read_bytes()
    return procesar_audio(ruta_archivo.name, contenido, limpiar_original=False)
