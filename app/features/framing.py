"""
framing.py
----------
Responsabilidad única: segmentar una señal de audio en frames (ventanas).

Todas las características frame-a-frame (RMS, pitch, jitter, shimmer, ZCR)
comparten la misma lógica de segmentación temporal. Se centraliza aquí para
que:
    - todas las features usen exactamente los mismos frames (comparables
      entre sí temporalmente)
    - agregar una nueva feature no implique reescribir el ventaneo

No depende de audio.py ni de ninguna feature concreta.
"""

import numpy as np

from app.config import FRAME_SIZE_MS, HOP_SIZE_MS


def segmentar_en_frames(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: int = FRAME_SIZE_MS,
    hop_ms: int = HOP_SIZE_MS,
) -> np.ndarray:
    """
    Divide una señal 1D en frames (ventanas) con solapamiento.

    Args:
        samples: señal de audio mono, float32 en [-1, 1]
        sample_rate: frecuencia de muestreo (Hz)
        frame_ms: duración de cada frame en milisegundos
        hop_ms: desplazamiento entre frames consecutivos en milisegundos

    Returns:
        Array 2D de forma (num_frames, frame_length). Si la señal es más
        corta que un frame, devuelve un único frame con padding de ceros.
    """
    frame_length = int(sample_rate * frame_ms / 1000)
    hop_length = int(sample_rate * hop_ms / 1000)

    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_ms y hop_ms deben producir tamaños > 0 muestras.")

    if len(samples) < frame_length:
        # Señal más corta que un frame: se rellena con ceros (padding)
        relleno = np.zeros(frame_length - len(samples), dtype=samples.dtype)
        return np.expand_dims(np.concatenate([samples, relleno]), axis=0)

    num_frames = 1 + (len(samples) - frame_length) // hop_length

    frames = np.stack(
        [
            samples[i * hop_length: i * hop_length + frame_length]
            for i in range(num_frames)
        ]
    )
    return frames


def frame_a_tiempo(indice_frame: int, sample_rate: int, hop_ms: int = HOP_SIZE_MS) -> float:
    """
    Convierte un índice de frame a su marca de tiempo (segundos) de inicio.
    Útil para reportar en qué momento de la llamada ocurre cada valor.
    """
    hop_length = int(sample_rate * hop_ms / 1000)
    return (indice_frame * hop_length) / sample_rate
