"""
silence.py
----------
Feature 6 del proyecto: Silencios y pausas.

Responsabilidad única: detectar tramos de silencio a partir del RMS por
frame (reutiliza app/features/energy.py, no vuelve a calcular RMS) y medir
la duración de cada pausa. Especialmente relevante en llamadas telefónicas.

Flujo:
    RMS por frame (de energy.py)
        ↓
    frame considerado "silencio" si RMS < SILENCE_RMS_THRESHOLD
        ↓
    agrupar frames de silencio consecutivos -> runs
        ↓
    convertir cada run a duración en milisegundos
        ↓
    descartar pausas menores a MIN_SILENCE_DURATION_MS
        ↓
    vector [dur_1_ms, dur_2_ms, ...] -> listo para Ley de Benford
"""

import numpy as np

from app.config import (
    FRAME_SIZE_MS,
    HOP_SIZE_MS,
    MIN_SILENCE_DURATION_MS,
    SILENCE_RMS_THRESHOLD,
)
from app.features.energy import calcular_rms_por_frame


def _runs_de_silencio(mascara_silencio: np.ndarray) -> np.ndarray:
    """
    Dado un array booleano (True = frame de silencio), devuelve la longitud
    (en número de frames) de cada tramo consecutivo de silencio.

    Ejemplo: [F,F,T,T,T,F,T,T] -> runs de silencio: [3, 2]
    """
    if mascara_silencio.size == 0:
        return np.array([], dtype=np.int64)

    extendida = np.concatenate(([0], mascara_silencio.astype(int), [0]))
    diffs = np.diff(extendida)

    inicios = np.where(diffs == 1)[0]
    fines = np.where(diffs == -1)[0]

    return fines - inicios  # longitud de cada run, en frames


def _run_a_duracion_ms(longitud_frames: int, frame_ms: int, hop_ms: int) -> float:
    """
    Convierte una longitud de run (en número de frames) a duración real
    en milisegundos, considerando que el primer frame del run cubre
    'frame_ms' completos y cada frame adicional solo aporta 'hop_ms' más
    (por el solapamiento entre frames consecutivos).
    """
    return frame_ms + (longitud_frames - 1) * hop_ms


def detectar_pausas(
    samples: np.ndarray,
    sample_rate: int,
    rms_threshold: float = SILENCE_RMS_THRESHOLD,
    min_silence_ms: float = MIN_SILENCE_DURATION_MS,
) -> np.ndarray:
    """
    Detecta las pausas/silencios de la señal y devuelve sus duraciones.

    Args:
        samples: señal de audio mono, float32 en [-1, 1]
        sample_rate: frecuencia de muestreo (Hz)
        rms_threshold: RMS por debajo del cual un frame se considera silencio
        min_silence_ms: duración mínima para contar una pausa como tal

    Returns:
        Array 1D con la duración (en ms) de cada pausa detectada,
        ya filtrando las demasiado cortas.
    """
    rms = calcular_rms_por_frame(samples, sample_rate)
    mascara_silencio = rms < rms_threshold

    longitudes_runs = _runs_de_silencio(mascara_silencio)

    duraciones_ms = np.array(
        [_run_a_duracion_ms(l, FRAME_SIZE_MS, HOP_SIZE_MS) for l in longitudes_runs],
        dtype=np.float64,
    )

    duraciones_ms = duraciones_ms[duraciones_ms >= min_silence_ms]
    return duraciones_ms


def resumen_silencios(duraciones_ms: np.ndarray, duracion_total_seg: float) -> dict:
    """
    Estadísticas descriptivas rápidas de las pausas detectadas.
    Solo informativo/debug, no reemplaza el análisis de Benford.
    """
    if duraciones_ms.size == 0:
        return {
            "num_pausas": 0,
            "tiempo_total_silencio_ms": 0.0,
            "ratio_silencio": 0.0,
        }

    tiempo_total_silencio_ms = float(np.sum(duraciones_ms))
    duracion_total_ms = duracion_total_seg * 1000 if duracion_total_seg > 0 else 1.0

    return {
        "num_pausas": int(duraciones_ms.size),
        "duracion_min_ms": round(float(np.min(duraciones_ms)), 1),
        "duracion_max_ms": round(float(np.max(duraciones_ms)), 1),
        "duracion_mean_ms": round(float(np.mean(duraciones_ms)), 1),
        "tiempo_total_silencio_ms": round(tiempo_total_silencio_ms, 1),
        "ratio_silencio": round(tiempo_total_silencio_ms / duracion_total_ms, 3),
    }
