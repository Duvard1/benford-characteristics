"""
energy.py
---------
Feature 4 del proyecto: Energía RMS por frame.

Responsabilidad única: calcular el RMS (Root Mean Square) de cada frame
de la señal, es decir, la evolución temporal de la energía — NO un único
valor de RMS para todo el audio.

Flujo:
    samples (audio normalizado)
        ↓
    segmentar_en_frames()
        ↓
    RMS por frame
        ↓
    vector [rms_1, rms_2, rms_3, ...]  (listo para Ley de Benford)
"""

import numpy as np

from app.features.framing import segmentar_en_frames


def calcular_rms_por_frame(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Calcula el RMS de cada frame de la señal.

    Args:
        samples: señal de audio mono, float32 en [-1, 1]
        sample_rate: frecuencia de muestreo (Hz)

    Returns:
        Array 1D con el valor RMS de cada frame.
        RMS de un frame = sqrt(mean(frame^2))
    """
    frames = segmentar_en_frames(samples, sample_rate)

    # RMS por frame: raíz cuadrada de la media de los cuadrados
    rms = np.sqrt(np.mean(np.square(frames), axis=1))

    return rms


def resumen_rms(rms: np.ndarray) -> dict:
    """
    Estadísticas descriptivas rápidas del vector RMS.
    Útil para inspección/debug antes de aplicar Benford (fase posterior).
    No reemplaza el análisis de Benford, solo da contexto legible.
    """
    if rms.size == 0:
        return {"num_frames": 0}

    return {
        "num_frames": int(rms.size),
        "rms_min": float(np.min(rms)),
        "rms_max": float(np.max(rms)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
    }
