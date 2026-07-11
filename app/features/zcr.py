"""
zcr.py
------
Feature 5 del proyecto: Zero Crossing Rate (tasa de cruces por cero).

Responsabilidad única: medir, por frame, cuántas veces la señal cruza el
eje cero, normalizado por el número de muestras del frame. Puede aportar
información sobre ruido, fricción y articulación.

Flujo:
    samples (audio normalizado)
        ↓
    segmentar_en_frames()
        ↓
    ZCR por frame = (cruces por cero) / (frame_length - 1)
        ↓
    vector [zcr_1, zcr_2, zcr_3, ...] -> listo para Ley de Benford
"""

import numpy as np

from app.features.framing import segmentar_en_frames


def calcular_zcr_por_frame(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Calcula la tasa de cruces por cero de cada frame.

    Args:
        samples: señal de audio mono, float32 en [-1, 1]
        sample_rate: frecuencia de muestreo (Hz)

    Returns:
        Array 1D con el ZCR de cada frame, en rango [0, 1].
    """
    frames = segmentar_en_frames(samples, sample_rate)

    # np.sign da -1, 0, 1. Tratamos el 0 como positivo para no generar
    # cruces artificiales en tramos de silencio digital exacto.
    signos = np.sign(frames)
    signos[signos == 0] = 1

    cambios_de_signo = np.abs(np.diff(signos, axis=1)) > 0
    cruces = np.sum(cambios_de_signo, axis=1)

    frame_length = frames.shape[1]
    zcr = cruces / (frame_length - 1)

    return zcr


def resumen_zcr(zcr: np.ndarray) -> dict:
    """
    Estadísticas descriptivas rápidas del ZCR. Solo informativo/debug.
    """
    if zcr.size == 0:
        return {"num_frames": 0}

    return {
        "num_frames": int(zcr.size),
        "zcr_min": round(float(np.min(zcr)), 6),
        "zcr_max": round(float(np.max(zcr)), 6),
        "zcr_mean": round(float(np.mean(zcr)), 6),
        "zcr_std": round(float(np.std(zcr)), 6),
    }
