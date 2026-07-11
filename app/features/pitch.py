"""
pitch.py
--------
Feature 1 del proyecto: Pitch / Frecuencia fundamental (F0).

Responsabilidad única: estimar F0 por frame mediante autocorrelación.

Método: autocorrelación normalizada (sin dependencias externas de
detección de pitch tipo librosa/parselmouth, para mantener el MVP simple).

Flujo:
    samples (audio normalizado)
        ↓
    segmentar_en_frames()
        ↓
    por cada frame: autocorrelación -> pico dominante en rango F0 humano
        ↓
    F0 (Hz) si el frame es sonoro (voiced), NaN si no (silencio/consonante sorda)
        ↓
    vector [f0_1, f0_2, NaN, f0_4, ...] -> filtrar NaN -> listo para Benford
"""

import numpy as np

from app.config import PITCH_FMAX_HZ, PITCH_FMIN_HZ, PITCH_VOICED_THRESHOLD
from app.features.framing import segmentar_en_frames


def _estimar_f0_frame(
    frame: np.ndarray,
    sample_rate: int,
    fmin: int = PITCH_FMIN_HZ,
    fmax: int = PITCH_FMAX_HZ,
    umbral_voiced: float = PITCH_VOICED_THRESHOLD,
) -> float:
    """
    Estima F0 de un único frame vía autocorrelación.

    Returns:
        F0 en Hz si el frame es sonoro y periódico dentro del rango
        [fmin, fmax]. np.nan si el frame es silencio, ruido, o una
        consonante sorda sin periodicidad clara.
    """
    frame = frame.astype(np.float64) - np.mean(frame)

    # Frame prácticamente silencioso: no hay pitch que estimar
    if np.max(np.abs(frame)) < 1e-4:
        return np.nan

    # Autocorrelación completa, nos quedamos con la mitad no negativa (lags >= 0)
    corr = np.correlate(frame, frame, mode="full")
    corr = corr[len(corr) // 2:]

    if corr[0] <= 0:
        return np.nan

    lag_min = int(sample_rate / fmax)
    lag_max = int(sample_rate / fmin)
    lag_max = min(lag_max, len(corr) - 1)

    if lag_min >= lag_max:
        return np.nan

    segmento = corr[lag_min:lag_max]
    if segmento.size == 0:
        return np.nan

    pico_idx_relativo = int(np.argmax(segmento))
    pico_idx = pico_idx_relativo + lag_min

    fuerza_periodicidad = corr[pico_idx] / corr[0]

    if fuerza_periodicidad < umbral_voiced:
        # No hay periodicidad suficiente: frame no sonoro
        return np.nan

    f0 = sample_rate / pico_idx
    return float(f0)


def calcular_pitch_por_frame(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Calcula F0 para cada frame de la señal.

    Args:
        samples: señal de audio mono, float32 en [-1, 1]
        sample_rate: frecuencia de muestreo (Hz)

    Returns:
        Array 1D de igual longitud que el número de frames. Contiene el
        valor de F0 en Hz para frames sonoros, y np.nan para frames sin
        pitch detectable (silencios, consonantes sordas, ruido).
    """
    frames = segmentar_en_frames(samples, sample_rate)

    pitch = np.array(
        [_estimar_f0_frame(frame, sample_rate) for frame in frames],
        dtype=np.float64,
    )
    return pitch


def pitch_valores_sonoros(pitch: np.ndarray) -> np.ndarray:
    """
    Filtra únicamente los frames sonoros (descarta np.nan).

    Este es el vector que debe alimentar a Benford: no tiene sentido
    incluir "silencios" (NaN) en un análisis de primer dígito significativo.
    """
    return pitch[~np.isnan(pitch)]


def resumen_pitch(pitch: np.ndarray) -> dict:
    """
    Estadísticas descriptivas rápidas del pitch. Solo informativo/debug,
    no reemplaza el análisis de Benford (fase posterior).
    """
    sonoros = pitch_valores_sonoros(pitch)

    if sonoros.size == 0:
        return {
            "num_frames_totales": int(pitch.size),
            "num_frames_sonoros": 0,
            "ratio_sonoro": 0.0,
        }

    return {
        "num_frames_totales": int(pitch.size),
        "num_frames_sonoros": int(sonoros.size),
        "ratio_sonoro": round(float(sonoros.size / pitch.size), 3),
        "f0_min_hz": round(float(np.min(sonoros)), 2),
        "f0_max_hz": round(float(np.max(sonoros)), 2),
        "f0_mean_hz": round(float(np.mean(sonoros)), 2),
        "f0_std_hz": round(float(np.std(sonoros)), 2),
    }
