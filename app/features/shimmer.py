"""
shimmer.py
----------
Feature 3 del proyecto: Shimmer (irregularidad ciclo-a-ciclo de amplitud).

Responsabilidad única: medir cuánto varía la amplitud (usamos RMS por
frame como proxy) entre frames consecutivos que son ambos sonoros.

NOTA IMPORTANTE (misma limitación consciente que jitter.py):
El shimmer clínico se mide entre picos de amplitud de ciclos glóticos
individuales. Aquí usamos RMS por frame (25ms) como proxy de amplitud,
así que este módulo mide variación de energía de frame a frame, no
ciclo a ciclo real. Aproximación razonable para el MVP.

Solo se comparan frames marcados como "sonoros" según pitch.py, porque
shimmer solo tiene sentido fisiológico durante fonación sostenida
(no durante silencios o consonantes sordas).

Flujo:
    rms (vector completo, de features/energy.py)
    pitch_hz (vector completo, CON NaN; de features/pitch.py, usado
              únicamente para saber qué frames son sonoros)
        ↓
    para cada PAR de frames consecutivos donde AMBOS son sonoros:
        |rms_i+1 - rms_i|
        ↓
    vector [shimmer_1, shimmer_2, ...] -> listo para Ley de Benford
"""

import numpy as np


def calcular_shimmer_ciclo_a_ciclo(rms: np.ndarray, pitch_hz: np.ndarray) -> np.ndarray:
    """
    Calcula la diferencia absoluta de RMS entre frames consecutivos,
    solo cuando ambos frames son sonoros (según pitch_hz).

    Args:
        rms: array de RMS por frame (misma longitud y framing que pitch_hz)
        pitch_hz: array de F0 por frame, con np.nan en frames no sonoros
                   (usado únicamente para identificar frames sonoros)

    Returns:
        Array 1D con las diferencias absolutas de RMS entre pares de
        frames sonoros consecutivos.
    """
    if len(rms) != len(pitch_hz):
        raise ValueError(
            f"rms ({len(rms)} frames) y pitch_hz ({len(pitch_hz)} frames) "
            "deben tener la misma longitud (mismo framing)."
        )

    sonoro = ~np.isnan(pitch_hz)

    diferencias = []
    for i in range(len(rms) - 1):
        if sonoro[i] and sonoro[i + 1]:
            diferencias.append(abs(float(rms[i + 1]) - float(rms[i])))

    return np.array(diferencias, dtype=np.float64)


def resumen_shimmer(rms: np.ndarray, pitch_hz: np.ndarray, diferencias: np.ndarray) -> dict:
    """
    Estadísticas descriptivas del shimmer, incluyendo el shimmer relativo
    (%): shimmer_relativo = promedio(|ΔRMS|) / promedio(RMS_sonoro) * 100
    """
    sonoro = ~np.isnan(pitch_hz)
    rms_sonoro = rms[sonoro]

    if diferencias.size == 0 or rms_sonoro.size == 0:
        return {"num_pares_validos": 0, "shimmer_relativo_pct": 0.0}

    rms_medio = float(np.mean(rms_sonoro))
    shimmer_absoluto_medio = float(np.mean(diferencias))
    shimmer_relativo_pct = (
        (shimmer_absoluto_medio / rms_medio) * 100 if rms_medio > 0 else 0.0
    )

    return {
        "num_pares_validos": int(diferencias.size),
        "rms_medio_sonoro": round(rms_medio, 6),
        "shimmer_absoluto_medio": round(shimmer_absoluto_medio, 6),
        "shimmer_relativo_pct": round(shimmer_relativo_pct, 4),
    }
