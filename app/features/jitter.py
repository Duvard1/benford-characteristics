"""
jitter.py
---------
Feature 2 del proyecto: Jitter (irregularidad ciclo-a-ciclo del pitch).

Responsabilidad única: medir cuánto varía el período de la frecuencia
fundamental (F0) entre frames consecutivos que son ambos sonoros.

NOTA IMPORTANTE (limitación consciente del MVP):
El jitter "clínico" (el que usan herramientas como Praat) se mide entre
ciclos glóticos individuales, con resolución de submilisegundos. Aquí
nuestro pitch se estima por frame (25ms, hop 10ms; ver features/pitch.py),
así que este módulo mide variación del período de frame a frame, no
ciclo a ciclo real. Es una aproximación razonable para un MVP y sigue
siendo comparable entre voz real e IA, pero si más adelante se necesita
jitter clínico, habría que hacer detección de picos pitch-síncrona
(marcado de cada ciclo glótico individual).

Flujo:
    pitch_hz (vector completo, CON NaN para frames no sonoros; viene de
              features/pitch.py -> calcular_pitch_por_frame)
        ↓
    convertir F0 (Hz) -> período (ms) = 1000 / F0
        ↓
    para cada PAR de frames consecutivos donde AMBOS son sonoros:
        |período_i+1 - período_i|
        ↓
    vector [jitter_1_ms, jitter_2_ms, ...] -> listo para Ley de Benford
"""

import numpy as np


def calcular_jitter_ciclo_a_ciclo(pitch_hz: np.ndarray) -> np.ndarray:
    """
    Calcula la diferencia absoluta de período (ms) entre frames de pitch
    consecutivos, solo cuando ambos frames son sonoros.

    Si hay un frame no sonoro (NaN) entre dos sonoros, ese par se
    descarta: no tiene sentido comparar periodicidad a través de un hueco
    de silencio.

    Args:
        pitch_hz: array de F0 por frame, con np.nan en frames no sonoros
                   (salida directa de calcular_pitch_por_frame en pitch.py)

    Returns:
        Array 1D con las diferencias absolutas de período (en ms) entre
        pares de frames sonoros consecutivos.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        periodo_ms = np.where(pitch_hz > 0, 1000.0 / pitch_hz, np.nan)

    diferencias = []
    for i in range(len(periodo_ms) - 1):
        actual = periodo_ms[i]
        siguiente = periodo_ms[i + 1]
        if not np.isnan(actual) and not np.isnan(siguiente):
            diferencias.append(abs(siguiente - actual))

    return np.array(diferencias, dtype=np.float64)


def resumen_jitter(pitch_hz: np.ndarray, diferencias_ms: np.ndarray) -> dict:
    """
    Estadísticas descriptivas del jitter, incluyendo el jitter relativo
    (%), que es la forma más común de reportarlo en literatura de voz:
    jitter_relativo = promedio(|ΔT|) / promedio(T) * 100
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        periodo_ms = np.where(pitch_hz > 0, 1000.0 / pitch_hz, np.nan)
    periodos_validos = periodo_ms[~np.isnan(periodo_ms)]

    if diferencias_ms.size == 0 or periodos_validos.size == 0:
        return {"num_pares_validos": 0, "jitter_relativo_pct": 0.0}

    periodo_medio_ms = float(np.mean(periodos_validos))
    jitter_absoluto_medio_ms = float(np.mean(diferencias_ms))
    jitter_relativo_pct = (
        (jitter_absoluto_medio_ms / periodo_medio_ms) * 100
        if periodo_medio_ms > 0
        else 0.0
    )

    return {
        "num_pares_validos": int(diferencias_ms.size),
        "periodo_medio_ms": round(periodo_medio_ms, 4),
        "jitter_absoluto_medio_ms": round(jitter_absoluto_medio_ms, 4),
        "jitter_relativo_pct": round(jitter_relativo_pct, 4),
    }
