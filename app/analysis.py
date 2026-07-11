"""
analysis.py
-----------
Responsabilidad única: orquestar, para UNA señal de audio ya normalizada,
el cálculo de las 6 características + Ley de Benford + métricas.

Este módulo es el punto de unión entre app/features/*, app/benford/ y
app/metrics/. Se extrajo de routes.py para que tanto el endpoint HTTP
(una petición a la vez) como scripts/evaluar_dataset.py (muchos archivos
en batch) usen exactamente la misma lógica de análisis — así no hay dos
implementaciones que puedan desincronizarse.

No sabe nada de HTTP, FastAPI, ni de cómo se subió el archivo.
"""

import numpy as np

from app.benford.benford import analizar_benford
from app.features.energy import calcular_rms_por_frame, resumen_rms
from app.features.jitter import calcular_jitter_ciclo_a_ciclo, resumen_jitter
from app.features.pitch import (
    calcular_pitch_por_frame,
    pitch_valores_sonoros,
    resumen_pitch,
)
from app.features.shimmer import calcular_shimmer_ciclo_a_ciclo, resumen_shimmer
from app.features.silence import detectar_pausas, resumen_silencios
from app.features.zcr import calcular_zcr_por_frame, resumen_zcr
from app.metrics.metrics import calcular_todas_las_metricas


def _benford_y_metricas(valores) -> dict:
    """
    Aplica Benford + las 4 métricas (MAD, chi2, KL, JS) a un vector de
    una característica (pitch, RMS, etc).

    Si no hay suficientes valores válidos, devuelve un error legible
    en vez de romper todo el análisis.
    """
    try:
        benford = analizar_benford(valores)
    except ValueError as exc:
        return {"error": str(exc)}

    metricas = calcular_todas_las_metricas(
        observado=benford["distribucion_observada"],
        esperado=benford["distribucion_teorica"],
        n=benford["n_valores"],
    )

    return {
        "n_valores": benford["n_valores"],
        "distribucion_observada": [round(p, 4) for p in benford["distribucion_observada"]],
        "distribucion_teorica": [round(p, 4) for p in benford["distribucion_teorica"]],
        "metricas": metricas,
    }


def _bloque_feature(resumen: dict, valores, incluir_valores: bool, decimales: int = 6) -> dict:
    """
    Arma el bloque de respuesta de UNA característica: 'resumen' y 'benford'
    siempre; el array de valores crudos SOLO si se pide explícitamente.
    """
    bloque = {
        "resumen": resumen,
        "benford": _benford_y_metricas(valores),
    }
    if incluir_valores:
        bloque["valores"] = [round(float(v), decimales) for v in valores]
    return bloque


def analizar_features(
    samples: np.ndarray,
    sample_rate: int,
    duracion_segundos: float,
    incluir_valores: bool = False,
) -> dict:
    """
    Punto de entrada del módulo. Calcula las 6 características sobre una
    señal ya normalizada y devuelve el bloque "features" completo
    (mismo formato que consume la API y el script batch).

    Args:
        samples: señal de audio mono, float32 en [-1, 1], normalizada
        sample_rate: frecuencia de muestreo (Hz)
        duracion_segundos: duración total del audio (para calcular ratios)
        incluir_valores: si True, incluye los vectores crudos de cada feature

    Returns:
        dict con una clave por característica (rms, pitch, zcr, silencios,
        jitter, shimmer), cada una con 'resumen' y 'benford' (y 'valores'
        si se pidió).
    """
    rms = calcular_rms_por_frame(samples, sample_rate)
    pitch = calcular_pitch_por_frame(samples, sample_rate)
    pitch_sonoro = pitch_valores_sonoros(pitch)
    zcr = calcular_zcr_por_frame(samples, sample_rate)
    pausas_ms = detectar_pausas(samples, sample_rate)
    jitter_ms = calcular_jitter_ciclo_a_ciclo(pitch)
    shimmer_valores = calcular_shimmer_ciclo_a_ciclo(rms, pitch)

    return {
        "rms": _bloque_feature(resumen_rms(rms), rms, incluir_valores),
        "pitch": _bloque_feature(
            resumen_pitch(pitch), pitch_sonoro, incluir_valores, decimales=3
        ),
        "zcr": _bloque_feature(resumen_zcr(zcr), zcr, incluir_valores),
        "silencios": _bloque_feature(
            resumen_silencios(pausas_ms, duracion_segundos),
            pausas_ms,
            incluir_valores,
            decimales=1,
        ),
        "jitter": _bloque_feature(
            resumen_jitter(pitch, jitter_ms), jitter_ms, incluir_valores, decimales=4
        ),
        "shimmer": _bloque_feature(
            resumen_shimmer(rms, pitch, shimmer_valores), shimmer_valores, incluir_valores
        ),
    }