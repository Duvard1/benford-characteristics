"""
routes.py
---------
Responsabilidad única: definir los endpoints de la API (FastAPI) y orquestar
las llamadas a audio.py, app/features/*, app/benford/ y app/metrics/.

Cada uno de esos módulos no sabe nada de los demás; routes.py es el único
lugar que los conecta entre sí.
"""

from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.audio import AudioConversionError, AudioValidationError, procesar_audio
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
from app.config import RMS_MAD_THRESHOLD


router = APIRouter()


def _benford_y_metricas(valores) -> dict:
    """
    Aplica Benford + las 4 métricas (MAD, chi2, KL, JS) a un vector de
    una característica (pitch, RMS, etc).

    Si no hay suficientes valores válidos, devuelve un error legible
    en vez de romper toda la petición.
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


def _bloque_feature(
    resumen: dict,
    valores,
) -> dict:
    """
    Arma el bloque de respuesta de UNA característica: 'resumen' y 'benford' siempre.
    El array de valores crudos se omite para mantener la respuesta compacta y limpia.
    """
    return {
        "resumen": resumen,
        "benford": _benford_y_metricas(valores),
    }


@router.post("/benford-char")
async def benford_char(
    file: UploadFile = File(...),
    caracteristica: Optional[str] = Query(
        None,
        description="Filtra la respuesta para devolver únicamente una característica específica: "
                    "rms, pitch, zcr, silencios, jitter, shimmer. Si no se especifica o se "
                    "envía 'todas', se devuelven todas las características.",
    ),
):
    """
    Recibe un archivo de audio, lo normaliza (mono, 16kHz, WAV PCM),
    extrae las características (RMS, Pitch, ZCR, Silencios, Jitter, Shimmer),
    aplica la Ley de Benford + métricas (MAD, chi2, KL, JS) a cada una
    y realiza una clasificación heurística.

    Mediante el parámetro opcional `caracteristica` se puede filtrar la respuesta
    para obtener solo la característica deseada (ej. `rms`, `pitch`, etc.).
    """
    contenido = await file.read()

    try:
        resultado = procesar_audio(filename=file.filename, contenido=contenido)
    except AudioValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AudioConversionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    rms = calcular_rms_por_frame(resultado.samples, resultado.sample_rate)
    pitch = calcular_pitch_por_frame(resultado.samples, resultado.sample_rate)
    pitch_sonoro = pitch_valores_sonoros(pitch)
    zcr = calcular_zcr_por_frame(resultado.samples, resultado.sample_rate)
    pausas_ms = detectar_pausas(resultado.samples, resultado.sample_rate)
    jitter_ms = calcular_jitter_ciclo_a_ciclo(pitch)
    shimmer_valores = calcular_shimmer_ciclo_a_ciclo(rms, pitch)

    features = {
        "rms": _bloque_feature(resumen_rms(rms), rms),
        "pitch": _bloque_feature(resumen_pitch(pitch), pitch_sonoro),
        "zcr": _bloque_feature(resumen_zcr(zcr), zcr),
        "silencios": _bloque_feature(
            resumen_silencios(pausas_ms, resultado.duration_seconds), pausas_ms
        ),
        "jitter": _bloque_feature(resumen_jitter(pitch, jitter_ms), jitter_ms),
        "shimmer": _bloque_feature(
            resumen_shimmer(rms, pitch, shimmer_valores), shimmer_valores
        ),
    }

    # Clasificación heurística basada en el MAD de RMS
    rms_mad = features["rms"]["benford"]["metricas"].get("mad")
    
    # Manejar caso de que 'rms' tenga error en benford
    if rms_mad is None:
        prediccion = "INDETERMINADO"
        confianza = "Baja"
        rms_mad = 0.0
    else:
        if rms_mad < RMS_MAD_THRESHOLD:
            prediccion = "IA"
        else:
            prediccion = "REAL"

        distancia = abs(rms_mad - RMS_MAD_THRESHOLD)

        if distancia < 0.003:
            confianza = "Baja"
        elif distancia < 0.008:
            confianza = "Media"
        else:
            confianza = "Alta"

    clasificacion = {
        "prediccion": prediccion,
        "confianza": confianza,
        "feature_utilizada": "RMS MAD",
        "valor_obtenido": rms_mad,
        "umbral": RMS_MAD_THRESHOLD,
    }

    # Filtrar características si se solicita una específica
    if caracteristica and caracteristica.lower() != "todas":
        caracteristica_lower = caracteristica.lower()
        if caracteristica_lower not in features:
            raise HTTPException(
                status_code=400,
                detail=f"Característica '{caracteristica}' no válida. Debe ser una de: {', '.join(features.keys())} o 'todas'.",
            )
        features = {caracteristica_lower: features[caracteristica_lower]}

    return {
        "archivo": {
            "nombre_original": file.filename,
            "sample_rate": resultado.sample_rate,
            "duracion_segundos": round(resultado.duration_seconds, 3),
            "num_muestras": int(resultado.samples.shape[0]),
            "wav_normalizado": resultado.wav_path.name,
        },
        "features": features,
        "clasificacion": clasificacion,
        "mensaje": "Audio normalizado y analizado exitosamente con la Ley de Benford.",
    }


@router.get("/health")
async def health():
    """Endpoint simple para verificar que la API está viva."""
    return {"status": "ok"}


