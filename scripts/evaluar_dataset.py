"""
evaluar_dataset.py
------------------
Responsabilidad única: evaluar en batch los conjuntos de datos en data/real/ y data/ia/,
procesando cada audio, extrayendo las características, aplicando Benford + métricas,
guardando los resultados individuales en JSON y mostrando un reporte comparativo global.

Uso:
    python -m scripts.evaluar_dataset
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Asegurar que el directorio raíz del proyecto esté en el path de Python
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Reconfigurar salida estándar para soportar UTF-8 en Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


from app.audio import procesar_audio_desde_ruta, AudioValidationError, AudioConversionError
from app.analysis import analizar_features
from app.config import BASE_DIR, RESULTS_DIR, ALLOWED_EXTENSIONS

# Rutas de datos
DATA_DIR = BASE_DIR / "data"
REAL_DIR = DATA_DIR / "real"
IA_DIR = DATA_DIR / "ia"

# Rutas de resultados estructurados
RESULTS_REAL_DIR = RESULTS_DIR / "real"
RESULTS_IA_DIR = RESULTS_DIR / "ia"


def inicializar_directorios():
    """Crea los directorios de salida si no existen."""
    RESULTS_REAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_IA_DIR.mkdir(parents=True, exist_ok=True)


def obtener_audios(directorio: Path) -> List[Path]:
    """Obtiene todos los archivos de audio soportados en un directorio."""
    if not directorio.exists():
        return []
    return [
        p for p in directorio.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    ]


def analizar_grupo(
    archivos: List[Path],
    dir_salida: Path,
    etiqueta: str
) -> List[Dict[str, Any]]:
    """
    Procesa un grupo de archivos (real o ia) de forma secuencial.
    Guarda los resultados de cada audio en un archivo JSON individual.
    """
    resultados_exitosos = []
    
    print(f"\nProcesando grupo '{etiqueta}' ({len(archivos)} archivos)...")
    
    for idx, ruta_audio in enumerate(archivos, 1):
        print(f"[{idx}/{len(archivos)}] Analizando: {ruta_audio.name} ... ", end="", flush=True)
        
        try:
            # 1. Normalización y lectura de muestras
            resultado_audio = procesar_audio_desde_ruta(ruta_audio)
            
            # 2. Extracción de características y análisis de Benford
            features = analizar_features(
                samples=resultado_audio.samples,
                sample_rate=resultado_audio.sample_rate,
                duracion_segundos=resultado_audio.duration_seconds,
                incluir_valores=False  # Compacto para evaluación masiva
            )
            
            # Estructurar resultado para guardar
            datos_audio = {
                "archivo": {
                    "nombre_original": ruta_audio.name,
                    "tipo": etiqueta,
                    "sample_rate": resultado_audio.sample_rate,
                    "duracion_segundos": round(resultado_audio.duration_seconds, 3),
                    "num_muestras": int(resultado_audio.samples.shape[0]),
                },
                "features": features
            }
            
            # 3. Guardar JSON individual
            ruta_json = dir_salida / f"{ruta_audio.stem}.json"
            ruta_json.write_text(json.dumps(datos_audio, indent=2, ensure_ascii=False), encoding="utf-8")
            
            resultados_exitosos.append(datos_audio)
            print("OK")
            
        except (AudioValidationError, AudioConversionError) as exc:
            print(f"ERROR (Controlado): {exc}")
        except Exception as exc:
            print(f"ERROR (Inesperado): {exc}")
            
    return resultados_exitosos


def calcular_promedios_metricas(resultados: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Calcula los promedios de MAD, chi2, KL y JS para cada característica
    dentro de un grupo analizado.
    """
    if not resultados:
        return {}
        
    features_keys = ["rms", "pitch", "zcr", "silencios", "jitter", "shimmer"]
    metricas_keys = ["mad", "chi2", "kl", "js"]
    
    # Inicializar estructuras de acumulación
    acumuladores = {f: {m: 0.0 for m in metricas_keys} for f in features_keys}
    contadores = {f: 0 for f in features_keys}
    
    for res in resultados:
        for f in features_keys:
            feat_data = res["features"].get(f, {})
            # Evitar frames sin suficientes datos para Benford (si tienen clave 'error')
            if "error" in feat_data.get("benford", {}):
                continue
                
            metricas = feat_data.get("benford", {}).get("metricas", {})
            for m in metricas_keys:
                if m in metricas:
                    acumuladores[f][m] += metricas[m]
            contadores[f] += 1
            
    # Calcular promedios
    promedios = {}
    for f in features_keys:
        n = contadores[f]
        if n > 0:
            promedios[f] = {m: round(acumuladores[f][m] / n, 6) for m in metricas_keys}
            promedios[f]["n_validos"] = n
        else:
            promedios[f] = {m: 0.0 for m in metricas_keys}
            promedios[f]["n_validos"] = 0
            
    return promedios


def imprimir_reporte_comparativo(promedios_real: Dict[str, Any], promedios_ia: Dict[str, Any]):
    """Imprime una tabla comparativa de los promedios de métricas en consola."""
    print("\n" + "="*80)
    print("                REPORTE COMPARATIVO DE MÉTRICAS LEY DE BENFORD")
    print("           (Distancia/Divergencia menor = Más cercano a la Ley de Benford)")
    print("="*80)
    
    features_keys = ["rms", "pitch", "zcr", "silencios", "jitter", "shimmer"]
    metricas_keys = ["mad", "chi2", "kl", "js"]
    
    # Encabezado
    print(f"{'Feature / Métrica':<20} | {'REAL (Humano)':<27} | {'IA (Sintético)':<27}")
    print("-"*80)
    
    for f in features_keys:
        p_real = promedios_real.get(f, {})
        p_ia = promedios_ia.get(f, {})
        
        print(f"--- {f.upper()} (n_real={p_real.get('n_validos', 0)}, n_ia={p_ia.get('n_validos', 0)}) ---")
        
        for m in metricas_keys:
            val_real = p_real.get(m, 0.0)
            val_ia = p_ia.get(m, 0.0)
            
            # Destacar la diferencia
            dif = val_ia - val_real
            signo = "+" if dif >= 0 else ""
            
            print(f"  {m.upper():<16} | {val_real:<27.6f} | {val_ia:<27.6f} [Dif: {signo}{dif:.6f}]")
        print("-"*80)


def main():
    print("Iniciando Evaluación Batch de Dataset...")
    inicializar_directorios()
    
    audios_real = obtener_audios(REAL_DIR)
    audios_ia = obtener_audios(IA_DIR)
    
    if not audios_real and not audios_ia:
        print("\n[⚠️ ADVERTENCIA] No se encontraron archivos de audio en data/real/ ni en data/ia/")
        print("Asegúrate de colocar archivos con extensiones permitidas (.wav, .mp3, etc.) en esas carpetas.")
        return
        
    print(f"Audios Reales encontrados: {len(audios_real)}")
    print(f"Audios de IA encontrados: {len(audios_ia)}")
    
    # Procesar
    res_real = analizar_grupo(audios_real, RESULTS_REAL_DIR, "real")
    res_ia = analizar_grupo(audios_ia, RESULTS_IA_DIR, "ia")
    
    # Resúmenes promedio
    prom_real = calcular_promedios_metricas(res_real)
    prom_ia = calcular_promedios_metricas(res_ia)
    
    # Guardar reporte resumido global
    reporte_global = {
        "resumen": {
            "total_real_analizados": len(res_real),
            "total_ia_analizados": len(res_ia),
        },
        "promedios_real": prom_real,
        "promedios_ia": prom_ia
    }
    
    ruta_resumen = RESULTS_DIR / "reporte_global.json"
    ruta_resumen.write_text(json.dumps(reporte_global, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Mostrar resultados en pantalla
    imprimir_reporte_comparativo(prom_real, prom_ia)
    print(f"\nEvaluación completa. Reporte global guardado en: {ruta_resumen.name}")


if __name__ == "__main__":
    main()
