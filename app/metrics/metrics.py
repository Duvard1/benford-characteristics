"""
metrics.py
----------
Responsabilidad única: calcular métricas de distancia/divergencia entre
dos distribuciones de probabilidad (observada vs teórica).

Agnóstico de Benford: solo recibe dos arrays que suman ~1.0 y devuelve
números. app/benford/benford.py es quien construye esas distribuciones.
"""

import numpy as np

# Epsilon para evitar log(0) / división por cero en KL y JS
_EPS = 1e-12


def mad(observado: np.ndarray, esperado: np.ndarray) -> float:
    """
    Mean Absolute Deviation: promedio de las diferencias absolutas
    entre la distribución observada y la esperada (Benford teórica).

    Valores más bajos = más parecido a Benford.
    """
    observado = np.asarray(observado, dtype=np.float64)
    esperado = np.asarray(esperado, dtype=np.float64)
    return float(np.mean(np.abs(observado - esperado)))


def chi_cuadrado(observado: np.ndarray, esperado: np.ndarray, n: int) -> float:
    """
    Estadístico χ² (chi-cuadrado) de bondad de ajuste.

    Args:
        observado: distribución observada (proporciones, suman 1.0)
        esperado: distribución teórica de Benford (proporciones, suman 1.0)
        n: número de observaciones usadas para construir 'observado'
           (necesario porque χ² trabaja con conteos, no proporciones)

    Returns:
        Estadístico χ². Valores más altos = mayor desviación de Benford.
    """
    observado = np.asarray(observado, dtype=np.float64)
    esperado = np.asarray(esperado, dtype=np.float64)

    conteos_observados = observado * n
    conteos_esperados = esperado * n

    # Evitar división por cero (no debería pasar con la distribución teórica
    # de Benford ya que ningún dígito tiene probabilidad 0)
    conteos_esperados = np.where(conteos_esperados == 0, _EPS, conteos_esperados)

    chi2 = np.sum((conteos_observados - conteos_esperados) ** 2 / conteos_esperados)
    return float(chi2)


def kl_divergencia(observado: np.ndarray, esperado: np.ndarray) -> float:
    """
    Divergencia de Kullback-Leibler: D_KL(observado || esperado).

    Mide cuánta "información se pierde" al aproximar la distribución
    observada con la distribución teórica de Benford.
    No es simétrica. Valores más bajos = más parecido a Benford.
    """
    p = np.asarray(observado, dtype=np.float64) + _EPS
    q = np.asarray(esperado, dtype=np.float64) + _EPS

    # Renormalizar tras sumar epsilon, para que sigan sumando 1.0
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(p * np.log(p / q)))


def js_divergencia(observado: np.ndarray, esperado: np.ndarray) -> float:
    """
    Divergencia de Jensen-Shannon: versión simétrica y acotada [0, ln(2)]
    de KL, usando una distribución promedio M = (P+Q)/2.

    Más robusta que KL para comparar distribuciones que pueden tener
    ceros. Valores más bajos = más parecido a Benford.
    """
    p = np.asarray(observado, dtype=np.float64) + _EPS
    q = np.asarray(esperado, dtype=np.float64) + _EPS
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    return float(0.5 * kl_pm + 0.5 * kl_qm)


def calcular_todas_las_metricas(observado: np.ndarray, esperado: np.ndarray, n: int) -> dict:
    """
    Punto de entrada del módulo: calcula las 4 métricas de una vez.

    Returns:
        {"mad": ..., "chi2": ..., "kl": ..., "js": ...}
    """
    return {
        "mad": round(mad(observado, esperado), 6),
        "chi2": round(chi_cuadrado(observado, esperado, n), 6),
        "kl": round(kl_divergencia(observado, esperado), 6),
        "js": round(js_divergencia(observado, esperado), 6),
    }
