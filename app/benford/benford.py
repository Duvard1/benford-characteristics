"""
benford.py
----------
Responsabilidad única: extraer el primer dígito significativo de un vector
numérico y construir su distribución observada (proporciones para dígitos 1-9),
junto con la distribución teórica de Benford para comparar.

Este módulo es agnóstico de qué característica se le pase (pitch, jitter,
RMS, etc). Solo recibe un vector de números y devuelve distribuciones.

No calcula MAD/χ²/KL/JS — eso vive en app/metrics/metrics.py.
"""

import numpy as np

# Distribución teórica de Benford para el primer dígito significativo (1-9):
# P(d) = log10(1 + 1/d)
DIGITOS = np.arange(1, 10)
DISTRIBUCION_TEORICA = np.log10(1 + 1 / DIGITOS)


def primer_digito_significativo(valor: float) -> int:
    """
    Extrae el primer dígito significativo (1-9) de un número.

    Ignora el signo y los ceros a la izquierda / posición decimal:
    123.45 -> 1
    0.0042 -> 4
    -87.1  -> 8

    Returns:
        Entero entre 1 y 9.

    Raises:
        ValueError si el valor es 0, NaN o infinito (no tiene dígito
        significativo válido para Benford).
    """
    if valor == 0 or np.isnan(valor) or np.isinf(valor):
        raise ValueError(f"Valor no válido para Benford: {valor}")

    valor = abs(valor)

    while valor < 1:
        valor *= 10
    while valor >= 10:
        valor /= 10

    return int(valor)


def construir_distribucion_observada(valores) -> tuple[np.ndarray, int]:
    """
    Construye la distribución observada de primeros dígitos (1-9) a partir
    de un vector de números (ej. valores de pitch, RMS, jitter...).

    Valores inválidos para Benford (0, NaN, inf) se descartan silenciosamente
    antes de construir la distribución.

    Args:
        valores: iterable de números (float o int)

    Returns:
        (distribucion_observada, n_valores_validos)
        distribucion_observada: array de 9 proporciones (suman 1.0),
                                  una por cada dígito 1-9, en orden.
        n_valores_validos: cantidad de valores efectivamente usados.

    Raises:
        ValueError si no queda ningún valor válido tras filtrar.
    """
    conteos = np.zeros(9, dtype=np.int64)
    n_validos = 0

    for v in valores:
        try:
            digito = primer_digito_significativo(v)
        except ValueError:
            continue
        conteos[digito - 1] += 1
        n_validos += 1

    if n_validos == 0:
        raise ValueError(
            "No hay valores válidos para construir la distribución de Benford "
            "(vector vacío o todos los valores son 0/NaN/inf)."
        )

    distribucion_observada = conteos / n_validos
    return distribucion_observada, n_validos


def analizar_benford(valores) -> dict:
    """
    Punto de entrada del módulo: dado un vector de una característica,
    devuelve su distribución observada, la teórica, y los conteos crudos.

    Este dict es el insumo directo para app/metrics/metrics.py.
    """
    distribucion_observada, n_validos = construir_distribucion_observada(valores)

    return {
        "n_valores": n_validos,
        "digitos": DIGITOS.tolist(),
        "distribucion_observada": distribucion_observada.tolist(),
        "distribucion_teorica": DISTRIBUCION_TEORICA.tolist(),
    }
