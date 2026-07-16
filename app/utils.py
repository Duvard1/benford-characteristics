# utilidades reeutilisables 
import uuid
from pathlib import Path


def generar_nombre_unico(extension: str) -> str:
    """
    Genera un nombre de archivo único usando UUID4.

    Args:
        extension: extensión deseada, con o sin punto (ej. "wav" o ".wav")

    Returns:
        Nombre de archivo único, ej: "3f2a1b9c.wav"
    """
    ext = extension if extension.startswith(".") else f".{extension}"
    return f"{uuid.uuid4().hex}{ext}"


def eliminar_archivo_seguro(path: Path) -> None:
    """
    Elimina un archivo si existe, sin lanzar excepción si falla.
    """
    try:
        if path.exists():
            path.unlink()
    except OSError:
        # No es crítico si falla la limpieza; se podría loggear en el futuro.
        pass
