"""
main.py
-------
Responsabilidad única: inicializar la aplicación FastAPI y registrar routers.

No debe contener lógica de negocio. Ejecutar con:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI

from app.routes import router

app = FastAPI(
    title="Voice AI Detector",
    description="Detección de voz generada por IA en llamadas telefónicas "
                 "mediante características derivadas y Ley de Benford.",
    version="0.1.0",
)

app.include_router(router)
