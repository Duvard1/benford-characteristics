# Inicializar la aplicacion
from fastapi import FastAPI
from app.routes import router

app = FastAPI(
    title="Benford Características Acústicas",
    description="Detección de voz generada por IA en llamadas telefónicas "
                 "mediante características acústicas derivadas y Ley de Benford.",
    version="0.1.0",
)

app.include_router(router)