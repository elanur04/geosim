from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import upload, predict, simulate
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api/v1/upload", tags=["upload"])
app.include_router(predict.router, prefix="/api/v1/predict", tags=["predict"])
app.include_router(simulate.router, prefix="/api/v1/simulate", tags=["simulate"])

@app.get("/")
def read_root():
    return {"message": "Welcome to GeoSim API"}
