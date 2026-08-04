from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class PredictRequest(BaseModel):
    image_path: str

@router.post("/")
async def predict_geojson(request: PredictRequest):
    # Segmentasyon ve vektörizasyon işlemleri burada olacak
    return {"message": "GeoJSON tahmini başarılı.", "image": request.image_path}
