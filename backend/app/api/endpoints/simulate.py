from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter()

class SimulateRequest(BaseModel):
    action: str
    coordinates: Dict[str, Any]

@router.post("/")
async def simulate_interactions(request: SimulateRequest):
    # Bina silme, yol ekleme gibi harita etkileşimleri
    return {"message": f"Etkileşim simülasyonu başarılı. Aksiyon: {request.action}"}
