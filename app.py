from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from router.autoencoder_inference import get_recommendations
app = FastAPI()

# dummy dto
class RecRequest(BaseModel):
    num: int 

# dto
class RecResponse(BaseModel):
    items: List[str]

# health check
@app.get("/")
def read_root():
    return {"message": "Model inference server is running"}

@app.get("/predict", response_model=RecResponse)
def predict(request):
    response = get_recommendations(request)
    return response

# 서버 실행 (개발 환경에서만 사용, 프로덕션에서는 다른 방식 권장)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)