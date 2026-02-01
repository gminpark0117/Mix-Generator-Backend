from fastapi import FastAPI
from atomix.api.v1 import router as v1_router

app = FastAPI(title="atomix API", version="0.1.0")
app.include_router(v1_router, prefix="/v1")
