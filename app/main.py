from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import routes_forward
from app.api.v1 import routes_retro
from app.api.v1 import routes_yield
from app.api.v1 import tutor_ai;
# from app.api.v1 import route_mech_predict
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(routes_forward.router, prefix=f"{settings.API_V1_STR}", tags=["Forward Prediction"])
app.include_router(routes_retro.router, prefix=f"{settings.API_V1_STR}", tags=["Retrosynthesis"])
app.include_router(routes_yield.router, prefix=f"{settings.API_V1_STR}", tags=["Yield Prediction"])
app.include_router(tutor_ai.router , prefix=f"{settings.API_V1_STR}" , tags=["AI tutor"]);
# app.include_router(route_mech_predict , prefix=f"{settings.API_V1_STR}", tags=["Mechanism predection"])
@app.get("/")
async def read_root():
    return {"message": "AI Virtual Chemist API is running"}
