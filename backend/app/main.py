from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from app.categories import CATEGORIES
from app.config import get_settings
from app.database import Base, engine
from app.routers import analytics, chat, complaints, sales , survey_sales
from app import models  # noqa: F401 - registers SQLAlchemy models
from app.routers import survey_sales

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.auto_create_tables:
        Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="ACUD Operations & Real-Estate Analytics API",
    description="Complaint services plus database-backed real-estate sales analytics and Cohere business assistant.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(complaints.router)
app.include_router(chat.router)
app.include_router(analytics.router)
app.include_router(sales.router)
app.include_router(survey_sales.router)
app.include_router(survey_sales.router)
@app.get("/")
def root():
    return {"service": "ACUD Operations & Real-Estate Analytics API", "docs": "/docs", "sales_dashboard": "/api/v1/sales/dashboard"}


@app.get("/health")
def health():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database_status = "ok"
    except Exception:
        database_status = "unavailable"
    return {"status": "ok" if database_status == "ok" else "degraded", "database": database_status}


@app.get("/api/v1/categories")
def categories():
    return {"categories": CATEGORIES}
