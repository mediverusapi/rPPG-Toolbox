from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import TORCH_AVAILABLE, POS_AVAILABLE
from .routes.predict import router as predict_router
from .routes.preview import router as preview_router


def create_app() -> FastAPI:
    app = FastAPI(title="rPPG POS Demo (Modular)")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://circadify.com",
            "http://localhost:3000",
            "http://localhost:8000",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def health():
        from .services.bp_predictor import BP_PREDICTOR_AVAILABLE

        return {
            "status": "ok",
            "services": {
                "torch_available": TORCH_AVAILABLE,
                "pos_available": POS_AVAILABLE,
                "bp_predictor_available": BP_PREDICTOR_AVAILABLE,
            },
        }

    app.include_router(predict_router)
    app.include_router(preview_router)

    return app


# Provide module-level app for uvicorn app:app
app = create_app()


