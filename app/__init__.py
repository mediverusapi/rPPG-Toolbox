from .main import create_app

# Expose module-level `app` so `uvicorn app:app` works
app = create_app()

__all__ = ["app", "create_app"]


