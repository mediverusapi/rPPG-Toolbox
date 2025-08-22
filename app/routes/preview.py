from __future__ import annotations

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import ENH_OUT_DIR


router = APIRouter()


@router.get("/preview/{filename}")
async def get_preview(filename: str):
    file_path = os.path.join(ENH_OUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)


