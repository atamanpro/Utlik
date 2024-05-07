from typing import Annotated, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel
import logging
import os


router = APIRouter(
    prefix="/tools",
    tags=["Tool functions endpoints"],
)


@router.get("/search/google")
async def search_google():
    return {"ok": True, 'status': 200, 'data': 'get data from google'}


@router.get("/search/yandex")
async def search_yandex():
    return {"ok": True, 'status': 200, 'data': 'get data from yandex'}
