from typing import Annotated, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel
import logging
import os


router = APIRouter(
    prefix="/integrations",
    tags=["Integrations endpoints"],
)


class DataModel(BaseModel):
    key1: str
    key2: str
    key3: int


@router.post("/bitrix")
async def post_nothing(data: dict):
    print(data)
    return {"ok": True, "message": data}


@router.post("/bitrix2")
async def post_nothing(data: DataModel):
    print(data)
    return {"ok": True, "message": data}


@router.get("/bitrix")
async def get_nothing():
    return {"ok": True, 'status': 'get data from bitrix'}
