"""Public reading surface for explicitly published FAQs only."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from routers.services.faq_service import list_faq_records

router = APIRouter()


@router.get("")
async def public_faqs():
    return JSONResponse(jsonable_encoder(await list_faq_records(public=True)), headers={"Cache-Control": "no-store"})
