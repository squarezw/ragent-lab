"""Run with uvicorn ragent_lab.api.app:app --app-dir src."""
import logging
import os
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from . import vision
from .catalog import EXPERIMENTS
from .schemas import ChunkRequest, EmbeddingRequest
from .services import chunk_text, embed_texts, model_catalog, strategy_catalog

load_dotenv()
# App logs (e.g. one line per vision model call) go to stderr next to uvicorn's.
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('%(levelname)s:     %(name)s %(message)s'))
_app_log = logging.getLogger('ragent_lab')
_app_log.setLevel(os.getenv('LAB_LOG_LEVEL', 'INFO').upper())
_app_log.addHandler(_handler)
_app_log.propagate = False
app = FastAPI(title='Ragent Lab API', version='2.0.0')
app.include_router(vision.router)


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError):
    # Pydantic's default error body may echo input containing credentials.
    return JSONResponse(status_code=422, content={'detail': '输入不符合要求，请检查文本长度和参数。',
                        'fields': ['.'.join(map(str, e['loc'][1:])) for e in exc.errors()]})


@app.get('/api/health')
def health():
    return {'status': 'ok', 'version': '2.0.0', 'revision': os.getenv('APP_REVISION', 'development')}


@app.get('/api/experiments')
def experiments():
    return EXPERIMENTS


@app.get('/api/chunking/strategies')
def strategies():
    return strategy_catalog()


@app.get('/api/embedding/models')
def models():
    return model_catalog()


@app.post('/api/chunking/run')
def run_chunking(request: ChunkRequest):
    return chunk_text(request)


@app.post('/api/embedding/run')
def run_embedding(request: EmbeddingRequest):
    return embed_texts(request)
