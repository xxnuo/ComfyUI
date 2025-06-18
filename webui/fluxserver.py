import logging
import asyncio
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from webui.engine import FluxEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger()

default_model = "black-forest-labs/FLUX.1-schnell"
default_output_folder = "./output"

global_engine_lock = threading.RLock()
engine = FluxEngine(output_folder=default_output_folder)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 异步加载模型
    async def load_model_async():
        with global_engine_lock:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, engine.load_model, default_model)
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")

    # 启动异步任务加载模型
    asyncio.create_task(load_model_async())
    yield


# 初始化 FastAPI 应用
app = FastAPI(description="Videogen Server", lifespan=lifespan)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """添加处理时间头部的中间件"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["x-process-time"] = str(process_time)
    return response


# 添加Gzip压缩中间件
app.add_middleware(GZipMiddleware)


@app.get("/health", description="健康检查接口")
def health():
    return {"status": "ok"}


@app.get("/model/status", description="获取模型状态信息")
def model_status():
    """获取模型加载状态和最后使用时间"""
    return {
        "status": "ok",
        "model_loaded": engine.pipe is not None,
        "current_model": engine.selected_model,
        "supported_models": engine.supported_models(),
    }


@app.post("/model/list", description="获取模型列表")
def model_list():
    """获取模型列表"""
    return {"status": "success", "models": engine.supported_models()}


@app.post("/model/unload", description="手动卸载模型")
def manual_unload_model():
    """手动卸载模型接口"""
    with global_engine_lock:
        engine.unload_model()
    return {"status": "success", "message": "模型已卸载"}


@app.post("/model/load", description="手动加载模型")
def manual_load_model(model: str = "black-forest-labs/FLUX.1-schnell"):
    """手动加载模型接口"""
    try:
        with global_engine_lock:
            engine.load_model(checkpoint=model)
        return {"status": "success", "message": f"模型 {model} 已加载"}
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")


class GenerateRequest(BaseModel):
    prompt: str = "A girl in a cyberpunk city at night"
    model: Optional[str] = default_model
    steps: Optional[int] = 4
    height: Optional[int] = 512
    width: Optional[int] = 512
    images: Optional[int] = 1
    guidance_scale: Optional[float] = 3.5
    seed: Optional[int] = 0
    randomize_seed: Optional[bool] = True


class GenerateResponse(BaseModel):
    status: str
    seed: int
    image_paths: List[str]


@app.post("/generate", description="生成图片", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    try:
        with global_engine_lock:
            _, seed, saved_paths = engine.generate(
                prompt=request.prompt,
                checkpoint=request.model,
                num_images_per_prompt=request.images,
                randomize_seed=request.randomize_seed,
                width=request.width,
                height=request.height,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
                auto_unload=False,
            )

            return {
                "status": "success",
                "seed": seed,
                "image_paths": [p.lstrip(default_output_folder) for p in saved_paths],
            }
    except Exception as e:
        logger.error(f"生成图片失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成图片失败: {str(e)}")


@app.get("/output/{filename}", description="获取生成的图片")
def get_image(filename: str):
    file_path = Path(engine.output_folder) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(file_path)
