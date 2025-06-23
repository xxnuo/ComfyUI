import os

# set CUDA_MODULE_LOADING=LAZY to speed up the serverless function
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# set SAFETENSORS_FAST_GPU=1 to speed up the serverless function
os.environ["SAFETENSORS_FAST_GPU"] = "1"
import uuid
import time
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import uvicorn
import torch

from webui.engine import WanVideo, check_data_format

# 任务队列和处理器
task_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动事件
    global task_processor
    task_processor = asyncio.create_task(task_processor_func())
    yield
    # 关闭事件
    if task_processor:
        task_processor.cancel()
        try:
            await task_processor
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Video Generation API",
    description="API for video generation from text prompts",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型管理
model_instance = None
model_status = "unloaded"  # unloaded, loading, loaded, error
model_error = None


# 任务队列和结果存储
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoRequest(BaseModel):
    prompt: str
    steps: Optional[int] = 6
    num_frames: Optional[int] = 65  # 4秒 * 16fps + 1
    width: Optional[int] = 832
    height: Optional[int] = 480
    n_prompt: Optional[str] = (
        "Bright tones, overexposed, static, blurred details, subtitles, static, cg, cartoon,overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards."
    )
    cfg: Optional[float] = 1.0
    shift: Optional[float] = 4.0
    seed: Optional[int] = None


class ModelConfig(BaseModel):
    lora_name: Optional[str] = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"
    transformer_name: Optional[str] = "Wan2_1-T2V-14B_fp8_e5m2.safetensors"
    t5_model_name: Optional[str] = "umt5-xxl-enc-bf16.safetensors"
    vae_name: Optional[str] = "Wan2_1_VAE_bf16.safetensors"
    strength: Optional[float] = 0.5


class Task(BaseModel):
    id: str
    status: TaskStatus
    request: VideoRequest
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


tasks: Dict[str, Task] = {}
processing_lock = asyncio.Lock()
task_queue: List[str] = []


def encode_data(data_path):
    with open(data_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


async def process_task(task_id: str):
    """后台处理视频生成任务"""
    global model_instance

    task = tasks[task_id]
    if not task:
        return

    try:
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tasks[task_id] = task

        if model_instance is None:
            raise ValueError("Model is not loaded. Please load the model first.")

        job_input = task.request.dict()
        job_input = check_data_format(job_input)

        # 如果没有指定seed，生成一个随机种子
        if job_input["seed"] is None:
            job_input["seed"] = torch.randint(0, 1000000000, (1,)).item()

        save_path = model_instance.inference(
            prompt=job_input["prompt"],
            steps=job_input["steps"],
            num_frames=job_input["num_frames"],
            width=job_input["width"],
            height=job_input["height"],
            n_prompt=job_input["n_prompt"],
            cfg=job_input["cfg"],
            shift=job_input["shift"],
            seed=job_input["seed"],
        )

        video_data = encode_data(save_path)
        task.result = {
            "filename": os.path.basename(save_path),
            "data": video_data,
            "seed": job_input["seed"],
        }
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error = str(e)
        task.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    finally:
        tasks[task_id] = task


async def task_processor_func():
    """队列处理器，持续监控并处理队列中的任务"""
    while True:
        await asyncio.sleep(1)  # 检查频率

        async with processing_lock:
            if not task_queue:
                continue

            # 检查模型状态
            if model_status != "loaded":
                continue

            # 从队列头取出一个任务
            task_id = task_queue.pop(0)

        # 处理任务
        await process_task(task_id)


@app.get("/health")
def health_check():
    """健康检查接口"""
    return {
        "status": "ok",
        "model_status": model_status,
        "error": model_error,
        "queue_size": len(task_queue),
        "tasks_count": len(tasks),
    }


@app.get("/model/status")
async def get_model_status():
    """获取模型状态"""
    return {
        "status": model_status,
        "error": model_error,
        "loaded": model_status == "loaded",
    }


@app.post("/model/load")
async def load_model(config: ModelConfig = None):
    """加载模型"""
    global model_instance, model_status, model_error

    if model_status == "loading":
        raise HTTPException(status_code=409, detail="Model is already being loaded")

    if model_status == "loaded":
        raise HTTPException(status_code=409, detail="Model is already loaded")

    try:
        model_status = "loading"
        model_error = None

        if config is None:
            config = ModelConfig()

        model_instance = WanVideo(
            lora_name=config.lora_name,
            transformer_name=config.transformer_name,
            t5_model_name=config.t5_model_name,
            vae_name=config.vae_name,
            strength=config.strength,
        )

        model_status = "loaded"
        return {"status": "success", "message": "Model loaded successfully"}
    except Exception as e:
        model_status = "error"
        model_error = str(e)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/model/unload")
async def unload_model():
    """卸载模型"""
    global model_instance, model_status, model_error

    if model_status == "unloaded":
        return {"status": "success", "message": "Model is already unloaded"}

    try:
        if model_instance:
            # 释放模型占用的资源
            model_instance = None
            torch.cuda.empty_cache()

        model_status = "unloaded"
        model_error = None
        return {"status": "success", "message": "Model unloaded successfully"}
    except Exception as e:
        model_error = str(e)
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")


@app.post("/tasks", status_code=status.HTTP_202_ACCEPTED)
async def create_task(request: VideoRequest):
    """创建文生视频任务"""
    # 生成任务ID
    task_id = str(uuid.uuid4())

    # 创建任务
    task = Task(
        id=task_id,
        status=TaskStatus.PENDING,
        request=request,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # 存储任务
    tasks[task_id] = task

    # 添加到队列
    async with processing_lock:
        task_queue.append(task_id)

    return {"task_id": task_id, "status": "accepted", "queue_position": len(task_queue)}


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """获取任务状态和结果"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]

    # 如果任务已完成，只返回必要信息，不返回大型base64数据
    if task.status == TaskStatus.COMPLETED and task.result:
        response_task = task.dict()
        if "data" in response_task["result"]:
            response_task["result"]["data_available"] = True
            del response_task["result"]["data"]
        return response_task

    return task


@app.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str):
    """专门获取任务结果数据"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=404,
            detail=f"Task result not available, current status: {task.status}",
        )

    if not task.result:
        raise HTTPException(status_code=404, detail="Task result is empty")

    return task.result


@app.get("/tasks")
async def list_tasks(limit: int = 10, skip: int = 0):
    """列出所有任务"""
    task_list = list(tasks.values())
    return {
        "total": len(task_list),
        "tasks": [
            {
                "id": t.id,
                "status": t.status,
                "created_at": t.created_at,
                "completed_at": t.completed_at,
            }
            for t in task_list[skip : skip + limit]
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
