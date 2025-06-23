import base64
import logging
import os
import threading
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from webui.engine import WanVideo

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger()

# 设置环境变量
os.environ["CUDA_MODULE_LOADING"] = "LAZY"  # 延迟加载 CUDA 模块
os.environ["SAFETENSORS_FAST_GPU"] = "1"  # 直接加载 Safetensors 到 GPU

# 模型管理
model_instance = None
model_status = "unloaded"  # unloaded, loading, loaded, error
model_error = None
model_lock = threading.Lock()  # 添加模型操作锁

VIDEO_STORAGE_DIR = "./output"

app = FastAPI(
    title="Video Generation API",
    description="API for video generation",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 任务结果存储
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoRequest(BaseModel):
    prompt: str
    seconds: Optional[float] = 4.0  # 默认4秒视频
    width: Optional[int] = 832
    height: Optional[int] = 480
    # 其他参数使用默认值，不需要用户提供


class ModelConfig(BaseModel):
    lora_name: Optional[str]
    transformer_name: Optional[str]
    t5_model_name: Optional[str]
    vae_name: Optional[str]
    strength: Optional[float]


class Task(BaseModel):
    id: str
    status: TaskStatus
    prompt: str
    seconds: float
    width: int
    height: int
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


tasks: Dict[str, Task] = {}


def encode_data(data_path):
    with open(data_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def process_task(task_id: str) -> Task:
    """同步处理视频生成任务"""
    global model_instance

    task = tasks[task_id]
    if not task:
        return task

    try:
        task.status = TaskStatus.PROCESSING
        tasks[task_id] = task

        with model_lock:
            if model_instance is None:
                raise ValueError("Model is not loaded. Please load the model first.")

            # 计算帧数 (duration * fps + 1)
            fps = 16
            num_frames = int(task.seconds * fps) + 1

            seed = torch.randint(0, 1000000000, (1,)).item()

            save_path = model_instance.inference(
                prompt=task.prompt,
                num_frames=num_frames,
                width=task.width,
                height=task.height,
                seed=seed,
            )

        video_data = encode_data(save_path)
        task.result = {
            "filename": os.path.basename(save_path),
            "path": save_path,
            "data": video_data,
            "seed": seed,
        }
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error = str(e)
        task.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    finally:
        tasks[task_id] = task
        return task


@app.get("/health")
def health_check():
    """健康检查接口"""
    return {
        "status": "ok",
        "model_status": model_status,
        "error": model_error,
        "tasks_count": len(tasks),
    }


@app.get("/model/status")
def get_model_status():
    """获取模型状态"""
    return {
        "status": model_status,
        "error": model_error,
        "loaded": model_status == "loaded",
    }


@app.post("/model/load")
def load_model(config: ModelConfig = None):
    """加载模型"""
    global model_instance, model_status, model_error

    with model_lock:
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
            raise HTTPException(
                status_code=500, detail=f"Failed to load model: {str(e)}"
            )


@app.post("/model/unload")
def unload_model():
    """卸载模型"""
    global model_instance, model_status, model_error

    with model_lock:
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
            raise HTTPException(
                status_code=500, detail=f"Failed to unload model: {str(e)}"
            )


@app.post("/tasks")
def create_task(request: VideoRequest):
    """创建并执行文生视频任务（同步）"""
    # 生成任务ID
    task_id = str(uuid.uuid4())

    # 创建简化的任务
    task = Task(
        id=task_id,
        status=TaskStatus.PENDING,
        prompt=request.prompt,
        seconds=request.seconds,
        width=request.width,
        height=request.height,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # 存储任务
    tasks[task_id] = task

    # 同步处理任务
    task = process_task(task_id)

    # 如果任务已完成，只返回必要信息，不返回大型base64数据
    response_task = task.model_dump()
    if (
        task.status == TaskStatus.COMPLETED
        and task.result
        and "data" in response_task["result"]
    ):
        response_task["result"]["data_available"] = True
        del response_task["result"]["data"]

    return response_task


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    """获取任务状态和结果"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]

    # 如果任务已完成，只返回必要信息，不返回大型base64数据
    if task.status == TaskStatus.COMPLETED and task.result:
        response_task = task.model_dump()
        if "data" in response_task["result"]:
            response_task["result"]["data_available"] = True
            del response_task["result"]["data"]
        return response_task

    return task


@app.get("/tasks/{task_id}/result")
def get_task_result(task_id: str):
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
def list_tasks(limit: int = 10, skip: int = 0):
    """列出所有任务"""
    task_list = list(tasks.values())
    return {
        "total": len(task_list),
        "tasks": [
            {
                "id": t.id,
                "status": t.status,
                "prompt": t.prompt,
                "created_at": t.created_at,
                "completed_at": t.completed_at,
            }
            for t in task_list[skip : skip + limit]
        ],
    }


@app.delete("/tasks/{task_id}/video")
def delete_task_video(task_id: str):
    """删除特定任务的视频文件"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]

    if (
        task.status != TaskStatus.COMPLETED
        or not task.result
        or "path" not in task.result
    ):
        return {"status": "skipped", "message": "No video file available for this task"}

    video_path = task.result["path"]

    try:
        if os.path.exists(video_path):
            os.remove(video_path)
            # 从结果中移除数据
            if "data" in task.result:
                del task.result["data"]
            return {
                "status": "success",
                "message": f"Video for task {task_id} deleted successfully",
            }
        else:
            return {"status": "skipped", "message": "Video file not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete video: {str(e)}")


@app.delete("/videos/cleanup")
def cleanup_videos():
    """清理所有视频文件"""
    deleted_count = 0
    errors = []

    # 遍历所有任务，清理对应的视频
    for task_id, task in tasks.items():
        if (
            task.status == TaskStatus.COMPLETED
            and task.result
            and "path" in task.result
        ):
            video_path = task.result["path"]
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
                    # 从结果中移除数据
                    if "data" in task.result:
                        del task.result["data"]
                    deleted_count += 1
            except Exception as e:
                errors.append(f"Error deleting video for task {task_id}: {str(e)}")

    # 清理生成目录中的所有视频文件
    try:
        for filename in os.listdir(VIDEO_STORAGE_DIR):
            if filename.endswith(".mp4"):
                file_path = os.path.join(VIDEO_STORAGE_DIR, filename)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    errors.append(f"Error deleting file {filename}: {str(e)}")
    except Exception as e:
        errors.append(f"Error accessing video directory: {str(e)}")

    return {
        "status": "success" if not errors else "partial",
        "deleted_count": deleted_count,
        "errors": errors if errors else None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
