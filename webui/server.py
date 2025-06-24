import base64
import logging
import os
import threading
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psutil

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

# 视频存储目录
VIDEO_STORAGE_DIR = "./output"
# 确保输出目录存在
os.makedirs(VIDEO_STORAGE_DIR, exist_ok=True)

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


# class ModelConfig(BaseModel):
#     lora_name: Optional[str]
#     transformer_name: Optional[str]
#     t5_model_name: Optional[str]
#     vae_name: Optional[str]
#     strength: Optional[float]


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
    """编码视频文件为base64"""
    try:
        with open(data_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding video data: {e}")
        return None


def process_task(task_id: str) -> Task:
    """同步处理视频生成任务"""
    global model_instance

    task = tasks.get(task_id)
    if not task:
        logger.error(f"Task {task_id} not found")
        return None

    try:
        task.status = TaskStatus.PROCESSING
        tasks[task_id] = task
        logger.info(f"Processing task {task_id}...")

        with model_lock:
            if model_instance is None:
                raise ValueError("Model is not loaded. Please load the model first.")
            if model_status != "loaded":
                raise ValueError(
                    f"Model is not in loaded state. Current state: {model_status}"
                )

            # 计算帧数 (duration * fps + 1)
            fps = 16
            num_frames = int(task.seconds * fps) + 1

            seed = torch.randint(0, 1000000000, (1,)).item()
            logger.info(f"Generating video with seed: {seed}, frames: {num_frames}")

            # 调用推理函数生成视频
            try:
                save_path = model_instance.inference(
                    prompt=task.prompt,
                    num_frames=num_frames,
                    width=task.width,
                    height=task.height,
                    seed=seed,
                )
                logger.info(f"Video generated successfully: {save_path}")
            except Exception as inference_error:
                logger.error(f"Inference failed: {inference_error}")
                raise inference_error

        # 编码视频数据
        video_data = encode_data(save_path)
        if not video_data:
            raise ValueError("Failed to encode video data")

        task.result = {
            "filename": os.path.basename(save_path),
            "path": save_path,
            "data": video_data,
            "seed": seed,
        }
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
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
def load_model():  # small_model: bool = False):  # config: ModelConfig = None):
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
            logger.info("Starting model loading...")

            # if config is None:
            #     config = ModelConfig()

            # 检查GPU可用性
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available, cannot load model")

            # 检查系统内存，要求至少有 50GB 可用

            # 获取系统可用内存
            available_memory = psutil.virtual_memory().available
            swap_memory = psutil.swap_memory().free
            total_available_memory = available_memory + swap_memory
            available_memory_gb = total_available_memory / (1024**3)
            total_memory_gb = 64.0 + swap_memory / (1024**3)
            required_memory_gb = 50.0

            logger.info(
                f"Required more memory: {required_memory_gb}GB, Available system memory: {available_memory_gb:.2f}GB/{total_memory_gb:.2f}GB"
            )

            if available_memory_gb < required_memory_gb:
                raise ValueError(
                    f"Text to video required memory: {required_memory_gb}GB, Available system memory: {available_memory_gb:.2f}GB/{total_memory_gb:.2f}GB, please shutdown other applications and try again."
                )

            # if small_model:
            #     model_instance = WanVideo(
            #         lora_name="Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
            #         transformer_name="Wan2_1-T2V-1_3B_fp8_e4m3fn.safetensors",
            #         t5_model_name="umt5-xxl-enc-fp8_e4m3fn.safetensors",
            #         vae_name="Wan2_1_VAE_bf16.safetensors",
            #     )
            # else:
            model_instance = WanVideo(
                # lora_name=config.lora_name,
                # transformer_name=config.transformer_name,
                # t5_model_name=config.t5_model_name,
                # vae_name=config.vae_name,
                # strength=config.strength,
            )

            model_status = "loaded"
            logger.info("Model loaded successfully")
            return {"status": "success", "message": "Model loaded successfully"}
        except Exception as e:
            model_status = "error"
            model_error = str(e)
            logger.error(f"Failed to load model: {e}")
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
            logger.info("Unloading model...")
            if model_instance:
                # 释放模型占用的资源
                model_instance = None
                torch.cuda.empty_cache()

            model_status = "unloaded"
            model_error = None
            logger.info("Model unloaded successfully")
            return {"status": "success", "message": "Model unloaded successfully"}
        except Exception as e:
            model_error = str(e)
            logger.error(f"Failed to unload model: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to unload model: {str(e)}"
            )


@app.post("/tasks")
def create_task(request: VideoRequest):
    """创建并执行文生视频任务（同步）"""
    global tasks

    # 检查模型状态
    if model_status != "loaded":
        raise HTTPException(
            status_code=400,
            detail=f"Model is not ready. Current status: {model_status}",
        )

    # 生成任务ID
    task_id = str(uuid.uuid4())
    logger.info(f"Creating new task {task_id}...")

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
    if not task:
        raise HTTPException(status_code=500, detail="Failed to process task")

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
def list_tasks(
    status: Optional[str] = None,
):
    if status:
        try:
            task_status = TaskStatus(status)
            filtered_tasks = [t for t in tasks.values() if t.status == task_status]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    else:
        filtered_tasks = list(tasks.values())

    # 按创建时间倒序排序
    filtered_tasks.sort(key=lambda x: x.created_at, reverse=True)

    return {
        "total": len(filtered_tasks),
        "tasks": [
            {
                "id": t.id,
                "status": t.status,
                "prompt": t.prompt,
                "created_at": t.created_at,
                "completed_at": t.completed_at,
                "error": t.error if t.status == TaskStatus.FAILED else None,
            }
            for t in filtered_tasks
        ],
    }


@app.delete("/tasks/{task_id}")
def delete_task(task_id: str):
    """删除任务及其视频"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]

    # 如果任务有视频，先删除视频
    if task.status == TaskStatus.COMPLETED and task.result and "path" in task.result:
        video_path = task.result["path"]
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Deleted video for task {task_id}")
        except Exception as e:
            logger.error(f"Error deleting video for task {task_id}: {e}")

    # 删除任务
    del tasks[task_id]
    logger.info(f"Deleted task {task_id}")

    return {"status": "success", "message": f"Task {task_id} deleted successfully"}


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
            logger.info(f"Deleted video for task {task_id}")
            return {
                "status": "success",
                "message": f"Video for task {task_id} deleted successfully",
            }
        else:
            return {"status": "skipped", "message": "Video file not found"}
    except Exception as e:
        logger.error(f"Error deleting video for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete video: {str(e)}")


@app.delete("/tasks/cleanup")
def cleanup_tasks(keep_uncompleted: bool = True, keep_completed: bool = False):
    """清理任务列表"""
    global tasks

    # 按创建时间排序所有任务
    all_tasks = list(tasks.values())
    all_tasks.sort(key=lambda x: x.created_at, reverse=True)

    # 根据参数决定是否保留所有已完成任务
    tasks_to_keep = {}
    deleted_count = 0

    for task in all_tasks:
        # 未完成任务全部保留
        if keep_uncompleted and task.status in [
            TaskStatus.PENDING,
            TaskStatus.PROCESSING,
        ]:
            tasks_to_keep[task.id] = task
        elif keep_completed and task.status == TaskStatus.COMPLETED:
            tasks_to_keep[task.id] = task
        # 其他任务需要删除
        else:
            # 如果有视频，删除视频
            if task.result and "path" in task.result:
                try:
                    video_path = task.result["path"]
                    if os.path.exists(video_path):
                        os.remove(video_path)
                        logger.info(f"Deleted video for task {task.id} during cleanup")
                except Exception as e:
                    logger.error(f"Error deleting video for task {task.id}: {e}")
            deleted_count += 1

    removed_count = len(tasks) - len(tasks_to_keep)
    tasks = tasks_to_keep
    logger.info(
        f"Cleaned up tasks: removed {removed_count} tasks, kept {len(tasks_to_keep)}"
    )

    return {
        "status": "success",
        "removed_count": removed_count,
        "remaining_count": len(tasks_to_keep),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
