import logging
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from jtop import jtop  # type: ignore

from webui.engine import WanVideo
from webui.model import (
    DeleteTaskResponse,
    ErrorCode,
    ErrorDetail,
    ErrorMessage,
    LoadModelResponse,
    ModelStatus,
    ModelStatusResponse,
    ModelType,
    TaskInfo,
    TaskStatus,
    UnloadModelResponse,
    VideoRequest,
)

# from webui.utils import encode_data

load_dotenv()

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
class ModelManager:
    instance = None
    status = ModelStatus.UNLOADED
    error: Optional[ErrorDetail] = None
    lock = threading.Lock()  # 添加模型操作锁
    type = os.getenv("MODEL_TYPE", ModelType.SMALL)


model = ModelManager()

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


tasks: Dict[str, TaskInfo] = {}


def process_task(task_id: str) -> TaskInfo:
    """同步处理视频生成任务"""
    global model

    task = tasks.get(task_id)
    if not task:
        logger.error(f"Task {task_id} not found")
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                code=ErrorCode.TASK_NOT_FOUND,
                message=ErrorMessage.TASK_NOT_FOUND % task_id,
            ),
        )

    try:
        task.status = TaskStatus.PROCESSING
        tasks[task_id] = task
        logger.info(f"Processing task {task_id}...")

        with model.lock:
            if model.instance is None or model.status != ModelStatus.LOADED:
                raise HTTPException(
                    status_code=500,
                    detail=ErrorDetail(
                        code=ErrorCode.MODEL_NOT_LOADED,
                        message=ErrorMessage.MODEL_NOT_LOADED,
                    ),
                )

            # 计算帧数 (duration * fps + 1)
            fps = 16
            num_frames = int(task.seconds * fps) + 1

            seed = torch.randint(0, 1000000000, (1,)).item()
            logger.info(f"Generating video with seed: {seed}, frames: {num_frames}")

            # 调用推理函数生成视频
            try:
                save_path = model.instance.inference(
                    prompt=task.prompt,
                    num_frames=num_frames,
                    width=task.width,
                    height=task.height,
                    seed=seed,
                )
                logger.info(f"Video generated successfully: {save_path}")
            except Exception:
                logger.error(f"{ErrorMessage.INFER_FAILED % task_id}")
                raise HTTPException(
                    status_code=500,
                    detail=ErrorDetail(
                        code=ErrorCode.INFER_FAILED,
                        message=ErrorMessage.INFER_FAILED % task_id,
                    ),
                )

        # 编码视频数据
        # video_data = encode_data(save_path)
        # if not video_data:
        #     raise ValueError("Failed to encode video data")

        task.result = {
            "filename": os.path.basename(save_path),
            "path": save_path,
            # "data": video_data,
            "seed": seed,
        }
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        task.status = TaskStatus.FAILED
        task.error = ErrorDetail(
            code=ErrorCode.TASK_FAILED,
            message=ErrorMessage.TASK_FAILED % task_id,
        )
        task.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    finally:
        tasks[task_id] = task
        return task


@app.get("/health")
def health_check():
    """健康检查接口"""
    return {"status": "ok"}


@app.get("/model/status", response_model=ModelStatusResponse)
def get_model_status() -> ModelStatusResponse:
    """获取模型状态"""
    global model, tasks
    return ModelStatusResponse(
        status=model.status,
        tasks_count=len(tasks),
        loaded=model.status == ModelStatus.LOADED,
        error=model.error,
    )


@app.post("/model/load", response_model=LoadModelResponse)
def load_model() -> (
    LoadModelResponse
):  # small_model: bool = False):  # config: ModelConfig = None):
    """加载模型"""
    global model

    with model.lock:
        if model.status == ModelStatus.LOADING:
            raise HTTPException(
                status_code=409,
                detail=ErrorDetail(
                    code=ErrorCode.MODEL_LOADING,
                    message=ErrorMessage.MODEL_LOADING,
                ),
            )

        if model.status == ModelStatus.LOADED:
            return LoadModelResponse(
                status=model.status,
                message=ErrorMessage.MODEL_ALREADY_LOADED,
            )

        model.status = ModelStatus.LOADING
        model.error = None
        logger.info("Starting model loading...")

        # if config is None:
        #     config = ModelConfig()

        # 检查GPU可用性
        # if not torch.cuda.is_available():
        #     raise Exception("CUDA is not available, cannot load model")

        # 检查系统内存，要求至少有 21GB 或 51GB 可用

        # 获取系统可用内存
        with jtop() as jetson:
            if jetson.ok():
                tot = jetson.memory.get("RAM").get("tot")
                used = jetson.memory.get("RAM").get("used")
                available_memory_gb = (tot - used) / (1024**2)  # MB
                total_memory_gb = tot / (1024**2)  # MB
            else:
                raise HTTPException(
                    status_code=500,
                    detail=ErrorDetail(
                        code=ErrorCode.MODEL_JTOP_ERROR,
                        message=ErrorMessage.MODEL_JTOP_ERROR,
                    ),
                )

        if model.type == ModelType.SMALL:
            required_memory_gb = 21.0  # TODO: Const
        else:
            required_memory_gb = 51.0  # TODO: Const

        logger.info(
            f"Required more memory: {required_memory_gb}GB, Available system memory: {available_memory_gb:.2f}GB/{total_memory_gb:.2f}GB"
        )

        if available_memory_gb < required_memory_gb:
            raise HTTPException(
                status_code=500,
                detail=ErrorDetail(
                    code=ErrorCode.MODEL_MEMORY_NOT_ENOUGH,
                    message=ErrorMessage.MODEL_MEMORY_NOT_ENOUGH
                    % (required_memory_gb, available_memory_gb, total_memory_gb),
                ),
            )

        if model.type == ModelType.SMALL:
            model.instance = WanVideo(
                lora_name="Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
                transformer_name="Wan2_1-T2V-1_3B_fp8_e4m3fn.safetensors",
                t5_model_name="umt5-xxl-enc-fp8_e4m3fn.safetensors",
                vae_name="Wan2_1_VAE_bf16.safetensors",
            )
        else:
            model.instance = WanVideo(
                lora_name="Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
                transformer_name="Wan2_1-T2V-14B_fp8_e5m2.safetensors",
                t5_model_name="umt5-xxl-enc-bf16.safetensors",
                vae_name="Wan2_1_VAE_bf16.safetensors",
            )

        model.status = ModelStatus.LOADED
        logger.info("Model loaded successfully")
        return LoadModelResponse(
            status=model.status,
            message=ErrorMessage.OK,
        )


@app.post("/model/unload", response_model=UnloadModelResponse)
def unload_model() -> UnloadModelResponse:
    """卸载模型"""
    global model

    with model.lock:
        if model.status == ModelStatus.UNLOADED:
            return UnloadModelResponse(
                status=model.status,
                message=ErrorMessage.OK,
            )

        try:
            logger.info("Unloading model...")
            if model.instance:
                # 释放模型占用的资源
                model.instance = None
                torch.cuda.empty_cache()

            model.status = ModelStatus.UNLOADED
            model.error = None
            logger.info("Model unloaded successfully")
            return UnloadModelResponse(
                status=model.status,
                message=ErrorMessage.OK,
            )
        except Exception as e:
            model.error = ErrorDetail(
                code=ErrorCode.MODEL_ERROR,
                message=ErrorMessage.MODEL_ERROR,
            )
            logger.error(f"Failed to unload model: {e}")
            raise HTTPException(
                status_code=500,
                detail=ErrorDetail(
                    code=ErrorCode.MODEL_ERROR,
                    message=ErrorMessage.MODEL_ERROR,
                ),
            )


@app.post("/tasks")
def create_task(request: VideoRequest):
    """创建并执行文生视频任务（同步）"""
    global model, tasks

    # 检查模型状态
    if model.status != ModelStatus.LOADED:
        raise HTTPException(
            status_code=400,
            detail=ErrorDetail(
                code=ErrorCode.MODEL_NOT_LOADED,
                message=ErrorMessage.MODEL_NOT_LOADED,
            ),
        )

    # 生成任务ID
    task_id = str(uuid.uuid4())
    logger.info(f"Creating new task {task_id}...")

    # 创建简化的任务
    task = TaskInfo(
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
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                code=ErrorCode.TASK_FAILED,
                message=ErrorMessage.TASK_FAILED % task_id,
            ),
        )

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


@app.get("/tasks/{task_id}", response_model=TaskInfo)
def get_task(task_id: str) -> TaskInfo:
    """获取任务状态和结果"""
    global tasks
    if task_id not in tasks:
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                code=ErrorCode.TASK_NOT_FOUND,
                message=ErrorMessage.TASK_NOT_FOUND % task_id,
            ),
        )

    task = tasks[task_id]

    # 如果任务已完成，只返回必要信息，不返回大型base64数据
    if task.status == TaskStatus.COMPLETED and task.result:
        response_task = task.model_dump()
        if "data" in response_task.get("result", {}):
            response_task["result"]["data_available"] = True
            del response_task["result"]["data"]

        return TaskInfo(**response_task)

    return TaskInfo(**task.model_dump())


@app.get("/tasks/{task_id}/result")
def get_task_result(task_id: str):
    """专门获取任务结果数据"""
    global tasks
    if task_id not in tasks:
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                code=ErrorCode.TASK_NOT_FOUND,
                message=ErrorMessage.TASK_NOT_FOUND % task_id,
            ),
        )

    task = tasks[task_id]

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                code=ErrorCode.TASK_NOT_COMPLETED,
                message=ErrorMessage.TASK_NOT_COMPLETED % task_id,
            ),
        )

    if not task.result:
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                code=ErrorCode.TASK_RESULT_NOT_AVAILABLE,
                message=ErrorMessage.TASK_RESULT_NOT_AVAILABLE % task_id,
            ),
        )

    return task.result


@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks(
    status: Optional[str] = None,
) -> List[TaskInfo]:
    global tasks
    if status:
        try:
            task_status = TaskStatus(status)
            filtered_tasks = [t for t in tasks.values() if t.status == task_status]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=ErrorDetail(
                    code=ErrorCode.TASK_INVALID_STATUS,
                    message=ErrorMessage.TASK_INVALID_STATUS % status,
                ),
            )
    else:
        filtered_tasks = list(tasks.values())

    # 按创建时间倒序排序
    filtered_tasks.sort(key=lambda x: x.created_at, reverse=True)

    return [TaskInfo(**t.model_dump()) for t in filtered_tasks]


@app.delete("/tasks/{task_id}", response_model=DeleteTaskResponse)
def delete_task(task_id: str) -> DeleteTaskResponse:
    """删除任务及其视频"""
    if task_id not in tasks:
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                code=ErrorCode.TASK_NOT_FOUND,
                message=ErrorMessage.TASK_NOT_FOUND % task_id,
            ),
        )

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

    return DeleteTaskResponse(
        status=TaskStatus.COMPLETED,
        message=ErrorMessage.OK,
    )


@app.delete("/tasks/{task_id}/video", response_model=DeleteTaskResponse)
def delete_task_video(task_id: str) -> DeleteTaskResponse:
    """删除特定任务的视频文件"""
    global tasks
    if task_id not in tasks:
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                code=ErrorCode.TASK_NOT_FOUND,
                message=ErrorMessage.TASK_NOT_FOUND % task_id,
            ),
        )

    task = tasks[task_id]

    if (
        task.status != TaskStatus.COMPLETED
        or not task.result
        or "path" not in task.result
    ):
        return DeleteTaskResponse(
            status=TaskStatus.COMPLETED,
            message=ErrorMessage.OK,
        )

    video_path = task.result["path"]

    try:
        if os.path.exists(video_path):
            os.remove(video_path)
            # 从结果中移除数据
            if "data" in task.result:
                del task.result["data"]
            logger.info(f"Deleted video for task {task_id}")
            return DeleteTaskResponse(
                status=TaskStatus.COMPLETED,
                message=ErrorMessage.OK,
            )
        else:
            return DeleteTaskResponse(
                status=TaskStatus.COMPLETED,
                message=ErrorMessage.OK,
            )
    except Exception as e:
        logger.error(f"Error deleting video for task {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                code=ErrorCode.TASK_DELETE_FAILED,
                message=ErrorMessage.TASK_DELETE_FAILED % task_id,
            ),
        )


@app.delete("/tasks/cleanup", response_model=DeleteTaskResponse)
def cleanup_tasks(
    keep_uncompleted: bool = True, keep_completed: bool = False
) -> DeleteTaskResponse:
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

    return DeleteTaskResponse(
        status=TaskStatus.COMPLETED,
        message=ErrorMessage.OK,
    )


@app.get("/output/{filename}", description="获取生成的视频")
def get_video(filename: str):
    file_path = Path(VIDEO_STORAGE_DIR) / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                code=ErrorCode.TASK_NOT_FOUND,
                message=ErrorMessage.TASK_NOT_FOUND % filename,
            ),
        )
    return FileResponse(file_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
