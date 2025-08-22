from enum import Enum, StrEnum, IntEnum
from pydantic import BaseModel
from typing import Optional, Dict, Any


class TaskStatus(StrEnum):
    # 任务状态: 待处理、处理中、已完成、失败
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoRequest(BaseModel):
    # 视频请求
    prompt: str
    seconds: Optional[float] = 4.0  # 默认4秒视频
    width: Optional[int] = 832
    height: Optional[int] = 480


class ErrorCode(IntEnum):
    SUCCESS = 0
    MODEL_NOT_LOADED = 1000
    MODEL_ALREADY_LOADED = 1001
    MODEL_LOADING = 1002
    MODEL_ERROR = 1003
    MODEL_MEMORY_NOT_ENOUGH = 1004
    MODEL_JTOP_ERROR = 1005

    INFER_FAILED = 2000

    TASK_SUCCESS = 3000
    TASK_NOT_FOUND = 3001
    TASK_FAILED = 3002
    TASK_NOT_COMPLETED = 3003
    TASK_RESULT_NOT_AVAILABLE = 3004
    TASK_INVALID_STATUS = 3005
    TASK_DELETE_FAILED = 3006

    ENCODE_VIDEO_FAILED = 4000


class ErrorMessage(str, Enum):
    OK = "ok"
    MODEL_NOT_LOADED = "Model is not loaded. Please load the model first"
    MODEL_ALREADY_LOADED = "Model is already loaded"
    MODEL_LOADING = "Model is already being loaded"
    MODEL_ERROR = "Model error (perhaps unload failed)"
    MODEL_JTOP_ERROR = "Failed to get system memory info"
    MODEL_MEMORY_NOT_ENOUGH = "Text to video required memory: %sGB, Available system memory: %sGB/%sGB, please shutdown other applications and try again."

    INFER_FAILED = "Inference failed: %s"

    TASK_NOT_FOUND = "Task %s not found"
    TASK_FAILED = "Task %s failed"
    TASK_NOT_COMPLETED = "Task %s is not completed"
    TASK_RESULT_NOT_AVAILABLE = "Task %s result is empty"
    TASK_INVALID_STATUS = "Invalid status: %s"
    TASK_DELETE_FAILED = "Failed to delete task %s"

    ENCODE_VIDEO_FAILED = "Encode video failed"


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: ErrorMessage


class ModelStatus(StrEnum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class ModelType(StrEnum):
    SMALL = "small"
    LARGE = "large"


class ModelStatusResponse(BaseModel):
    status: ModelStatus
    tasks_count: int
    loaded: bool
    error: Optional[ErrorDetail] = None


class LoadModelResponse(BaseModel):
    status: ModelStatus
    message: str


class UnloadModelResponse(BaseModel):
    status: ModelStatus
    message: str


class TaskInfo(BaseModel):
    # 任务
    id: str
    status: TaskStatus
    prompt: str
    seconds: float
    width: int
    height: int
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[ErrorDetail] = None


class DeleteTaskResponse(BaseModel):
    status: TaskStatus
    message: str
