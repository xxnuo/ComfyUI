from enum import StrEnum, IntEnum
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, Literal


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


class ErrorMessages:
    """Error messages in different languages"""

    # English messages
    EN = {
        "OK": "ok",
        "MODEL_NOT_LOADED": "Model is not loaded. Please load the model first",
        "MODEL_ALREADY_LOADED": "Model is already loaded",
        "MODEL_LOADING": "Model is already being loaded",
        "MODEL_ERROR": "Model error (perhaps unload failed)",
        "MODEL_JTOP_ERROR": "Failed to get system memory info",
        "MODEL_MEMORY_NOT_ENOUGH": "Text to video required memory: %sGB, Available system memory: %sGB/%sGB, please shutdown other services and try again.",
        "INFER_FAILED": "Inference failed: %s",
        "TASK_NOT_FOUND": "Task %s not found",
        "TASK_FAILED": "Task %s failed",
        "TASK_NOT_COMPLETED": "Task %s is not completed",
        "TASK_RESULT_NOT_AVAILABLE": "Task %s result is empty",
        "TASK_INVALID_STATUS": "Invalid status: %s",
        "TASK_DELETE_FAILED": "Failed to delete task %s",
        "ENCODE_VIDEO_FAILED": "Encode video failed",
    }

    # Chinese messages
    ZH = {
        "OK": "成功",
        "MODEL_NOT_LOADED": "模型未加载，请先加载模型",
        "MODEL_ALREADY_LOADED": "模型已经加载",
        "MODEL_LOADING": "模型正在加载中",
        "MODEL_ERROR": "模型错误（可能卸载失败）",
        "MODEL_JTOP_ERROR": "获取系统内存信息失败",
        "MODEL_MEMORY_NOT_ENOUGH": "文本生成视频需要内存：%sGB，系统可用内存：%sGB/%sGB，请在服务管理关闭其他服务后重试。",
        "INFER_FAILED": "推理失败：%s",
        "TASK_NOT_FOUND": "任务 %s 未找到",
        "TASK_FAILED": "任务 %s 失败",
        "TASK_NOT_COMPLETED": "任务 %s 未完成",
        "TASK_RESULT_NOT_AVAILABLE": "任务 %s 结果为空",
        "TASK_INVALID_STATUS": "无效的状态：%s",
        "TASK_DELETE_FAILED": "删除任务 %s 失败",
        "ENCODE_VIDEO_FAILED": "视频编码失败",
    }


class ErrorMessage(StrEnum):
    OK = "OK"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    MODEL_ALREADY_LOADED = "MODEL_ALREADY_LOADED"
    MODEL_LOADING = "MODEL_LOADING"
    MODEL_ERROR = "MODEL_ERROR"
    MODEL_JTOP_ERROR = "MODEL_JTOP_ERROR"
    MODEL_MEMORY_NOT_ENOUGH = "MODEL_MEMORY_NOT_ENOUGH"
    INFER_FAILED = "INFER_FAILED"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    TASK_FAILED = "TASK_FAILED"
    TASK_NOT_COMPLETED = "TASK_NOT_COMPLETED"
    TASK_RESULT_NOT_AVAILABLE = "TASK_RESULT_NOT_AVAILABLE"
    TASK_INVALID_STATUS = "TASK_INVALID_STATUS"
    TASK_DELETE_FAILED = "TASK_DELETE_FAILED"
    ENCODE_VIDEO_FAILED = "ENCODE_VIDEO_FAILED"


def get_error_message(
    error_code: ErrorMessage, lang: Literal["en", "zh"] = "en", *args
) -> str:
    """Get the error message in the specified language"""
    messages = ErrorMessages.EN if lang.lower() == "en" else ErrorMessages.ZH
    message_template = messages.get(error_code, messages["OK"])

    if args:
        return message_template % args
    return message_template


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str


class APIHTTPException(HTTPException):
    def __init__(
        self, status_code: int, detail: ErrorDetail, headers: dict | None = None
    ):
        # 将 Pydantic 模型转换为字典
        if isinstance(detail, BaseModel):
            serializable_detail = detail.model_dump()
        else:
            serializable_detail = detail

        super().__init__(
            status_code=status_code, detail=serializable_detail, headers=headers
        )


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
