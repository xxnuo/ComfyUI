import base64

from webui.model import ErrorCode, ErrorDetail, ErrorMessage


def encode_data(data_path):
    """编码视频文件为base64"""
    try:
        with open(data_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
    except Exception as e:
        # 抛出一个包含ErrorDetail的异常
        error_detail = ErrorDetail(
            code=ErrorCode.ENCODE_VIDEO_FAILED,
            message=ErrorMessage.ENCODE_VIDEO_FAILED
        )
        raise Exception(f"{error_detail.message}: {e}")
