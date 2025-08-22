import base64

from webui.model import ErrorMessage


def encode_data(data_path):
    """编码视频文件为base64"""
    try:
        with open(data_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
    except Exception as e:
        raise Exception(f"{ErrorMessage.ENCODE_VIDEO_FAILED}: {e}")
