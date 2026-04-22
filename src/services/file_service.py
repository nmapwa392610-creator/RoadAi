import tempfile
import os


def run_with_temp_file(file_bytes, ext, pipeline_func):
    """
    Universal temp file handler
    """

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result = pipeline_func(tmp_path)
        return result

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)