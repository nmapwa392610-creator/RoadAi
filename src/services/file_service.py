import tempfile
import os


def run_with_temp_file(file_bytes, ext, pipeline_func):
    # Создаём временный файл, запускаем pipeline, удаляем файл после обработки
    # delete=False — нужно чтобы файл не удалился до того как pipeline его прочитает
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)

        # Принудительно сбрасываем буфер Python
        tmp.flush()

        # Гарантируем запись файла на диск
        os.fsync(tmp.fileno())

        tmp_path = tmp.name

    try:
        result = pipeline_func(tmp_path)
        return result

    finally:
        # Удаляем файл в любом случае — даже если pipeline упал с ошибкой
        if os.path.exists(tmp_path):
            os.remove(tmp_path)