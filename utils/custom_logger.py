import json
import logging
import os
import sys
from logging.handlers import WatchedFileHandler
from pathlib import Path

LOGS_DIR = ".logs"

TIME_FORMAT = "%H:%M:%S"
LOG_FORMAT = "%(asctime)s %(levelname)-10s %(message)s"


class _JsonFormatter(logging.Formatter):
    """Format incoming logs to a JSON object.
    The formatter adds metadata such as the origin of the log
    and datetime in addition to user-provided info. If the user
    logs a Python dictionary, its keys are added to the JSON.
    If only a string is provided, it is put under the 'message'
    key in the JSON.
    """

    def format(self, record: logging.LogRecord) -> str:
        common_format = {
            "module": record.module,
            "filename": record.filename,
            "function": record.funcName,
            "level": record.levelname,
            "date": self.formatTime(record, "%H:%M:%S"),
            "lineno": record.lineno,
            "logger_name": record.name,
        }
        try:
            # If dict was passed, add user keys to metadata
            # Dictionaries are transformed into their string
            # representation by logging library, we override it

            # Quick trick: read string representation it as JSON
            # just replace the single quotes with double
            json_form = record.getMessage().replace("'", '"')
            message = json.loads(json_form)
            return json.dumps({**common_format, **message})
        except json.JSONDecodeError:
            # Assume simple string message if dict cannot be parsed
            return json.dumps({**common_format, "message": record.getMessage()})


def get_custom_logger(
    name: str | None = None,
    log_level: int = 10,
    to_stdout: bool = False,
) -> logging.Logger:
    """Return customised logger capable of ingressing dict objects.
    Logs are saved in a JSON line format which allows integration
    with observability stacks like ELK.
    """

    logger = logging.getLogger(name=name)

    if len(logger.handlers) != 0:
        # We have instantiated the logger already
        return logger

    logger.setLevel(log_level)
    os.makedirs(logs_dir := Path(LOGS_DIR), exist_ok=True)

    json_h = WatchedFileHandler(logs_dir.joinpath(f"{name or 'root'}.ndjson"))
    json_h.setFormatter(_JsonFormatter())
    logger.addHandler(json_h)

    if to_stdout:
        stdout_h = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=TIME_FORMAT)

        stdout_h.setFormatter(formatter)
        logger.addHandler(stdout_h)
    return logger


def log_debug(msg, *args, **kwargs):
    get_custom_logger().debug(msg, *args, **kwargs)


def log_info(msg, *args, **kwargs):
    get_custom_logger().info(msg, *args, **kwargs)


def log_warning(msg, *args, **kwargs):
    get_custom_logger().warning(msg, *args, **kwargs)


def log_error(msg, *args, **kwargs):
    get_custom_logger().error(msg, *args, **kwargs)


def log_exception(msg, *args, exc_info=True, **kwargs):
    get_custom_logger().error(msg, *args, exc_info=exc_info, **kwargs)


def log_critical(msg, *args, **kwargs):
    get_custom_logger().critical(msg, args, **kwargs)
