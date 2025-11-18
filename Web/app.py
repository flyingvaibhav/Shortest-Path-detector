"""ASGI entrypoint to run with `uvicorn app:app`."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from app import app  # noqa: E402  pylint: disable=wrong-import-position

