from contextlib import contextmanager
from threading import RLock

from filelock import FileLock


LOCK_FILE = "data/chroma_storage.lock"
_PROCESS_LOCK = RLock()


@contextmanager
def storage_access():
    with _PROCESS_LOCK:
        with FileLock(LOCK_FILE):
            yield
