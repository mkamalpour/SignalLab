"""Same-directory staged writes keep an existing export intact on failure."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


@contextmanager
def atomic_target(path):
    target = Path(path)
    descriptor, name = tempfile.mkstemp(prefix="." + target.stem + "-", suffix=target.suffix, dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
