"""Atomic immutable JSON artifacts and a PID-checked single workflow lock."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from .k1_protocol import canonical_json


def freeze_json(path,payload):
    path=Path(path);encoded=canonical_json(payload);path.parent.mkdir(parents=True,exist_ok=True)
    def same():
        if canonical_json(json.loads(path.read_text(encoding='utf-8')))!=encoded:raise ValueError('frozen artifact differs: '+str(path))
    if path.exists():same();return
    temporary=path.parent/f'.atomic-{os.getpid()}-{time.time_ns()}.tmp'
    try:
        with temporary.open('xb') as stream:stream.write(encoded);stream.flush();os.fsync(stream.fileno())
        try:os.link(temporary,path)
        except FileExistsError:same()
        except OSError:
            if os.name!='nt':raise
            try:os.rename(temporary,path)
            except FileExistsError:same()
    finally:
        if temporary.exists():temporary.unlink()


@contextmanager
def workflow_lock(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);path=directory/'RUNNING.json'
    # Portable runtime is Windows-only. Query only the prior recorded PID; never
    # kill a foreign process or guess from a stale plain-text lock.
    if path.exists():
        prior=json.loads(path.read_text(encoding='utf-8'))
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,prior['pid'])
        if handle:kernel.CloseHandle(handle);raise RuntimeError('workflow process is still running; do not launch a second copy')
        prior_path=directory/'locks'/f'stale-{time.time_ns()}.json';prior_path.parent.mkdir(exist_ok=True)
        path.rename(prior_path)
    with path.open('x',encoding='utf-8') as stream:json.dump(dict(pid=os.getpid(),started=time.time()),stream)
    try:yield
    finally:
        if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('pid')==os.getpid():path.unlink()
