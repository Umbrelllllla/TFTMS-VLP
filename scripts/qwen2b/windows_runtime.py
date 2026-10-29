"""Prefer the installed system C++ runtime for this process, before importing torch."""
import ctypes
import os
import sys
from pathlib import Path

_HANDLES = []


def prepare_runtime():
    if sys.platform != 'win32' or _HANDLES:
        return
    # Anaconda's older application-local runtime can otherwise be selected by
    # torch's name-only DLL loads. Do not replace any system or Anaconda files.
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetSystemDirectoryW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint]
    kernel.GetSystemDirectoryW.restype = ctypes.c_uint
    buffer = ctypes.create_unicode_buffer(32768)
    length = kernel.GetSystemDirectoryW(buffer, len(buffer))
    if not 0 < length < len(buffer):
        raise ctypes.WinError(ctypes.get_last_error())
    system = Path(buffer.value)
    for name in ('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll'):
        path = system / name
        _HANDLES.append(ctypes.WinDLL(os.fspath(path)))
