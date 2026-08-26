import os

# Windows: torch and opencv/numpy each ship their own OpenMP runtime, which
# crashes the process (segfault) when both get loaded. Must be set before
# torch is imported anywhere in the package.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
