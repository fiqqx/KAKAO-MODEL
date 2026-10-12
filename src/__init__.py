"""Modul penelitian deteksi multi-objek; cache tetap di folder proyek."""

import os
import config

os.environ.setdefault("TORCH_HOME", str(config.PROJECT_DIR / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(config.PROJECT_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(config.PROJECT_DIR / ".cache"))
