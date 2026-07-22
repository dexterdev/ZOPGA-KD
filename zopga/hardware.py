"""Hardware detection and hardware-aware run settings.

Reads the optional ``hardware:`` section of a config and turns it into
concrete runtime decisions:

- device selection (auto / cpu / cuda / cuda:N / mps)
- mixed precision (AMP autocast + GradScaler on CUDA)
- batch-size scaling from available GPU VRAM or system RAM
- synthesis teacher-query batch sizing from available memory
- DataLoader workers / pin_memory from CPU count and device
- global backend knobs: cuDNN benchmark, TF32 matmul, CPU threads
- optional torch.compile of the frozen teacher

Every ``auto`` value is resolved once at startup and logged, so a run's
effective settings are always visible and reproducible.
"""

import os

import torch

_GiB = 1024 ** 3

DEFAULTS = {
    "device": "auto",          # auto | cpu | cuda | cuda:N | mps
    "precision": "auto",       # auto | amp | fp32
    "batch_size": "auto",      # auto | int | null (null = keep per-stage values)
    "query_batch": "auto",     # auto | int | null (synthesis teacher queries)
    "num_workers": "auto",     # auto | int
    "pin_memory": "auto",      # auto | true | false
    "cpu_threads": "auto",     # auto | int
    "cudnn_benchmark": True,
    "allow_tf32": True,
    "compile_teacher": False,  # torch.compile the frozen teacher (PyTorch 2.x)
    "memory_fraction": 0.8,    # fraction of free memory budgeted for auto sizing
}


def cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def system_ram():
    """(total_bytes, available_bytes) of system RAM; psutil if present,
    /proc/meminfo or sysconf otherwise."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total, vm.available
    except ImportError:
        pass
    total = available = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
    except OSError:
        pass
    if total is None:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            total = 8 * _GiB
    if available is None:
        available = int(total * 0.5)
    return total, available


def gpu_memory(device):
    """(total_bytes, free_bytes) for a CUDA device, or (None, None)."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, None
    try:
        free, total = torch.cuda.mem_get_info(device)
        return total, free
    except (RuntimeError, AssertionError):
        props = torch.cuda.get_device_properties(device)
        return props.total_memory, int(props.total_memory * 0.9)


def resolve_device(requested):
    """Resolve a device string; 'auto' prefers cuda, then mps, then cpu."""
    if requested in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


class HardwareManager:
    """Resolves the hardware config section into concrete runtime settings."""

    def __init__(self, hw_cfg=None, cli_device=None, logger=None):
        self.cfg = dict(DEFAULTS)
        self.cfg.update(hw_cfg or {})
        self.logger = logger

        # CLI --device (anything but 'auto') beats the config value.
        requested = self.cfg["device"]
        if cli_device not in (None, "auto"):
            requested = cli_device
        self.device = resolve_device(requested)

        self.cpu_count = cpu_count()
        self.ram_total, self.ram_available = system_ram()
        self.gpu_total, self.gpu_free = gpu_memory(self.device)
        self.gpu_name = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_properties(self.device).name

        self.use_amp = self._resolve_precision()
        self.num_workers = self._resolve_num_workers()
        self.pin_memory = self._resolve_pin_memory()
        self.batch_scale = self._resolve_batch_scale()

    # -- resolution helpers -------------------------------------------------

    def _budget_bytes(self):
        """Memory budget (bytes) that auto sizing may plan for."""
        frac = float(self.cfg.get("memory_fraction", 0.8))
        if self.gpu_free is not None:
            return int(self.gpu_free * frac)
        return int(self.ram_available * frac)

    def _resolve_precision(self):
        p = str(self.cfg["precision"]).lower()
        if p not in ("auto", "amp", "fp32"):
            raise ValueError(f"hardware.precision must be auto|amp|fp32, got '{p}'")
        if self.device.type != "cuda":
            return False  # AMP is only enabled on CUDA
        if p == "amp":
            return True
        if p == "fp32":
            return False
        major, _ = torch.cuda.get_device_capability(self.device)
        return major >= 7  # tensor cores -> AMP pays off

    def _resolve_num_workers(self):
        v = self.cfg["num_workers"]
        if v != "auto":
            return int(v)
        if self.device.type == "cuda":
            return min(8, max(2, self.cpu_count // 2))
        return min(4, max(0, self.cpu_count // 2))

    def _resolve_pin_memory(self):
        v = self.cfg["pin_memory"]
        if v != "auto":
            return bool(v)
        return self.device.type == "cuda"

    def _resolve_batch_scale(self):
        """Multiplier applied to per-stage batch sizes when batch_size: auto."""
        budget = self._budget_bytes()
        if self.gpu_free is not None:
            tiers = [(20 * _GiB, 4.0), (10 * _GiB, 2.0), (4 * _GiB, 1.0)]
        else:
            tiers = [(24 * _GiB, 2.0), (8 * _GiB, 1.0)]
        for floor, scale in tiers:
            if budget >= floor:
                return scale
        return 0.5

    # -- public API ---------------------------------------------------------

    def batch_size(self, base):
        """Effective batch size for a stage whose config asks for `base`."""
        mode = self.cfg["batch_size"]
        if mode is None:
            return int(base)
        if mode != "auto":
            return int(mode)
        scaled = int(base * self.batch_scale)
        return max(16, min(2048, scaled))

    def query_batch(self, base):
        """Effective synthesis teacher-query batch size."""
        mode = self.cfg["query_batch"]
        if mode is None:
            return int(base)
        if mode != "auto":
            return int(mode)
        budget = self._budget_bytes()
        if self.gpu_free is not None:
            for floor, qb in [(20 * _GiB, 4096), (10 * _GiB, 2048),
                              (4 * _GiB, 1024), (2 * _GiB, 512)]:
                if budget >= floor:
                    return qb
            return 256
        return 512 if budget >= 8 * _GiB else 256

    def autocast(self):
        """Autocast context for the run device (no-op unless AMP is on)."""
        return torch.autocast(device_type=self.device.type,
                              enabled=self.use_amp)

    def grad_scaler(self):
        """GradScaler matching the AMP decision (works on torch>=2.0)."""
        try:
            return torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def apply_global_settings(self):
        """Set process-wide backend knobs once at startup."""
        if self.cfg["cpu_threads"] != "auto":
            torch.set_num_threads(int(self.cfg["cpu_threads"]))
        elif self.device.type == "cpu":
            torch.set_num_threads(self.cpu_count)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = bool(self.cfg["cudnn_benchmark"])
            allow_tf32 = bool(self.cfg["allow_tf32"])
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            if allow_tf32:
                torch.set_float32_matmul_precision("high")

    def maybe_compile_teacher(self, teacher):
        """torch.compile the frozen teacher when enabled and supported."""
        if not self.cfg["compile_teacher"]:
            return teacher
        if not hasattr(torch, "compile"):
            self._log("torch.compile unavailable; running teacher eagerly")
            return teacher
        try:
            return torch.compile(teacher)
        except Exception as e:  # compile support varies by platform
            self._log(f"torch.compile failed ({e}); running teacher eagerly")
            return teacher

    def summary(self):
        """Effective settings + detected hardware, for logs and results.json."""
        s = {
            "device": str(self.device),
            "gpu_name": self.gpu_name,
            "gpu_total_gb": round(self.gpu_total / _GiB, 2)
            if self.gpu_total else None,
            "gpu_free_gb": round(self.gpu_free / _GiB, 2)
            if self.gpu_free else None,
            "cpu_count": self.cpu_count,
            "ram_total_gb": round(self.ram_total / _GiB, 2),
            "ram_available_gb": round(self.ram_available / _GiB, 2),
            "amp": self.use_amp,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "batch_scale": self.batch_scale,
            "batch_size_mode": self.cfg["batch_size"],
            "query_batch_mode": self.cfg["query_batch"],
            "cudnn_benchmark": bool(self.cfg["cudnn_benchmark"]),
            "allow_tf32": bool(self.cfg["allow_tf32"]),
            "compile_teacher": bool(self.cfg["compile_teacher"]),
        }
        return s

    def log_summary(self):
        s = self.summary()
        gpu = f"{s['gpu_name']} ({s['gpu_free_gb']}/{s['gpu_total_gb']} GiB free)" \
            if s["gpu_name"] else "none"
        self._log(f"[hardware] device={s['device']}  gpu={gpu}  "
                  f"cpu_count={s['cpu_count']}  "
                  f"ram={s['ram_available_gb']}/{s['ram_total_gb']} GiB free")
        self._log(f"[hardware] amp={s['amp']}  batch_scale=x{s['batch_scale']:g}  "
                  f"num_workers={s['num_workers']}  pin_memory={s['pin_memory']}  "
                  f"tf32={s['allow_tf32']}  cudnn_benchmark={s['cudnn_benchmark']}")

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)


def default_hardware(device=None, logger=None):
    """A HardwareManager for callers that have no hardware config section."""
    return HardwareManager({}, cli_device=device, logger=logger)
