import os
import sys
import time
import threading
import subprocess
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Set, List, Tuple, Dict
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError
import numpy as np

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import torch for GPU detection and memory monitoring
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    NUM_GPUS = torch.cuda.device_count() if CUDA_AVAILABLE else 0
except ImportError:
    torch = None
    CUDA_AVAILABLE = False
    NUM_GPUS = 0

# Try to import pynvml for more reliable GPU memory queries
NVML_AVAILABLE = False
nvml_handle = None
try:
    import pynvml
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Just check if it works
    NVML_AVAILABLE = True
    print(f"    [GPU] Using pynvml for GPU memory monitoring")
except (ImportError, Exception) as e:
    print(f"    [GPU] pynvml not available: {e}")
    print(f"    [GPU] Will use nvidia-smi/torch fallback for GPU memory monitoring")


def get_gpu_memory_info(gpu_id: int, process_allocated_memory: Dict[int, int] = None) -> Dict[str, float]:
    """
    Get GPU memory information for a specific GPU.

    Tries multiple methods in order of reliability:
    1. pynvml (NVIDIA management library) - most reliable
    2. nvidia-smi command - system-wide info but may fail
    3. torch.cuda - current process only

    Args:
        gpu_id: GPU device ID
        process_allocated_memory: Dictionary tracking memory allocated by each subprocess (in bytes)

    Returns:
        Dictionary with 'total', 'used', 'free' in GB, and 'free_percent'
    """
    if not CUDA_AVAILABLE:
        return {'total': 0, 'used': 0, 'free': 0, 'free_percent': 0}

    # Method 1: Try pynvml (most reliable)
    if NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            total_memory = mem_info.total / (1024 ** 3)  # Convert to GB
            used_memory = mem_info.used / (1024 ** 3)
            free_memory = mem_info.free / (1024 ** 3)
            free_percent = (free_memory / total_memory) * 100 if total_memory > 0 else 0

            return {
                'total': total_memory,
                'used': used_memory,
                'free': free_memory,
                'free_percent': free_percent,
            }
        except Exception as e:
            print(f"    [GPU Warning] pynvml query failed for GPU {gpu_id}: {e}")

    # Method 2: Try nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
             '--format=csv,noheader,nounits', f'--id={gpu_id}'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            values = result.stdout.strip().split(',')
            if len(values) >= 3:
                total_memory_mb = int(values[0].strip())
                used_memory_mb = int(values[1].strip())
                free_memory_mb = int(values[2].strip())

                # Convert to GB
                total_memory = total_memory_mb / 1024
                used_memory = used_memory_mb / 1024
                free_memory = free_memory_mb / 1024

                # nvidia-smi already accounts for all process memory, just use it directly
                free_percent = (free_memory / total_memory) * 100 if total_memory > 0 else 0
                return {
                    'total': total_memory,
                    'used': used_memory,
                    'free': free_memory,
                    'free_percent': free_percent,
                }
    except Exception as e:
        print(f"    [GPU Warning] nvidia-smi query failed for GPU {gpu_id}: {e}")

    # Method 3: Fallback to torch-based memory info (current process only)
    try:
        props = torch.cuda.get_device_properties(gpu_id)
        total_memory = props.total_memory / (1024 ** 3)  # Convert to GB
        # Get current memory usage
        reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)

        # If we have process tracking, add subprocess memory for this GPU only
        if process_allocated_memory:
            gpu_process_memory = sum(
                mem for (gid, _), mem in process_allocated_memory.items()
                if gid == gpu_id
            )
            subprocess_used = gpu_process_memory / (1024 ** 3)  # Convert bytes to GB
            allocated += subprocess_used

        # Estimate free memory
        free_memory = total_memory - max(reserved, allocated)
        free_percent = (free_memory / total_memory) * 100 if total_memory > 0 else 0

        return {
            'total': total_memory,
            'used': allocated,
            'free': free_memory,
            'free_percent': free_percent,
        }
    except Exception as e:
        print(f"    [GPU Warning] Could not get memory info for GPU {gpu_id}: {e}")
        return {'total': 0, 'used': 0, 'free': 0, 'free_percent': 0}


def print_gpu_status():
    """Print current GPU status."""
    if not CUDA_AVAILABLE or NUM_GPUS == 0:
        print("    [GPU Status] No CUDA GPUs available")
        return

    print(f"    [GPU Status] Found {NUM_GPUS} GPU(s):")
    for gpu_id in range(NUM_GPUS):
        info = get_gpu_memory_info(gpu_id)
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"      GPU {gpu_id}: {gpu_name}")
        print(f"        Total: {info['total']:.2f} GB | "
              f"Used: {info['used']:.2f} GB | "
              f"Free: {info['free']:.2f} GB ({info['free_percent']:.1f}%)")


class GPUMemoryMonitor:
    """Monitor and manage GPU allocation based on available VRAM.

    Allocation Strategy:
    - If GPUs >= threads: Each thread gets a unique GPU (round-robin)
    - If GPUs < threads: Multiple threads can share GPUs, balanced evenly
    """

    def __init__(self, min_vram_gb: float = 8.0, min_free_percent: float = 10.0, max_threads: int = 2):
        """
        Initialize GPU memory monitor.

        Args:
            min_vram_gb: Minimum required VRAM in GB
            min_free_percent: Minimum free VRAM percentage
            max_threads: Maximum number of threads for GPU allocation
        """
        self.min_vram_gb = min_vram_gb
        self.min_free_percent = min_free_percent
        self.max_threads = max_threads
        self.alloc_lock = threading.Lock()
        # Track thread count per GPU instead of just reserved status
        self.gpu_thread_count: Dict[int, int] = {i: 0 for i in range(NUM_GPUS)}
        self.process_allocated_memory: Dict[int, int] = {}  # Track memory allocated by each process (in bytes)
        self.subprocess_pids: Set[int] = set()  # Track PIDs of spawned subprocesses

        # Get initial GPU status
        self.gpu_status = {}
        for gpu_id in range(NUM_GPUS):
            self.gpu_status[gpu_id] = get_gpu_memory_info(gpu_id, self.process_allocated_memory)

        # Pre-assign GPUs in round-robin fashion when GPUs >= threads
        self.round_robin_assignments: List[int] = []
        if NUM_GPUS >= max_threads:
            self.round_robin_assignments = list(range(max_threads))
            print(f"    [GPU] Round-robin mode: {max_threads} threads -> {max_threads} unique GPUs")
        else:
            print(f"    [GPU] Shared mode: {max_threads} threads -> {NUM_GPUS} GPUs")

    def is_available(self, gpu_id: int) -> bool:
        """
        Check if a GPU has sufficient memory available and can accept more threads.

        Args:
            gpu_id: GPU ID to check

        Returns:
            True if GPU has sufficient memory and can accept more threads
        """
        if not CUDA_AVAILABLE or gpu_id >= NUM_GPUS:
            return False

        with self.alloc_lock:
            current_threads = self.gpu_thread_count.get(gpu_id, 0)

        # In round-robin mode (GPUs >= threads), each GPU gets at most 1 thread
        if NUM_GPUS >= self.max_threads:
            if current_threads >= 1:
                return False
        # In shared mode (GPUs < threads), limit threads per GPU
        else:
            max_threads_per_gpu = (self.max_threads + NUM_GPUS - 1) // NUM_GPUS
            if current_threads >= max_threads_per_gpu:
                return False

        info = get_gpu_memory_info(gpu_id, self.process_allocated_memory)

        # Check both absolute GB and percentage thresholds
        has_enough_gb = info['free'] >= self.min_vram_gb
        has_enough_percent = info['free_percent'] >= self.min_free_percent

        return has_enough_gb and has_enough_percent

    def allocate_gpu_for_thread(self, thread_id: int, timeout: int = 300, check_interval: int = 5) -> Optional[int]:
        """
        Allocate a GPU for a specific thread using the new allocation strategy.

        Strategy:
        - If GPUs >= threads: Use round-robin, each thread gets a unique GPU
        - If GPUs < threads: Assign to GPU with most free memory

        Args:
            thread_id: Thread ID (0-indexed) for round-robin assignment
            timeout: Maximum wait time in seconds
            check_interval: Check interval in seconds

        Returns:
            GPU ID or None if timeout reached
        """
        start_time = time.time()
        last_update_time = 0

        while time.time() - start_time < timeout:
            # Update subprocess memory tracking periodically (every 10 seconds)
            if time.time() - last_update_time > 10:
                self.update_subprocess_memory()
                last_update_time = time.time()

            # Round-robin mode: GPUs >= threads
            if NUM_GPUS >= self.max_threads:
                assigned_gpu = self.round_robin_assignments[thread_id % len(self.round_robin_assignments)]
                if self.is_available(assigned_gpu):
                    # Reserve this GPU for this thread
                    with self.alloc_lock:
                        self.gpu_thread_count[assigned_gpu] += 1
                    print(f"    [GPU] Thread {thread_id} assigned to GPU {assigned_gpu} (round-robin)")
                    return assigned_gpu
                else:
                    print(f"    [GPU] Thread {thread_id} waiting for GPU {assigned_gpu}...")
            # Shared mode: GPUs < threads, find best available GPU
            else:
                best_gpu = self._get_best_available_gpu()
                if best_gpu is not None:
                    with self.alloc_lock:
                        self.gpu_thread_count[best_gpu] += 1
                    print(f"    [GPU] Thread {thread_id} assigned to GPU {best_gpu} (shared, {self.gpu_thread_count[best_gpu]} threads)")
                    return best_gpu
                else:
                    print(f"    [GPU] Thread {thread_id} waiting for available GPU...")

            wait_time = min(check_interval, timeout - int(time.time() - start_time))
            if wait_time > 0:
                time.sleep(wait_time)

        print(f"    [GPU] Thread {thread_id} timeout waiting for available GPU")
        return None

    def _get_best_available_gpu(self) -> Optional[int]:
        """
        Get the GPU with the most free memory that meets requirements.

        Returns:
            GPU ID or None if no suitable GPU available
        """
        if not CUDA_AVAILABLE or NUM_GPUS == 0:
            return None

        best_gpu = None
        max_free_memory = -1

        for gpu_id in range(NUM_GPUS):
            if self.is_available(gpu_id):
                info = get_gpu_memory_info(gpu_id, self.process_allocated_memory)
                if info['free'] > max_free_memory:
                    max_free_memory = info['free']
                    best_gpu = gpu_id

        return best_gpu

    def wait_for_available_gpu(self, timeout: int = 300, check_interval: int = 5) -> Optional[int]:
        """
        Wait for a GPU to become available with sufficient memory.

        This method is deprecated in favor of allocate_gpu_for_thread,
        but kept for backward compatibility.

        Args:
            timeout: Maximum wait time in seconds
            check_interval: Check interval in seconds

        Returns:
            GPU ID or None if timeout reached
        """
        return self._get_best_available_gpu()

    def release_gpu(self, gpu_id: int):
        """
        Release a previously allocated GPU (decrement thread count).

        Args:
            gpu_id: GPU ID to release
        """
        with self.alloc_lock:
            if gpu_id in self.gpu_thread_count and self.gpu_thread_count[gpu_id] > 0:
                self.gpu_thread_count[gpu_id] -= 1
                print(f"    [GPU] Released GPU {gpu_id} ({self.gpu_thread_count[gpu_id]} threads remaining)")

    def update_subprocess_memory(self):
        """
        Update subprocess memory tracking by querying GPU usage.

        Tries multiple methods in order:
        1. pynvml (NVIDIA management library) - most reliable
        2. nvidia-smi command - may fail in some environments
        This queries each GPU separately and stores per-GPU process memory.
        """
        try:
            # Clear old tracking
            with self.alloc_lock:
                self.process_allocated_memory.clear()

            # Method 1: Try pynvml
            if NVML_AVAILABLE:
                for gpu_id in range(NUM_GPUS):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

                        with self.alloc_lock:
                            for proc in procs:
                                # Store with GPU-specific key: (gpu_id, pid) -> memory_bytes
                                self.process_allocated_memory[(gpu_id, proc.pid)] = proc.usedGpuMemory
                        return  # Success, skip nvidia-smi
                    except Exception:
                        continue

            # Method 2: Fallback to nvidia-smi
            for gpu_id in range(NUM_GPUS):
                result = subprocess.run(
                    ['nvidia-smi', '--query-compute-apps=pid,used_memory',
                     f'--id={gpu_id}', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode == 0:
                    with self.alloc_lock:
                        # Store per-GPU process memory using (gpu_id, pid) as key
                        for line in result.stdout.strip().split('\n'):
                            if line:
                                parts = line.split(',')
                                if len(parts) >= 2:
                                    try:
                                        pid = int(parts[0].strip())
                                        memory_mb = int(parts[1].strip())
                                        # Store with GPU-specific key: (gpu_id, pid) -> memory_bytes
                                        self.process_allocated_memory[(gpu_id, pid)] = memory_mb * 1024 * 1024
                                    except ValueError:
                                        continue
        except Exception:
            # Silently fail if both methods fail
            pass

    def print_status(self):
        """Print current GPU status including thread count tracking."""
        if not CUDA_AVAILABLE or NUM_GPUS == 0:
            print("    [GPU Status] No CUDA GPUs available")
            return

        # Update subprocess memory before printing
        self.update_subprocess_memory()

        print(f"    [GPU Status] Found {NUM_GPUS} GPU(s):")
        for gpu_id in range(NUM_GPUS):
            info = get_gpu_memory_info(gpu_id, self.process_allocated_memory)
            gpu_name = torch.cuda.get_device_name(gpu_id)
            with self.alloc_lock:
                thread_count = self.gpu_thread_count.get(gpu_id, 0)
            thread_mark = f" [{thread_count} thread(s)]" if thread_count > 0 else ""
            print(f"      GPU {gpu_id}: {gpu_name}{thread_mark}")
            print(f"        Total: {info['total']:.2f} GB | "
                  f"Used: {info['used']:.2f} GB | "
                  f"Free: {info['free']:.2f} GB ({info['free_percent']:.1f}%)")


class LangFeatDownloadCompressor:
    """Download lang_feat.npy and immediately compress using SVD."""

    def __init__(
        self,
        repo_id: str,
        local_dir: str,
        target_subfolder: str = "",
        token: str = '',
        grid_size: float = 0.01,
        ranks: str = "8,16,32",
        max_threads: int = 2,
        compression_device: str = "cuda",
        no_rpca: bool = False,
        rpca_max_iter: int = 50,
        rpca_tol: float = 1e-3,
        max_retries: int = 10,
        initial_delay: int = 1,
        min_vram_gb: float = 16.0,
        min_free_percent: float = 10.0,
        gpu_wait_timeout: int = 600,
        single_gpu: Optional[int] = None,
        test_mode: bool = False,
        force_reprocess: bool = False,
        stats_only: bool = False,
        skip_download: bool = False,
        delete_after_compress: bool = False,
    ):
        """
        Initialize the downloader and compressor.

        Args:
            repo_id: HuggingFace repository ID
            local_dir: Local directory to save files
            target_subfolder: Target subfolder in the repository
            token: HuggingFace access token
            grid_size: Grid size for SVD compression
            max_threads: Maximum number of concurrent downloads/compressions
            max_retries: Maximum download retries per file (0 = infinite)
            initial_delay: Initial retry delay in seconds
            single_gpu: GPU ID to use for single-GPU single-thread mode (overrides max_threads to 1)
            test_mode: If True, only process the first scene for testing
            force_reprocess: If True, force reprocess all scenes including already compressed ones
            stats_only: If True, only show statistics of existing compressed scenes without downloading/compressing
            skip_download: If True, skip download and only compress scenes with existing lang_feat.npy files
            delete_after_compress: If True, delete lang_feat.npy after successful compression (for training data)
        """
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self.target_subfolder = target_subfolder
        self.token = token
        self.single_gpu = single_gpu
        self.test_mode = test_mode
        self.force_reprocess = force_reprocess
        self.stats_only = stats_only
        self.skip_download = skip_download
        self.delete_after_compress = delete_after_compress

        # Single GPU mode: force single thread
        if single_gpu is not None:
            max_threads = 1

        self.grid_size = grid_size
        # Handle both string and list input for ranks
        if isinstance(ranks, str):
            self.ranks = [int(r.strip()) for r in ranks.split(',')]
        elif isinstance(ranks, list):
            self.ranks = [int(r) if isinstance(r, str) else r for r in ranks]
        else:
            self.ranks = list(ranks)
        self.max_threads = max_threads
        self.compression_device = compression_device
        self.use_rpca = not no_rpca
        self.rpca_max_iter = rpca_max_iter
        self.rpca_tol = rpca_tol
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.min_vram_gb = min_vram_gb
        self.min_free_percent = min_free_percent
        self.gpu_wait_timeout = gpu_wait_timeout

        # Initialize GPU memory monitor
        if compression_device.startswith("cuda") and CUDA_AVAILABLE and NUM_GPUS > 0:
            self.gpu_monitor = GPUMemoryMonitor(
                min_vram_gb=min_vram_gb,
                min_free_percent=min_free_percent,
                max_threads=max_threads
            )
            self.use_gpu = True
        else:
            self.gpu_monitor = None
            self.use_gpu = False

        # Thread-safe sets and thread ID tracking
        self.processed_scenes: Set[str] = set()
        self.failed_scenes: List[Tuple[str, str]] = []
        self.lock = threading.Lock()
        self.stats = {
            'downloaded': 0,
            'compressed': 0,
            'deleted': 0,
            'failed': 0,
        }
        self.total_scenes = 0
        self.worker_thread_counter = 0  # Track worker thread IDs

        # Determine output directory for compression (same as download path)
        self.compression_output_dir = str(self.local_dir)

    def _get_scene_lang_feat_files(self) -> List[str]:
        """Get list of scene directories that need lang_feat.npy download.

        Skips scenes that already have grid_meta_data.json (compression complete).
        If grid_meta_data.json doesn't exist, delete lang_feat.npy to force re-download.
        If force_reprocess=True, delete both grid_meta_data.json and lang_feat.npy to force full reprocessing.
        If skip_download=True, find scenes with lang_feat.npy that need compression.
        """
        lang_feat_files = []
        base_path = self.local_dir / self.target_subfolder

        if not base_path.exists():
            return []

        # Skip download mode: find scenes with lang_feat.npy that need compression
        if self.skip_download:
            for item in base_path.rglob('*'):
                if item.is_dir():
                    coord_path = item / "coord.npy"
                    lang_feat_path = item / "lang_feat.npy"
                    grid_meta_path = item / "grid_meta_data.json"

                    # Need both coord.npy and lang_feat.npy, but not compressed yet
                    if coord_path.exists() and lang_feat_path.exists() and not grid_meta_path.exists():
                        rel_path = item.relative_to(self.local_dir)
                        lang_feat_files.append(str(rel_path).replace('\\', '/'))
            return sorted(lang_feat_files)

        # Normal mode: find scenes that need download
        # Find all directories that contain coord.npy but might not have lang_feat.npy
        for item in base_path.rglob('*'):
            if item.is_dir():
                coord_path = item / "coord.npy"
                lang_feat_path = item / "lang_feat.npy"
                grid_meta_path = item / "grid_meta_data.json"

                if coord_path.exists():
                    rel_path = item.relative_to(self.local_dir)

                    # Force reprocess mode: delete existing compressed files
                    if self.force_reprocess and grid_meta_path.exists():
                        print(f"  [Force Reprocess] Deleting compressed files: {rel_path}")
                        try:
                            # Delete grid_meta_data.json
                            grid_meta_path.unlink()
                            # Delete compressed SVD files (U, S, V matrices)
                            for svd_file in item.glob("lang_feat_grid_svd_r*.npz"):
                                svd_file.unlink()
                            print(f"  [Force Reprocess] Deleted compressed files for: {rel_path}")
                        except Exception as e:
                            print(f"  [Force Reprocess] Warning: Could not delete compressed files for {rel_path}: {e}")

                    # Skip if already compressed (grid_meta_data.json exists)
                    if grid_meta_path.exists():
                        print(f"  [Skip] Already compressed: {rel_path}")
                        continue

                    # If grid_meta_data.json doesn't exist, need to process
                    # Delete existing lang_feat.npy to force re-download
                    if lang_feat_path.exists():
                        print(f"  [Cleanup] Deleting existing lang_feat.npy for re-download: {rel_path}")
                        try:
                            lang_feat_path.unlink()
                        except Exception as e:
                            print(f"  [Cleanup] Warning: Could not delete {lang_feat_path}: {e}")

                    # Add scene to download list
                    lang_feat_files.append(str(rel_path).replace('\\', '/'))

        return sorted(lang_feat_files)

    def _print_progress(self):
        """Print current processing progress."""
        completed = self.stats['compressed'] + self.stats['failed']
        total = self.total_scenes
        if total > 0:
            print(f"  [Progress] {completed}/{total} scenes processed")

    def _check_metadata_completeness(self, meta_file: Path) -> Tuple[bool, List[str]]:
        """
        Check if grid_meta_data.json contains all required statistics.

        Args:
            meta_file: Path to grid_meta_data.json file

        Returns:
            Tuple of (is_complete, missing_keys)
        """
        try:
            with open(meta_file, 'r') as f:
                meta_data = json.load(f)

            missing_keys = []
            for r in self.ranks:
                error_key = f'reconstruction_error_r{r}'
                energy_key = f'rank_energy_ratio_r{r}'
                if error_key not in meta_data:
                    missing_keys.append(error_key)
                if energy_key not in meta_data:
                    missing_keys.append(energy_key)

            return len(missing_keys) == 0, missing_keys
        except Exception as e:
            return False, ['file_read_error']

    def _collect_grid_meta_stats(self) -> Tuple[Dict[str, any], List[str]]:
        """
        Collect statistics from all existing grid_meta_data.json files.

        Also identifies scenes with incomplete metadata that need re-processing.

        Returns:
            Tuple of (statistics dictionary, list of scenes needing re-processing)
        """
        base_path = self.local_dir / self.target_subfolder
        stats = {
            'total_scenes': 0,
            'num_grids_list': [],
            'grid_point_counts_mean_list': [],
            'num_points_with_grid_list': [],
            'reconstruction_errors': {},  # r -> [error1, error2, ...]
            'rank_energy_ratios': {},  # r -> [ratio1, ratio2, ...]
        }
        incomplete_scenes = []  # List of scene_rel_path that need re-processing

        if not base_path.exists():
            return stats, incomplete_scenes

        # Find all grid_meta_data.json files
        from tqdm import tqdm
        for meta_file in tqdm(base_path.rglob("grid_meta_data.json"), desc="Collecting grid metadata stats"):
            try:
                with open(meta_file, 'r') as f:
                    meta_data = json.load(f)

                # Check metadata completeness
                is_complete, missing_keys = self._check_metadata_completeness(meta_file)
                if not is_complete:
                    # Get relative path from local_dir
                    scene_dir = meta_file.parent
                    rel_path = scene_dir.relative_to(self.local_dir)
                    scene_rel_path = str(rel_path).replace('\\', '/')
                    incomplete_scenes.append(scene_rel_path)
                    print(f"  [Incomplete Metadata] {scene_rel_path} missing: {missing_keys}")
                    # Still count it and collect whatever stats we have
                    stats['total_scenes'] += 1
                    continue

                stats['total_scenes'] += 1

                # Collect basic stats
                if 'num_grids' in meta_data:
                    stats['num_grids_list'].append(meta_data['num_grids'])
                if 'grid_point_counts' in meta_data:
                    stats['grid_point_counts_mean_list'].append(meta_data['grid_point_counts'])
                if 'num_points_with_grid' in meta_data:
                    stats['num_points_with_grid_list'].append(meta_data['num_points_with_grid'])

                # Collect reconstruction errors and energy ratios for each rank
                for key, value in meta_data.items():
                    if key.startswith('reconstruction_error_r'):
                        rank = key.split('_r')[-1]
                        if rank not in stats['reconstruction_errors']:
                            stats['reconstruction_errors'][rank] = []
                        stats['reconstruction_errors'][rank].append(value)
                    elif key.startswith('rank_energy_ratio_r'):
                        rank = key.split('_r')[-1]
                        if rank not in stats['rank_energy_ratios']:
                            stats['rank_energy_ratios'][rank] = []
                        stats['rank_energy_ratios'][rank].append(value)

            except Exception as e:
                print(f"  [Warning] Could not read {meta_file}: {e}")
                continue

        return stats, incomplete_scenes

    def _reprocess_incomplete_scenes(self, incomplete_scenes: List[str]) -> Dict[str, int]:
        """
        Re-download and compress scenes with incomplete metadata using multi-threading.

        For each incomplete scene:
        1. Delete existing grid_meta_data.json and SVD npz files
        2. Re-download lang_feat.npy (if not skip_download mode)
        3. Re-compress the scene
        4. After compression, verify JSON completeness
        5. If still incomplete, preserve lang_feat.npy for manual inspection

        Args:
            incomplete_scenes: List of scene relative paths needing re-processing

        Returns:
            Dictionary with reprocessing statistics
        """
        reprocess_stats = {
            'total': len(incomplete_scenes),
            'reprocessed': 0,
            'failed': 0,
            'still_incomplete': 0,
        }

        if not incomplete_scenes:
            print("\n  [Reprocess] No incomplete scenes found.")
            return reprocess_stats

        print(f"\n  [Reprocess] Found {len(incomplete_scenes)} scenes with incomplete metadata")
        print(f"  [Reprocess] These scenes will be re-downloaded and re-compressed...")

        # Print GPU status
        if self.use_gpu:
            print("\n  [Reprocess] GPU Status:")
            self.gpu_monitor.print_status()

        # Use multi-threading for reprocessing
        print(f"  [Reprocess] Using multi-threaded processing (max {self.max_threads} threads)")

        from tqdm import tqdm
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            # Assign thread_id to each worker (round-robin based on submission order)
            futures = {}
            for i, scene in enumerate(incomplete_scenes):
                thread_id = i % self.max_threads
                future = executor.submit(self._reprocess_single_scene_worker, scene, thread_id)
                futures[future] = (scene, thread_id)

            # Track progress with tqdm
            completed = 0
            with tqdm(total=len(incomplete_scenes), desc="Reprocessing incomplete scenes", unit="scene") as pbar:
                for future in as_completed(futures):
                    scene, thread_id = futures[future]
                    completed += 1
                    try:
                        result = future.result()
                        if result == 'success':
                            reprocess_stats['reprocessed'] += 1
                            pbar.set_postfix({'success': reprocess_stats['reprocessed'], 'failed': reprocess_stats['failed']})
                        elif result == 'still_incomplete':
                            reprocess_stats['still_incomplete'] += 1
                            pbar.set_postfix({'still_incomplete': reprocess_stats['still_incomplete'], 'failed': reprocess_stats['failed']})
                        else:  # failed
                            reprocess_stats['failed'] += 1
                            pbar.set_postfix({'success': reprocess_stats['reprocessed'], 'failed': reprocess_stats['failed']})
                    except Exception as e:
                        print(f"\n  [Error] Exception during reprocessing of {scene}: {e}")
                        import traceback
                        traceback.print_exc()
                        reprocess_stats['failed'] += 1
                    pbar.update(1)

        return reprocess_stats

    def _reprocess_single_scene_worker(self, scene_rel_path: str, thread_id: int = 0) -> str:
        """
        Worker function for reprocessing a single incomplete scene.

        Args:
            scene_rel_path: Relative path to scene directory
            thread_id: Thread ID (0-indexed) for GPU allocation

        Returns:
            'success' if reprocessing completed successfully
            'still_incomplete' if metadata still incomplete after reprocessing
            'failed' if reprocessing failed
        """
        from tqdm import tqdm

        scene_dir = self.local_dir / scene_rel_path
        grid_meta_path = scene_dir / "grid_meta_data.json"

        try:
            # Step 1: Delete existing compressed files
            tqdm.write(f"  [Reprocess] Cleaning up: {scene_rel_path}")
            if grid_meta_path.exists():
                grid_meta_path.unlink()

            # Delete SVD npz files
            for svd_file in scene_dir.glob("lang_feat_grid_svd_r*.npz"):
                svd_file.unlink()

            # Step 2: Ensure lang_feat.npy exists (download if needed, unless skip_download)
            lang_feat_path = scene_dir / "lang_feat.npy"

            if not lang_feat_path.exists() and not self.skip_download:
                tqdm.write(f"    [Download] Re-downloading lang_feat.npy for {scene_rel_path}")
                if self._download_lang_feat_for_scene(scene_rel_path):
                    with self.lock:
                        self.stats['downloaded'] += 1
                else:
                    tqdm.write(f"    [Error] Failed to download lang_feat.npy for {scene_rel_path}")
                    return 'failed'
            elif not lang_feat_path.exists() and self.skip_download:
                tqdm.write(f"    [Skip] lang_feat.npy missing but skip_download=True")
                return 'failed'

            # Step 3: Re-compress the scene
            tqdm.write(f"    [Compress] Re-compressing {scene_rel_path}")
            if self._compress_scene(scene_rel_path, thread_id=thread_id):
                # Step 4: Verify JSON completeness
                is_complete, missing_keys = self._check_metadata_completeness(grid_meta_path)
                if not is_complete:
                    tqdm.write(f"    [Warning] Metadata still incomplete: {missing_keys}")
                    return 'still_incomplete'
                else:
                    tqdm.write(f"    [Success] {scene_rel_path} reprocessed successfully")
                    # Delete lang_feat.npy only if metadata is complete AND delete_after_compress is True
                    if self.delete_after_compress and lang_feat_path.exists():
                        lang_feat_path.unlink()
                        with self.lock:
                            self.stats['deleted'] += 1
                    return 'success'
            else:
                tqdm.write(f"    [Error] Compression failed for {scene_rel_path}")
                return 'failed'

        except Exception as e:
            tqdm.write(f"    [Error] Exception during reprocessing of {scene_rel_path}: {e}")
            return 'failed'

    def _print_grid_meta_stats(self, stats: Dict[str, any]):
        """
        Print aggregated grid metadata statistics.

        Args:
            stats: Statistics dictionary from _collect_grid_meta_stats
        """
        print("\n" + "=" * 60)
        print("Existing Compressed Scenes Statistics")
        print("=" * 60)

        if stats['total_scenes'] == 0:
            print("No compressed scenes found.")
            print("=" * 60)
            return

        print(f"Total compressed scenes: {stats['total_scenes']}")

        # Basic stats
        if stats['num_grids_list']:
            num_grids = stats['num_grids_list']
            print(f"\nGrids per scene:")
            print(f"  Min: {min(num_grids):,}")
            print(f"  Max: {max(num_grids):,}")
            print(f"  Mean: {np.mean(num_grids):.1f}")
            print(f"  Median: {np.median(num_grids):.1f}")

        if stats['grid_point_counts_mean_list']:
            grid_point_counts = stats['grid_point_counts_mean_list']
            print(f"\nMean grid point counts per scene:")
            print(f"  Min: {min(grid_point_counts):.2f}")
            print(f"  Max: {max(grid_point_counts):.2f}")
            print(f"  Mean: {np.mean(grid_point_counts):.2f}")
            print(f"  Median: {np.median(grid_point_counts):.2f}")

        if stats['num_points_with_grid_list']:
            num_points = stats['num_points_with_grid_list']
            print(f"\nPoints with grid per scene:")
            print(f"  Min: {min(num_points):,}")
            print(f"  Max: {max(num_points):,}")
            print(f"  Mean: {np.mean(num_points):.0f}")
            print(f"  Median: {np.median(num_points):.0f}")

        # Reconstruction errors by rank
        if stats['reconstruction_errors']:
            print(f"\nReconstruction Errors by Rank:")
            for rank in sorted(stats['reconstruction_errors'].keys(), key=int):
                errors = stats['reconstruction_errors'][rank]
                print(f"  Rank {rank}:")
                print(f"    Min: {min(errors):.6f}")
                print(f"    Max: {max(errors):.6f}")
                print(f"    Mean: {np.mean(errors):.6f}")
                print(f"    Median: {np.median(errors):.6f}")

        # Energy ratios by rank
        if stats['rank_energy_ratios']:
            print(f"\nRank Energy Ratios:")
            for rank in sorted(stats['rank_energy_ratios'].keys(), key=int):
                ratios = stats['rank_energy_ratios'][rank]
                print(f"  Rank {rank}:")
                print(f"    Min: {min(ratios):.6f}")
                print(f"    Max: {max(ratios):.6f}")
                print(f"    Mean: {np.mean(ratios):.6f}")
                print(f"    Median: {np.median(ratios):.6f}")

        print("=" * 60)

    def _download_lang_feat_for_scene(self, scene_rel_path: str) -> bool:
        """
        Download lang_feat.npy for a single scene using huggingface_hub.

        Uses hf_hub_download for efficient single-file download with connection pooling.

        Args:
            scene_rel_path: Relative path to scene directory

        Returns:
            True if download successful, False otherwise
        """
        retries = 0
        delay = self.initial_delay
        infinite_retry = (self.max_retries <= 0)

        # Construct the file path in the repository
        # Note: scene_rel_path already includes target_subfolder
        repo_file_path = f"{scene_rel_path}/lang_feat.npy"

        while infinite_retry or (retries <= self.max_retries):
            try:
                print(f"  [Download] Starting: {repo_file_path}")

                # Use hf_hub_download for efficient single-file download
                # This reuses connections and is more efficient than snapshot_download
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=repo_file_path,
                    repo_type="dataset",
                    local_dir=str(self.local_dir),
                    token=self.token,
                    resume_download=True,
                )

                # Verify file was downloaded
                lang_feat_path = self.local_dir / scene_rel_path / "lang_feat.npy"
                if lang_feat_path.exists():
                    file_size_mb = lang_feat_path.stat().st_size / (1024 * 1024)
                    print(f"  [Download] Success: {scene_rel_path} ({file_size_mb:.2f} MB)")
                    return True
                else:
                    print(f"  [Download] File not found after download: {scene_rel_path}")
                    return False

            except LocalEntryNotFoundError as e:
                print(f"  [Download] Local cache error for {scene_rel_path}: {e}")
            except HfHubHTTPError as e:
                print(f"  [Download] HTTP error for {scene_rel_path}: {e}")
            except Exception as e:
                print(f"  [Download] Error for {scene_rel_path}: {e}")

            retries += 1
            if not infinite_retry and retries > self.max_retries:
                print(f"  [Download] Max retries reached for {scene_rel_path}")
                break

            print(f"  [Download] Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2

        return False

    def _retry_compression_single_threaded(
        self,
        scene_dir: Path,
        cmd_str: str,
        max_retries: int = 1
    ) -> bool:
        """
        Retry compression in single-threaded mode until metadata is complete.

        This function is called when normal multi-threaded compression produces
        incomplete metadata. It uses os.system for real-time output and retries
        until the grid_meta_data.json is complete or max retries is reached.

        Args:
            scene_dir: Path to scene directory
            cmd_str: Command string to execute
            max_retries: Maximum number of retry attempts

        Returns:
            True if metadata is complete after retries, False otherwise
        """
        grid_meta_path = scene_dir / "grid_meta_data.json"
        lang_feat_path = scene_dir / "lang_feat.npy"

        for retry in range(max_retries):
            print(f"  [Retry] Attempt {retry + 1}/{max_retries} (single-threaded mode)...")
            returncode = os.system(cmd_str)

            if returncode == 0 and grid_meta_path.exists():
                is_complete, missing_keys = self._check_metadata_completeness(grid_meta_path)
                if is_complete:
                    print(f"  [Verified] Metadata complete after retry")
                    # Only delete lang_feat.npy if delete_after_compress is True
                    if self.delete_after_compress:
                        try:
                            lang_feat_path.unlink()
                            print(f"  [Cleanup] Deleted lang_feat.npy (metadata verified complete)")
                            with self.lock:
                                self.stats['deleted'] += 1
                        except Exception as e:
                            print(f"  [Cleanup] Warning: Could not delete {lang_feat_path}: {e}")
                    else:
                        print(f"  [Keep] lang_feat.npy preserved after retry")
                    return True

            # Cleanup incomplete files before retry
            if grid_meta_path.exists():
                grid_meta_path.unlink()
            for svd_file in scene_dir.glob("lang_feat_grid_svd_r*.npz"):
                svd_file.unlink()

        # All retries failed
        print(f"  [Error] All retries failed, metadata still incomplete")
        print(f"  [Warning] Preserving lang_feat.npy for manual inspection")
        return False

    def _compress_scene(self, scene_rel_path: str, thread_id: int = 0) -> bool:
        """
        Compress lang_feat.npy for a single scene using compress_grid_svd.py.

        Allocates a GPU with sufficient VRAM before compression.

        Args:
            scene_rel_path: Relative path to scene directory
            thread_id: Thread ID (0-indexed) for GPU allocation

        Returns:
            True if compression successful, False otherwise
        """
        scene_dir = self.local_dir / scene_rel_path

        # Check if lang_feat.npy exists
        lang_feat_path = scene_dir / "lang_feat.npy"
        if not lang_feat_path.exists():
            print(f"  [Compress] Skip: lang_feat.npy not found for {scene_rel_path}")
            return False

        # Allocate GPU with sufficient VRAM
        gpu_id = None
        device = self.compression_device

        if self.use_gpu:
            print(f"  [GPU] Allocating GPU for {scene_rel_path}...")
            self.gpu_monitor.print_status()

            # Single GPU mode: use the specified GPU directly
            if self.single_gpu is not None:
                gpu_id = self.single_gpu
                # Validate GPU ID
                if gpu_id >= NUM_GPUS:
                    print(f"  [GPU] Invalid GPU ID {gpu_id}, only {NUM_GPUS} GPU(s) available")
                    return False
                device = f"cuda:{gpu_id}"
                gpu_name = torch.cuda.get_device_name(gpu_id)
                print(f"  [GPU] Using fixed GPU {gpu_id} ({gpu_name}) for {scene_rel_path}")
            else:
                # Multi-GPU mode: use new allocation strategy
                gpu_id = self.gpu_monitor.allocate_gpu_for_thread(
                    thread_id=thread_id,
                    timeout=self.gpu_wait_timeout
                )

                if gpu_id is None:
                    print(f"  [GPU] No available GPU for {scene_rel_path}, skipping...")
                    return False

                device = f"cuda:{gpu_id}"
                gpu_name = torch.cuda.get_device_name(gpu_id)
                # print(f"  [GPU] Allocated GPU {gpu_id} ({gpu_name}) for {scene_rel_path}")

        try:
            # Build command for compress_grid_svd.py
            compress_script = PROJECT_ROOT / "tools" / "compress_grid_svd.py"

            cmd = [
                sys.executable,
                str(compress_script),
                "--data_dir", str(scene_dir),
                "--grid_size", str(self.grid_size),
                "--ranks", ",".join(map(str, self.ranks)),
                "--output_dir", str(Path(self.compression_output_dir) / self.target_subfolder),
                "--rpca_max_iter", str(self.rpca_max_iter),
                "--rpca_tol", str(self.rpca_tol),
                "--device", device,
            ]

            if not self.use_rpca:
                cmd.append("--no_rpca")

            cmd_str = " ".join(cmd)

            print(f"  [Compress] Starting: {scene_rel_path} on {device}")

            # Test mode: use os.system to show output directly
            if self.test_mode:
                print(f"  [Compress] Test mode: showing real-time output")
                returncode = os.system(cmd_str)
            else:
                # Normal mode: run compression subprocess (multi-threaded)
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour timeout per scene
                )
                returncode = result.returncode

            success = (returncode == 0)

            if not success:
                print(f"  [Compress] Failed for {scene_rel_path}")
                if not self.test_mode and 'result' in locals() and result.stderr:
                    print(f"  [Compress] Error output:\n{result.stderr[:500]}")
                return False

            print(f"  [Compress] Success: {scene_rel_path} on {device}")

            # Check metadata completeness
            grid_meta_path = scene_dir / "grid_meta_data.json"

            if grid_meta_path.exists():
                is_complete, missing_keys = self._check_metadata_completeness(grid_meta_path)

                if is_complete:
                    # Metadata is complete
                    print(f"  [Verified] Metadata complete for {scene_rel_path}")
                    # Only delete lang_feat.npy if delete_after_compress is True
                    if self.delete_after_compress:
                        try:
                            lang_feat_path.unlink()
                            print(f"  [Cleanup] Deleted lang_feat.npy: {scene_rel_path} (metadata verified complete)")
                            with self.lock:
                                self.stats['deleted'] += 1
                        except Exception as e:
                            print(f"  [Cleanup] Warning: Could not delete {lang_feat_path}: {e}")
                    else:
                        print(f"  [Keep] lang_feat.npy preserved: {scene_rel_path}")
                    return True
                else:
                    # Metadata is incomplete, switch to single-threaded mode and retry
                    print(f"  [Warning] Metadata incomplete (missing: {missing_keys})")
                    print(f"  [Retry] Switching to single-threaded mode...")

                    # Delete incomplete files before retry
                    grid_meta_path.unlink()
                    for svd_file in scene_dir.glob("lang_feat_grid_svd_r*.npz"):
                        svd_file.unlink()

                    # Call retry function (GPU is still allocated, will be released in finally block)
                    return self._retry_compression_single_threaded(scene_dir, cmd_str)
            else:
                # grid_meta_data.json doesn't exist, retry in single-threaded mode
                print(f"  [Warning] grid_meta_data.json not found after compression")
                print(f"  [Retry] Switching to single-threaded mode...")
                return self._retry_compression_single_threaded(scene_dir, cmd_str)

        finally:
            # Release GPU when done (only in multi-GPU mode with monitor)
            if gpu_id is not None and self.single_gpu is None:
                self.gpu_monitor.release_gpu(gpu_id)

    def _download_and_compress_worker(self, scene_rel_path: str, thread_id: int):
        """Worker thread: download then immediately compress a single scene.

        Args:
            scene_rel_path: Relative path to scene directory
            thread_id: Thread ID (0-indexed) for GPU allocation
        """
        scene_name = Path(scene_rel_path).name

        # Skip download mode: directly compress
        if self.skip_download:
            self._print_progress()
            if self._compress_scene(scene_rel_path, thread_id=thread_id):
                with self.lock:
                    self.stats['compressed'] += 1
                    self.processed_scenes.add(scene_name)
            else:
                with self.lock:
                    self.stats['failed'] += 1
                    self.failed_scenes.append((scene_rel_path, "compression failed"))
            return

        # Normal mode: download then compress
        # Step 1: Download lang_feat.npy
        self._print_progress()
        if self._download_lang_feat_for_scene(scene_rel_path):
            with self.lock:
                self.stats['downloaded'] += 1

            # Step 2: Immediately compress
            if self._compress_scene(scene_rel_path, thread_id=thread_id):
                with self.lock:
                    self.stats['compressed'] += 1
                    self.processed_scenes.add(scene_name)
            else:
                with self.lock:
                    self.stats['failed'] += 1
                    self.failed_scenes.append((scene_rel_path, "download failed"))
        else:
            with self.lock:
                self.stats['failed'] += 1
                self.failed_scenes.append((scene_rel_path, "compression failed"))

    def _print_summary(self):
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Downloaded: {self.stats['downloaded']}")
        print(f"  Compressed: {self.stats['compressed']}")
        print(f"  Deleted: {self.stats['deleted']}")
        print(f"  Failed: {self.stats['failed']}")

        if self.failed_scenes:
            print("\nFailed scenes:")
            for scene, reason in self.failed_scenes[:10]:
                print(f"  - {scene}: {reason}")
            if len(self.failed_scenes) > 10:
                print(f"  ... and {len(self.failed_scenes) - 10} more")

        print("=" * 60)

    def run(self):
        """Main execution: download and compress all scenes."""
        # Stats-only mode: just show statistics and exit
        if self.stats_only:
            from tqdm import tqdm

            existing_stats, incomplete_scenes = self._collect_grid_meta_stats()
            self._print_grid_meta_stats(existing_stats)

            # If incomplete scenes found, reprocess them
            if incomplete_scenes:
                tqdm.write("\n" + "=" * 60)
                tqdm.write("Incomplete Metadata Detected")
                tqdm.write("=" * 60)
                tqdm.write(f"Found {len(incomplete_scenes)} scenes with incomplete metadata.")
                tqdm.write("These scenes will be re-downloaded and re-compressed...")

                # Reprocess incomplete scenes
                reprocess_stats = self._reprocess_incomplete_scenes(incomplete_scenes)

                # Print reprocessing summary
                tqdm.write("\n" + "=" * 60)
                tqdm.write("Reprocessing Summary")
                tqdm.write("=" * 60)
                tqdm.write(f"  Total incomplete scenes: {reprocess_stats['total']}")
                tqdm.write(f"  Successfully reprocessed: {reprocess_stats['reprocessed']}")
                tqdm.write(f"  Failed: {reprocess_stats['failed']}")
                tqdm.write(f"  Still incomplete after reprocessing: {reprocess_stats['still_incomplete']}")

                # If reprocessing was successful, re-collect and print final stats
                if reprocess_stats['reprocessed'] > 0:
                    tqdm.write("\n" + "=" * 60)
                    tqdm.write("Re-collecting statistics after reprocessing...")
                    tqdm.write("=" * 60)
                    final_stats, _ = self._collect_grid_meta_stats()
                    self._print_grid_meta_stats(final_stats)
            return

        # Print GPU status at startup
        print("\n" + "=" * 60)
        print("GPU Detection")
        print("=" * 60)
        print_gpu_status()

        # Find all scenes that need lang_feat.npy download
        print("\n" + "=" * 60)
        print("Finding scenes requiring lang_feat.npy download...")
        print("=" * 60)

        scenes = self._get_scene_lang_feat_files()

        if not scenes:
            print("No scenes found requiring lang_feat.npy download.")
            print("All scenes may already have lang_feat.npy or compressed files.")
            return

        # Test mode: only process the first scene
        if self.test_mode:
            print("\n" + "=" * 60)
            print("TEST MODE: Only processing the first scene")
            print("=" * 60)
            scenes = scenes[:1]

        # Set total scenes for progress tracking
        self.total_scenes = len(scenes)

        print(f"Found {len(scenes)} scenes requiring download:")
        for scene in scenes[:5]:
            print(f"  - {scene}")
        if len(scenes) > 5:
            print(f"  ... and {len(scenes) - 5} more")

        print("\n" + "=" * 60)
        print("Configuration")
        print("=" * 60)
        print(f"  Repository: {self.repo_id}")
        print(f"  Local directory: {self.local_dir}")
        print(f"  Target subfolder: {self.target_subfolder}")
        print(f"  Grid size: {self.grid_size}m")
        print(f"  SVD ranks: {self.ranks}")
        print(f"  Max compression threads: {self.max_threads}")
        print(f"  Compression device: {self.compression_device}")
        print(f"  RPCA enabled: {self.use_rpca}")
        print(f"  Skip download: {self.skip_download}")
        print(f"  Delete lang_feat after compress: {self.delete_after_compress}")

        if self.use_gpu:
            if self.single_gpu is not None:
                print(f"  GPU allocation: Single-GPU mode (GPU {self.single_gpu}, single-threaded)")
            else:
                if NUM_GPUS >= self.max_threads:
                    print(f"  GPU allocation: Round-robin mode ({self.max_threads} threads -> {self.max_threads} unique GPUs)")
                else:
                    print(f"  GPU allocation: Shared mode ({self.max_threads} threads -> {NUM_GPUS} GPUs)")

        # Process scenes with threading
        print("\n" + "=" * 60)
        print("Processing scenes (download + compress)...")
        print("=" * 60)

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            # Assign thread_id to each worker (round-robin based on submission order)
            futures = {}
            for i, scene in enumerate(scenes):
                thread_id = i % self.max_threads
                future = executor.submit(self._download_and_compress_worker, scene, thread_id)
                futures[future] = (scene, thread_id)

            for future in as_completed(futures):
                scene, thread_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  [Error] Exception for {scene}: {e}")
                    with self.lock:
                        self.stats['failed'] += 1
                        self.failed_scenes.append((scene, str(e)))

        # Print summary
        self._print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download lang_feat.npy from HuggingFace and immediately compress using SVD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python tools/download_and_compress_langfeat.py \\
        --repo_id <repo_id> --local_dir <dir> --target_subfolder <subfolder>

    # Required parameters:
    #   --repo_id:          HuggingFace repository ID
    #   --local_dir:         Local directory to save files
    #   --target_subfolder:  Target subfolder in the repository

    # Optional mode flags (mutually exclusive):
    #   --stats_only:        Only show statistics of existing compressed scenes
    #   --skip_download:     Skip download, only compress existing lang_feat.npy files
    #   --force_reprocess:   Force reprocess all scenes (delete existing compressed files)
    #   --test_mode:         Only process the first scene for testing

    # Lang feat file management:
    #   --delete_after_compress:  Delete lang_feat.npy after successful compression
    #                           (useful for training data to save disk space)
    #                           Default: keep lang_feat.npy (for test/val datasets)

    # Optional parameters:
    #   --token:             HuggingFace access token (for private repos)
    #   --single_gpu N:      Use specific GPU ID, disable multi-threading
    #   --grid_size:         Grid size in meters (default: 0.01)
    #   --ranks:             SVD ranks, comma-separated (default: 8,16,32)
    #   --max_threads:       Max concurrent downloads/compressions (default: 2)
    #   --rpca_tol:          RPCA convergence tolerance (default: 1e-3)
    """
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., GaussianWorld/scannet_mcmc_3dgs_lang_base)",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        required=True,
        help="Local directory to save files",
    )
    parser.add_argument(
        "--target_subfolder",
        type=str,
        default="",
        help="Target subfolder in the repository (e.g., train, val, test)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="HuggingFace access token (required for private repositories)",
    )
    parser.add_argument(
        "--grid_size",
        type=float,
        default=0.01,
        help="Grid size in meters for SVD compression",
    )
    parser.add_argument(
        "--ranks",
        type=str,
        default="8,16,32",
        help="Comma-separated SVD ranks (e.g., '8,16,32')",
    )
    parser.add_argument(
        "--max_threads",
        type=int,
        default=2,
        help="Maximum number of concurrent downloads/compressions",
    )
    parser.add_argument(
        "--compression_device",
        type=str,
        default="cuda",
        help="Device for SVD compression (cuda/cpu)",
    )
    parser.add_argument(
        "--no_rpca",
        action="store_true",
        help="Disable RPCA preprocessing for faster compression",
    )
    parser.add_argument(
        "--rpca_max_iter",
        type=int,
        default=50,
        help="Maximum iterations for RPCA convergence",
    )
    parser.add_argument(
        "--rpca_tol",
        type=float,
        default=1e-3,
        help="Tolerance for RPCA convergence",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Maximum download retries per file (0 = infinite)",
    )
    parser.add_argument(
        "--initial_delay",
        type=int,
        default=1,
        help="Initial retry delay in seconds",
    )
    parser.add_argument(
        "--min_vram_gb",
        type=float,
        default=16.0,
        help="Minimum required VRAM in GB",
    )
    parser.add_argument(
        "--min_free_percent",
        type=float,
        default=10.0,
        help="Minimum free VRAM percentage",
    )
    parser.add_argument(
        "--gpu_wait_timeout",
        type=int,
        default=600,
        help="Maximum wait time for available GPU in seconds",
    )
    parser.add_argument(
        "--single_gpu",
        type=int,
        default=None,
        help="Single GPU mode: use specified GPU ID and disable multi-threading (overrides max_threads to 1)",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Test mode: only process the first scene for testing",
    )
    parser.add_argument(
        "--force_reprocess",
        action="store_true",
        help="Force reprocess all scenes, including those already compressed (deletes grid_meta_data.json and SVD files)",
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Only show statistics of existing compressed scenes without downloading/compressing",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download and only compress existing lang_feat.npy files (useful for val/test datasets)",
    )
    parser.add_argument(
        "--delete_after_compress",
        action="store_true",
        help="Delete lang_feat.npy after successful compression (useful for training data to save disk space)",
    )

    args = parser.parse_args()

    # Create and run processor
    processor = LangFeatDownloadCompressor(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        target_subfolder=args.target_subfolder,
        token=args.token,
        grid_size=args.grid_size,
        ranks=args.ranks.split(','),
        max_threads=args.max_threads,
        compression_device=args.compression_device,
        no_rpca=args.no_rpca,
        rpca_max_iter=args.rpca_max_iter,
        rpca_tol=args.rpca_tol,
        max_retries=args.max_retries,
        initial_delay=args.initial_delay,
        min_vram_gb=args.min_vram_gb,
        min_free_percent=args.min_free_percent,
        gpu_wait_timeout=args.gpu_wait_timeout,
        single_gpu=args.single_gpu,
        test_mode=args.test_mode,
        force_reprocess=args.force_reprocess,
        stats_only=args.stats_only,
        skip_download=args.skip_download,
        delete_after_compress=args.delete_after_compress,
    )

    # Run processing
    try:
        processor.run()
    finally:
        # Cleanup pynvml
        if NVML_AVAILABLE:
            pynvml.nvmlShutdown()
            print("    [GPU] pynvml shutdown complete")

