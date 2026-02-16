import os
import sys
import time
import threading
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Set, List, Tuple, Dict
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError

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


def get_gpu_memory_info(gpu_id: int, process_allocated_memory: Dict[int, int] = None) -> Dict[str, float]:
    """
    Get GPU memory information for a specific GPU using nvidia-smi for accurate per-process tracking.

    Args:
        gpu_id: GPU device ID
        process_allocated_memory: Dictionary tracking memory allocated by each subprocess (in bytes)

    Returns:
        Dictionary with 'total', 'used', 'free' in GB, and 'free_percent'
    """
    if not CUDA_AVAILABLE:
        return {'total': 0, 'used': 0, 'free': 0, 'free_percent': 0}

    try:
        # First try to get memory info using nvidia-smi for accurate per-process tracking
        import subprocess
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

    # Fallback to torch-based memory info
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
    """Monitor and manage GPU allocation based on available VRAM."""

    def __init__(self, min_vram_gb: float = 8.0, min_free_percent: float = 10.0):
        """
        Initialize GPU memory monitor.

        Args:
            min_vram_gb: Minimum required VRAM in GB
            min_free_percent: Minimum free VRAM percentage
        """
        self.min_vram_gb = min_vram_gb
        self.min_free_percent = min_free_percent
        self.alloc_lock = threading.Lock()
        self.reserved_gpus: Set[int] = set()  # Track reserved GPUs
        self.process_allocated_memory: Dict[int, int] = {}  # Track memory allocated by each process (in bytes)
        self.subprocess_pids: Set[int] = set()  # Track PIDs of spawned subprocesses

        # Get initial GPU status
        self.gpu_status = {}
        for gpu_id in range(NUM_GPUS):
            self.gpu_status[gpu_id] = get_gpu_memory_info(gpu_id, self.process_allocated_memory)

    def is_available(self, gpu_id: int) -> bool:
        """
        Check if a GPU has sufficient memory available and is not reserved.

        Args:
            gpu_id: GPU ID to check

        Returns:
            True if GPU has sufficient memory and is not reserved
        """
        if not CUDA_AVAILABLE or gpu_id >= NUM_GPUS:
            return False

        # Check if GPU is already reserved
        with self.alloc_lock:
            if gpu_id in self.reserved_gpus:
                return False  # GPU is reserved by another thread

        info = get_gpu_memory_info(gpu_id, self.process_allocated_memory)

        # Check both absolute GB and percentage thresholds
        has_enough_gb = info['free'] >= self.min_vram_gb
        has_enough_percent = info['free_percent'] >= self.min_free_percent

        return has_enough_gb and has_enough_percent

    def get_best_available_gpu(self) -> Optional[int]:
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

        Immediately reserve the GPU when found to prevent other threads from taking it.

        Args:
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

            gpu_id = self.get_best_available_gpu()
            if gpu_id is not None:
                # Immediately reserve this GPU for this thread
                with self.alloc_lock:
                    self.reserved_gpus.add(gpu_id)
                return gpu_id

            wait_time = min(check_interval, timeout - int(time.time() - start_time))
            if wait_time > 0:
                print(f"    [GPU] Waiting for available GPU... ({timeout - int(time.time() - start_time)}s remaining)")
                time.sleep(wait_time)

        print(f"    [GPU] Timeout waiting for available GPU")
        return None

    def release_gpu(self, gpu_id: int):
        """
        Release a previously reserved GPU.

        Args:
            gpu_id: GPU ID to release
        """
        with self.alloc_lock:
            if gpu_id in self.reserved_gpus:
                self.reserved_gpus.remove(gpu_id)
                print(f"    [GPU] Released GPU {gpu_id}")

    def update_subprocess_memory(self):
        """
        Update subprocess memory tracking by querying nvidia-smi for current GPU usage.
        This queries each GPU separately and stores per-GPU process memory.
        """
        try:
            # Clear old tracking
            with self.alloc_lock:
                self.process_allocated_memory.clear()

            # Query each GPU separately
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
                                        # Store with GPU-specific key: (gpu_id, pid) -> memory_mb
                                        self.process_allocated_memory[(gpu_id, pid)] = memory_mb * 1024 * 1024
                                    except ValueError:
                                        continue
        except Exception:
            # Silently fail if nvidia-smi query fails
            pass

    def print_status(self):
        """Print current GPU status including subprocess memory tracking."""
        if not CUDA_AVAILABLE or NUM_GPUS == 0:
            print("    [GPU Status] No CUDA GPUs available")
            return

        # Update subprocess memory before printing
        self.update_subprocess_memory()

        print(f"    [GPU Status] Found {NUM_GPUS} GPU(s):")
        for gpu_id in range(NUM_GPUS):
            info = get_gpu_memory_info(gpu_id, self.process_allocated_memory)
            gpu_name = torch.cuda.get_device_name(gpu_id)
            reserved_mark = " [RESERVED]" if gpu_id in self.reserved_gpus else ""
            print(f"      GPU {gpu_id}: {gpu_name}{reserved_mark}")
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
        """
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self.target_subfolder = target_subfolder
        self.token = token
        self.single_gpu = single_gpu
        self.test_mode = test_mode
        self.force_reprocess = force_reprocess

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
                min_free_percent=min_free_percent
            )
            self.use_gpu = True
        else:
            self.gpu_monitor = None
            self.use_gpu = False

        # Thread-safe sets
        self.processed_scenes: Set[str] = set()
        self.failed_scenes: List[Tuple[str, str]] = []
        self.lock = threading.Lock()
        self.stats = {
            'downloaded': 0,
            'compressed': 0,
            'deleted': 0,
            'failed': 0,
        }

        # Determine output directory for compression (same as download path)
        self.compression_output_dir = str(self.local_dir)

    def _get_scene_lang_feat_files(self) -> List[str]:
        """Get list of scene directories that need lang_feat.npy download.

        Skips scenes that already have grid_meta_data.json (compression complete).
        If grid_meta_data.json doesn't exist, delete lang_feat.npy to force re-download.
        If force_reprocess=True, delete both grid_meta_data.json and lang_feat.npy to force full reprocessing.
        """
        lang_feat_files = []
        base_path = self.local_dir / self.target_subfolder

        if not base_path.exists():
            return []

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

    def _compress_scene(self, scene_rel_path: str) -> bool:
        """
        Compress lang_feat.npy for a single scene using compress_grid_svd.py.

        Allocates a GPU with sufficient VRAM before compression.

        Args:
            scene_rel_path: Relative path to scene directory

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
                # Multi-GPU mode: wait for available GPU
                gpu_id = self.gpu_monitor.wait_for_available_gpu(timeout=self.gpu_wait_timeout)

                if gpu_id is None:
                    print(f"  [GPU] No available GPU for {scene_rel_path}, skipping...")
                    return False

                device = f"cuda:{gpu_id}"
                gpu_name = torch.cuda.get_device_name(gpu_id)
                print(f"  [GPU] Allocated GPU {gpu_id} ({gpu_name}) for {scene_rel_path}")

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

            print(f"  [Compress] Starting: {scene_rel_path} on {device}")

            # Test mode: use os.system to show output directly
            if self.test_mode:
                print(f"  [Compress] Test mode: showing real-time output")
                cmd_str = " ".join(cmd)
                returncode = os.system(cmd_str)
            else:
                # Normal mode: run compression subprocess
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour timeout per scene
                )
                returncode = result.returncode
            success = (returncode == 0)

            if success:
                print(f"  [Compress] Success: {scene_rel_path} on {device}")

                # Delete original lang_feat.npy to save space
                try:
                    lang_feat_path.unlink()
                    print(f"  [Cleanup] Deleted lang_feat.npy: {scene_rel_path}")
                    with self.lock:
                        self.stats['deleted'] += 1
                except Exception as e:
                    print(f"  [Cleanup] Warning: Could not delete {lang_feat_path}: {e}")
            else:
                print(f"  [Compress] Failed for {scene_rel_path}")
                # Only show stderr in non-test mode (test mode already showed output)
                if not self.test_mode and 'result' in locals() and result.stderr:
                    print(f"  [Compress] Error output:\n{result.stderr[:500]}")
                return False
        finally:
            # Release GPU when done (only in multi-GPU mode with monitor)
            if gpu_id is not None and self.single_gpu is None:
                self.gpu_monitor.release_gpu(gpu_id)

        return True

    def _download_and_compress_worker(self, scene_rel_path: str):
        """Worker thread: download then immediately compress a single scene."""
        scene_name = Path(scene_rel_path).name

        # Step 1: Download lang_feat.npy
        if self._download_lang_feat_for_scene(scene_rel_path):
            with self.lock:
                self.stats['downloaded'] += 1

            # Step 2: Immediately compress
            if self._compress_scene(scene_rel_path):
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

        if self.use_gpu:
            if self.single_gpu is not None:
                print(f"  GPU allocation: Single-GPU mode (GPU {self.single_gpu}, single-threaded)")
            else:
                print(f"  GPU allocation: Multi-GPU mode (min {self.gpu_monitor.min_vram_gb} GB VRAM)")

        # Process scenes with threading
        print("\n" + "=" * 60)
        print("Processing scenes (download + compress)...")
        print("=" * 60)

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {
                executor.submit(self._download_and_compress_worker, scene): scene
                for scene in scenes
            }

            for future in as_completed(futures):
                scene = futures[future]
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
    # Download and compress scannet train scenes
    python tools/download_and_compress_langfeat.py \\
        --repo_id GaussianWorld/scannet_mcmc_3dgs_lang_base \\
        --local_dir ./gaussian_train/scannet/train \\
        --target_subfolder train

    # With custom token and compression parameters
    python tools/download_and_compress_langfeat.py \\
        --repo_id GaussianWorld/scannet_mcmc_3dgs_lang_base \\
        --local_dir ./gaussian_train/scannet/train \\
        --target_subfolder train \\
        --token hf_xxx \\
        --grid_size 0.01 \\
        --ranks 8,16,32 \\
        --max_threads 4 \\
        --min_vram_gb 16

    # Force reprocess all scenes (delete existing compressed files)
    python tools/download_and_compress_langfeat.py \\
        --repo_id clapfor/scannetpp_v2_mcmc_3dgs_lang_base \\
        --local_dir /home/cyf/SceneSplat7k/scannetpp_v2 \\
        --target_subfolder train_grid1.0cm_chunk6x6_stride3x3 \\
        --token hf_xxx \\
        --rpca_tol 1e-3 \\
        --single_gpu 0 \\
        --force_reprocess
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
    )

    # Run processing
    processor.run()
