import multiprocessing
import os
import sys
import contextlib


def child_target():
    """Child process target function - tries to import and init NVML."""
    print(f"[CHILD {os.getpid()}] Starting child process")
    
    try:
        from vllm.utils.import_utils import import_pynvml
        pynvml = import_pynvml()
        pynvml.nvmlInit()
        print("!!!nvmlInit in child")
        print(pynvml.nvmlDeviceGetCount())
        pynvml.nvmlShutdown()
        print("!!!nvmlShutdown in child")

        sys.exit(0)
    except Exception as e:
        print(f"[CHILD] ✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    print("=" * 70)
    print("NVML Initialization Test (WSL Reproduction)")
    print("=" * 70)
    print(f"[PARENT {os.getpid()}] Starting test\n")
    
    # Match basic.py: Check CUDA availability but DON'T initialize it
    print("[PARENT] Checking CUDA availability...")
    import torch
    print(f"[PARENT] CUDA available: {torch.cuda.is_available()}")
    print(f"[PARENT] Device count: {torch.cuda.device_count()}")
    print(f"[PARENT] CUDA initialized: {torch.cuda.is_initialized()}\n")
    # Note: We intentionally DON'T initialize CUDA here to match basic.py
    
    # Use get_mp_context() like vLLM does - it will use fork since CUDA not initialized
    print("[PARENT] Getting multiprocessing context...")
    from vllm.utils.system_utils import get_mp_context
    context = get_mp_context()
    print(f"[PARENT] Using context method: {context.get_start_method()}\n")
    
    # Create process object (like lines 120-128)
    print("[PARENT] Creating process object...")
    proc = context.Process(
        target=child_target,
        name="NVMLTestChild"
    )
    
    # Before starting, do NVML init/shutdown in parent (like lines 162-169)
    print("\n[PARENT] NVML init/shutdown before proc.start()...")
    with contextlib.nullcontext():
        print("[PARENT] Importing pynvml...")
        from vllm.utils.import_utils import import_pynvml
        pynvml = import_pynvml()
        
        print("[PARENT] Calling nvmlInit()...")
        pynvml.nvmlInit()
        print("[PARENT] ✓ nvmlInit() succeeded")
        
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"[PARENT] Device count: {device_count}")
        
        print("[PARENT] Calling nvmlShutdown()...")
        pynvml.nvmlShutdown()
        print("[PARENT] ✓ nvmlShutdown() succeeded")
        print("[PARENT] Done with NVML init/shutdown cycle\n")
    
    # Now start the process (like line 172)
    print("[PARENT] Starting child process...")
    proc.start()
    proc.join()
    
    exit_code = proc.exitcode
    
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    if exit_code == 0:
        print("✓ SUCCESS - No NVML initialization failure")
    else:
        print("✗ FAILED - NVML initialization issue reproduced!")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())