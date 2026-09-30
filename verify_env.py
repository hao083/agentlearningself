"""
环境验证脚本 —— 检查 PyTorch 是否安装成功、GPU 是否可用。

用法：
    conda activate pytorch
    python verify_env.py
"""

import sys

print("=" * 52)
print("  PyTorch 环境验证")
print("=" * 52)

print(f"{'Python 解释器':<14}: {sys.executable}")
print(f"{'Python 版本':<14}: {sys.version.split()[0]}")

try:
    import torch
except ImportError:
    print("\n[失败] 没找到 torch —— 说明当前解释器里没装 PyTorch。")
    print("       请确认已 conda activate pytorch，")
    print("       或在 VS Code 里把解释器切到 D:\\Anaconda\\envs\\pytorch\\python.exe")
    sys.exit(1)

print(f"{'PyTorch 版本':<14}: {torch.__version__}")
print(f"{'编译 CUDA':<14}: {torch.version.cuda}")

# 核心判定
cuda_ok = torch.cuda.is_available()
print(f"{'CUDA 可用':<14}: {cuda_ok}")
print(f"{'cuDNN 版本':<14}: {torch.backends.cudnn.version()}")

if cuda_ok:
    print(f"{'显卡名称':<14}: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"{'算力 (CC)':<14}: {cap[0]}.{cap[1]}")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"{'显存总量':<14}: {total:.2f} GB")

    # 实跑一次矩阵乘法，确认 GPU 真的能算
    print("-" * 52)
    print("跑一次 GPU 矩阵乘法实测 ...")
    a = torch.randn(2000, 2000, device="cuda")
    b = torch.randn(2000, 2000, device="cuda")
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 10 * 1000
    print(f"  2000x2000 矩阵乘法 x10，平均单次 {dt:.2f} ms")
    print(f"  结果张量所在设备: {c.device}")

    # 对比 CPU
    a_c = a.cpu()
    b_c = b.cpu()
    for _ in range(2):
        _ = a_c @ b_c
    t0 = time.perf_counter()
    for _ in range(3):
        _c = a_c @ b_c
    dt_cpu = (time.perf_counter() - t0) / 3 * 1000
    print(f"  同尺寸 CPU 版平均单次 {dt_cpu:.2f} ms")
    print(f"  >>> GPU 比 CPU 快约 {dt_cpu / dt:.1f} 倍")
    print("-" * 52)
    print("\n[成功] GPU 可用，PyTorch 安装正确。")
else:
    print("\n[失败] CUDA 不可用 —— 装成了 CPU 版，或驱动不匹配。")
    print("       请重装：")
    print("       pip install torch torchvision torchaudio "
          "--index-url https://download.pytorch.org/whl/cu121")

print("=" * 52)
