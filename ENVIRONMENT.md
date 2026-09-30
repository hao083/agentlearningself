# 开发环境说明

> 记录本机硬件配置与 Python / PyTorch 环境，便于以后迁移到实验室 GPU 服务器时复现。

## 一、硬件配置

| 项目 | 参数 |
| --- | --- |
| 机型 | LENOVO 82JQ（拯救者 R9000P 2021） |
| CPU | AMD Ryzen 7 5800H，8 核 16 线程，基准 3.2 GHz |
| 内存 | 15.86 GB |
| 显卡 | NVIDIA GeForce RTX 3070 Laptop GPU，**8 GB 显存** |
| 显卡驱动 | 546.30（支持 CUDA 12.3） |
| 操作系统 | Windows 11 家庭中文版，build 22631，64 位 |

## 二、Python 环境

| 项目 | 参数 |
| --- | --- |
| Anaconda | 24.9.2，安装位置 `D:\Anaconda` |
| base 环境 Python | 3.12.7 |
| **PyTorch 专用环境** | **`pytorch`**，路径 `D:\Anaconda\envs\pytorch`，Python 3.12.14 |

### 环境切换

```powershell
conda activate pytorch      # 进入 PyTorch 环境
conda deactivate            # 退出
conda env list              # 查看所有环境
```

### VS Code 中切换解释器

`Ctrl+Shift+P` → `Python: Select Interpreter` → 选择：

```
D:\Anaconda\envs\pytorch\python.exe
```

> 选好后，新开的终端会自动激活 `pytorch` 环境。

## 三、PyTorch 安装记录

| 项目 | 值 |
| --- | --- |
| PyTorch 版本 | 2.x + **cu121**（CUDA 12.1 编译版） |
| 安装方式 | pip，官方源 `https://download.pytorch.org/whl/cu121` |
| 兼容性说明 | 驱动支持 CUDA 12.3，向下兼容 12.1 版本，完全可用 |

### 关键结论：CUDA Toolkit 无需单独安装

PyTorch 的 pip 包已内置 CUDA 运行时，**不需要**去 NVIDIA 官网下载 CUDA Toolkit 或 cuDNN。

### 网络注意事项

- 本机沙箱/加速器代理为 `http://127.0.0.1:55124`（会变动）。
- 清华 conda 源经过该代理时**极慢甚至超时**（实测 15 秒），因此选用了**官方 PyTorch 源直连**（实测 0.87 秒）。
- 若将来需要走镜像，pip 的清华源已配置为全局默认（`~/AppData/Roaming/pip/pip.ini`），但 `--index-url` 参数会覆盖它。

## 四、验证方法

```powershell
conda activate pytorch
python verify_env.py
```

期望输出：

```
Python 解释器 : D:\Anaconda\envs\pytorch\python.exe
PyTorch 版本  : 2.x.x+cu121
CUDA 可用     : True
显卡名称      : NVIDIA GeForce RTX 3070 Laptop GPU
CUDA 版本     : 12.1
显存总量      : 8.00 GB
```

**判定标准：`CUDA 可用` 必须是 `True`。** 若为 `False`，说明装成了 CPU 版，需要重装。

## 五、以后迁到实验室服务器时

本地环境**不要往服务器搬**（Windows → Linux 不通用），正确做法是：

1. VS Code 用 Remote-SSH 连上服务器
2. 在服务器上 `conda create -n pytorch python=3.12`
3. 在服务器上重装 PyTorch（服务器一般用 Linux + CUDA 版）

参考服务器端安装命令（在服务器上执行）：

```bash
conda create -n pytorch python=3.12 -y
conda activate pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

*更新于 2026-09-30*
