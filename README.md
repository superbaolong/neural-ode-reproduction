# Neural ODE 复现实验

本项目面向 RTX 3050 4GB（也支持 CPU），复现 Neural Ordinary Differential Equations 的核心现象：

1. 二维 Spiral 动力系统拟合；
2. MNIST 上 ODENet 与 ResNet 对比；
3. 直接反向传播与 Adjoint Method 对比；
4. ODE 求解器 tolerance 对 NFE、速度和准确率的影响。

代码默认使用 FP32，避免低显存设备上 ODE 数值不稳定。MNIST 使用原始 IDX 文件加载器，不依赖 torchvision。

## 安装

本项目固定使用已有的 D 盘环境，不直接调用系统 `python`：

```powershell
& 'D:\anaconda3\envs\bishe\python.exe' -m pip install -r requirements.txt
```

如需进入交互式环境，可执行：

```powershell
conda activate D:\anaconda3\envs\bishe
$env:PYTHONNOUSERSITE = '1'
```

`PYTHONNOUSERSITE=1` 用来避免误加载 C 盘用户目录中的 Python 包。

## 运行

先运行最轻量的 Spiral 实验：

```powershell
& 'D:\anaconda3\envs\bishe\python.exe' -m src.spiral --epochs 300 --output-dir outputs/spiral
```

运行 MNIST 主实验（第一次会自动下载数据）：

```powershell
& 'D:\anaconda3\envs\bishe\python.exe' -m src.mnist_experiment --model both --epochs 5 --batch-size 64 --output-dir outputs/mnist
```

比较 Adjoint 与 tolerance：

```powershell
& 'D:\anaconda3\envs\bishe\python.exe' -m src.mnist_experiment --model odenet --adjoint --tolerances 1e-3 1e-5 1e-7 --epochs 3 --batch-size 64 --output-dir outputs/tolerance
```

快速冒烟测试：

```powershell
& 'D:\anaconda3\envs\bishe\python.exe' -m src.spiral --epochs 2 --n-points 64 --output-dir outputs/smoke
& 'D:\anaconda3\envs\bishe\python.exe' -m src.mnist_experiment --model odenet --epochs 1 --max-train-batches 2 --max-test-batches 1 --output-dir outputs/smoke-mnist
```

结果会保存为 `metrics.csv`、`config.json` 和 PNG 图。完整报告建议至少报告 3 个随机种子的均值和标准差。

## 与论文的对应关系

- `src/spiral.py` 对应连续动力系统拟合示例；
- `src/models.py` 的 `ODENet` 对应连续深度分类模型；
- `src/models.py` 的 `odeint_adjoint` 分支对应伴随灵敏度方法；
- `src/mnist_experiment.py` 的 tolerance sweep 对应精度与计算量权衡。

这是一套可在个人电脑上完成的代表性复现，不声称重新完成论文所需的大规模预训练或逐数值复刻所有表格。
