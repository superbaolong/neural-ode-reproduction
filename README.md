# Neural ODE 复现实验

本项目面向 RTX 3050 4GB（也支持 CPU），复现 Neural Ordinary Differential Equations 的核心现象：

1. 二维 Spiral 动力系统拟合；
2. MNIST 上 ODENet 与 ResNet 对比；
3. 直接反向传播与 Adjoint Method 对比；
4. ODE 求解器 tolerance 对 NFE、速度和准确率的影响。

代码默认使用 FP32，避免低显存设备上 ODE 数值不稳定。MNIST 使用原始 IDX 文件加载器，不依赖 torchvision。

完整中文报告草稿见 [`report/neural_ode_report.md`](report/neural_ode_report.md)。

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
& 'D:\anaconda3\envs\bishe\python.exe' -m src.mnist_experiment --model both --epochs 30 --batch-size 64 --output-dir outputs/mnist
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

## 已完成结果

- Spiral：300 epochs，最终 MSE 约 `9.04e-4`，见 [`outputs/spiral-baseline`](outputs/spiral-baseline)。
- MNIST 公平实验（seed 0）：相同设置训练 5 epochs，ODENet 最终测试准确率 `98.33%`，ResNet 最终为 `87.51%`、最佳为 `91.01%`，见 [`outputs/mnist-fair-seed0/summary.md`](outputs/mnist-fair-seed0/summary.md)。
- MNIST 公平实验（seed 1、2）：ODENet 最佳测试准确率分别为 `97.56%`、`98.10%`，ResNet 分别为 `93.35%`、`92.69%`；三个 seed 的均值和标准差见 [`outputs/mnist-fair-summary.md`](outputs/mnist-fair-summary.md)。
- MNIST 云端 30-epoch 补充实验（seed 0）：ODENet 最终测试准确率 `99.27%`，ResNet 最终为 `97.54%`，说明 5 个 epoch 时两种模型尤其是 ResNet 尚未充分收敛，见 [`outputs/mnist-cloud-30/summary.md`](outputs/mnist-cloud-30/summary.md)。
- Direct/Adjoint 短对照：Adjoint 峰值显存约 `139 MB`，Direct 约 `397 MB`，见 [`outputs/adjoint-summary.md`](outputs/adjoint-summary.md)。
- 容差短实验：`rtol=1e-3/1e-5/1e-7` 的 NFE 约为 `38.0/38.9/39.2`，见 [`outputs/tolerance-summary.md`](outputs/tolerance-summary.md)。

