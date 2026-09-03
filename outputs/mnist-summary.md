# MNIST 主实验结果

本次结果使用 RTX 3050 Laptop GPU 4GB、PyTorch 2.2.1+cu118、FP32、随机种子 0。

| 模型 | Epoch | 测试准确率 | 参数量 | 训练时间/epoch | 平均 NFE |
|---|---:|---:|---:|---:|---:|
| ResNet baseline | 5 | 93.02% | 19,338 | 约 7 秒 | 0 |
| ODENet | 3 | 96.48% | 19,274 | 约 162–204 秒 | 45.6–57.1 |

ODENet 和 ResNet 的 epoch 数不同，因此该表用于记录本次运行结果，不能作为严格的同训练步数对比。ODENet 使用 Dopri5 自适应求解器，`rtol=1e-3`、`atol=1e-5`；其较长训练时间主要来自 ODE 函数的多次调用。

详细逐 epoch 指标见同目录下的 `mnist-resnet/metrics.csv` 和 `mnist-odenet/metrics.csv`。
