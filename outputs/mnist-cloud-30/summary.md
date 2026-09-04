# MNIST 云端 30 epoch 补充实验

本实验在 AutoDL 的 NVIDIA RTX 5090 32GB 实例上运行，用于检查 5 个 epoch 后模型是否已经收敛。ODENet 与参数量相近的 ResNet 使用同一数据集、batch size、优化器、学习率和随机种子。

运行命令：

```bash
python -u -m src.mnist_experiment \
  --model both --epochs 30 --batch-size 64 --seed 0 \
  --output-dir outputs/s0
```

统一设置：Adam，学习率 `1e-3`，batch size `64`，FP32；ODENet 使用 Dopri5，`rtol=1e-3`、`atol=1e-5`。

| 模型 | 参数量 | 第 30 轮训练准确率 | 第 30 轮测试准确率 |
|---|---:|---:|---:|
| ODENet | 19,274 | 99.415% | **99.27%** |
| ResNet | 19,338 | 97.53% | **97.54%** |

30 个 epoch 后，两种模型均明显优于 5-epoch 实验的早期结果。ResNet 的提升尤其明显，说明此前约 92% 的三 seed 结果主要反映训练轮数不足，不能作为其充分收敛后的性能。该实验只有 `seed=0`，因此适合用于判断训练收敛趋势，不替代三随机种子统计。

仓库中的 `final_metrics.csv` 保存了云端第 30 轮终值；云端完整逐 epoch 文件由程序写入 `outputs/s0/metrics.csv`。
