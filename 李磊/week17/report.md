# GRPO 算术题强化学习复现报告

> 本地运行结果（LoRA 模式）。模型：Qwen2-0.5B-Instruct，任务：6 个难度等级的算术题，用 GRPO（Group Relative Policy Optimization）做可验证奖励强化学习。

## 1. 运行环境

| 项目 | 值 |
|------|----|
| Conda 环境 | `bd_study`（Python 3.13.11） |
| 显卡 | RTX 4050 Laptop 6GB |
| torch | 2.7.0+cu126 |
| transformers | 5.1.0 |
| trl | 0.21.0 |
| peft | 0.18.1 |
| accelerate | 1.12.0 |
| datasets | 5.0.0 |
| matplotlib | 3.10.8 |
| 基座模型 | `models/Qwen2-0.5B-Instruct`（ModelScope 下载，权重 988MB） |

> 说明：项目原文档在 8GB 显存卡上验证，全量微调峰值 6.07GB。本机 6GB，因此只运行 LoRA 方案，峰值显存 3.09GB，与文档记录一致。

## 2. 复现命令

```powershell
# 1. 基线评测：每难度 50 题，greedy + 8 次采样，seed=42
python src/probe_baseline.py --seed 42 --out outputs_my/baseline_probe.json

# 2. LoRA GRPO 训练：200 步
python src/train_grpo.py --lora

# 3. 训练后复测：同一套题（seed=42），加载 LoRA adapter
python src/probe_baseline.py --model outputs_my/grpo_lora_ckpt --out outputs_my/post_train_probe_lora.json --seed 42

# 4. 生成对比表与训练曲线
python src/compare_results.py
```

## 3. 训练配置

| 参数 | 值 |
|------|----|
| 训练步数 | 200 |
| 训练集 | 1000 题（L2 25% / L3 50% / L5 25%） |
| 每组采样数 `num_generations` | 8 |
| 温度 `temperature` | 1.0 |
| KL 系数 `beta` | 0.0（不加载参考模型） |
| 裁剪 `epsilon` | 0.2 |
| 最大补全长度 | 64 |
| 批量 | 8 completions × 梯度累积 4（每步 4 prompt × 8 采样） |
| LoRA | r=16, alpha=32, 目标 q/k/v/o_proj |
| 学习率 | 2e-4（LoRA 模式自动切换） |
| 复合奖励 | 正确 1.0 + 格式 0.2 |
| 训练耗时 | 281.8 秒 |
| GPU 峰值显存 | 3.09 GB |

## 4. 结果对比

同一评估集，seed=42，每难度 50 题。表内数值依次为 `格式率 / greedy 正确率 / pass@8`。

| 难度 | 是否训练集 | 基线 | LoRA 训练后 |
|------|:---:|:---:|:---:|
| L1 个位加法 | 否 | 0.00 / 0.98 / 0.98 | 1.00 / 1.00 / 1.00 |
| L2 两位加减 | 是 | 0.02 / 0.76 / 0.96 | 1.00 / 0.96 / 0.98 |
| L3 三位加减 | 是 | 0.04 / 0.44 / 0.82 | 1.00 / 0.94 / 0.98 |
| L4 表内乘法 | 否 | 0.00 / 0.60 / 0.86 | 1.00 / 1.00 / 1.00 |
| L5 两位×一位 | 是 | 0.00 / 0.20 / 0.66 | 1.00 / 0.98 / 1.00 |
| L6 两位×两位 | 否 | 0.04 / 0.08 / 0.20 | 0.98 / 0.26 / 0.38 |

## 5. 关键结论

1. **格式信号几乎满分且完全泛化**：`<answer>` 格式率从接近 0 学到 1.00，包括没训练过的 L1/L4/L6，说明“按格式输出”是表层行为，RL 很容易学会并迁移。
2. **训练集内正确率大幅提升**：L5 从 20% → 98%，L3 从 44% → 94%，L2 从 76% → 96%。
3. **未训练难度也涨**：L4 从 60% → 100%，证明 RL 学的是通用计算能力，而不是背题。
4. **能力边界无法突破**：L6 只从 8% → 26%。两位×两位乘法需要多步进位，超出 0.5B 模型的能力边界，GRPO 只能把“偶尔蒙对”的概率略微抬高，无法凭空创造多步计算能力。
5. **LoRA 熵坍缩**：训练后期 `entropy` 降到约 0.01，`frac_reward_zero_std` 长期在 0.8 以上，符合文档对 LoRA 在短任务上“收敛快、探索空间耗尽早”的预判。
