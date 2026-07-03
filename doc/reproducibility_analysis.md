# 训练可复现性分析

> 日期：2026-07-02
> 状态：分析完成，暂未实施。后续修改时参考本文档。

---

## 问题

参数敏感性实验中，Step 2（K=16, α=0.05, γ=0.1）和 Step 3（γ=0.1, α=0.05, K=16）参数完全相同，但两次独立训练结果不同（MAE 差异 1-5%）。说明训练过程存在非确定性。

## 非确定性来源

### 已设置的（3 项）

| 措施 | 位置 | 状态 |
|------|------|------|
| `np.random.seed(42)` | `train/trainer.py` → `set_seed()` | ✅ |
| `torch.manual_seed(42)` | 同上 | ✅ |
| `torch.cuda.manual_seed_all(42)` | 同上 | ✅ |

### 未设置但影响可复现性的（6 项）

| # | 来源 | 影响机制 | 严重度 | 修复 |
|---|------|---------|--------|------|
| 1 | **PYTHONHASHSEED** | Python hash 随机化 → dict 迭代顺序不同 → `model.parameters()` 初始化顺序不同 → 随机数消耗序列偏移 → VariationalLSTM 的 eps 采样序列偏移 → 训练轨迹分叉。**这是 Step2/Step3 结果不同的主因。** | ⛔ 高 | `os.environ['PYTHONHASHSEED'] = '42'` |
| 2 | **cuDNN 非确定性算法** | cuDNN 默认使用非确定性算法（如 atomicAdd 在 backward 中） | ⚠️ 中 | `torch.backends.cudnn.deterministic = True` |
| 3 | **cuDNN benchmark** | 自动选择最快算法，不同运行可能选不同算法 | ⚠️ 中 | `torch.backends.cudnn.benchmark = False` |
| 4 | **TF32** | Ampere GPU 上 matmul/cuDNN 使用 TF32（低精度），引入微小差异 | 🔵 低 | `torch.backends.cuda.matmul.allow_tf32 = False` + `torch.backends.cudnn.allow_tf32 = False` |
| 5 | **torch 确定性算法** | 强制所有操作用确定性实现（部分操作无确定性版本会报错） | ⚠️ 中 | `torch.use_deterministic_algorithms(True, warn_only=True)` |
| 6 | **Python random 模块** | 可能被间接调用 | 🔵 低 | `import random; random.seed(42)` |

## 根因分析

Step 2 和 Step 3 结果不同的最大原因是 **#1（PYTHONHASHSEED）**：

1. 两次 bash 调用是独立 Python 进程
2. Python 默认每次进程随机化 hash seed
3. `model.parameters()` 的迭代顺序取决于 dict 的 hash 顺序
4. 参数初始化时 `nn.Parameter` 按不同顺序创建 → 消耗不同数量的 torch 随机数
5. VariationalLSTM 的 `eps = torch.randn_like(std)` 在训练中持续采样，RNG 状态已偏移
6. 偏移被 MIC 两阶段对抗动态放大 → 最终结果差异 1-5%

## 修复方案

在 `train/trainer.py` 的 `set_seed()` 函数中添加以下设置：

```python
def set_seed(seed: int):
    # Python 级别
    os.environ['PYTHONHASHSEED'] = str(seed)
    import random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # 强制确定性算法（warn_only 避免无确定性实现的操作报错）
    torch.use_deterministic_algorithms(True, warn_only=True)
```

同时在 bash 脚本中添加：
```bash
export PYTHONHASHSEED=42
```

## 潜在代价

| 代价 | 说明 |
|------|------|
| 训练速度 | 降低约 5-15%（cuDNN 不再自动选最快算法 + TF32 关闭） |
| 兼容性 | `use_deterministic_algorithms` 可能在极少数操作上报 warning（`warn_only=True` 不会报错，退回非确定性） |
| CPU 训练 | 不受影响（只有 GPU 相关设置有代价） |

## 修复后的预期效果

| 场景 | 修复后是否完全可复现 |
|------|:---:|
| 同一参数 + 同一机器 + 同一 GPU | ✅ 完全相同 |
| 同一参数 + 不同机器/不同 GPU 架构 | ⚠️ 可能有微小差异（浮点累加顺序不同，无法完全消除） |
| 同一参数 + CPU | ✅ 完全相同 |

## 对消融实验的影响

修复后，消融实验的 control（完整 STCross）在多次运行中结果完全一致，与 A1-A4 变体的对比更公平。但修复前的结果（包括已完成的参数敏感性实验和主实验）不会改变——它们已经跑完了。

**建议**：在跑消融实验前实施修复，确保消融实验本身可复现。已完成的敏感性实验和主实验不需要重跑。
