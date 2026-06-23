# ST08: Generalising Traffic Forecasting to Regions Without Traffic Observations

## 基本信息
- **会议/期刊**: AAAI 2026 (The Fortieth AAAI Conference on Artificial Intelligence)
- **年份**: 2026
- **作者**: Xinyu Su, Majid Sarvi, Feng Liu, Egemen Tanin, Jianzhong Qi（墨尔本大学 & 香港科技大学广州）
- **关键词**: 交通预测、无观测区域泛化、物理信息神经网络(PINN)、外部信号融合、空间分组、LWR方程、对比学习
- **代码**: https://github.com/suzy0223/GenCast
- **扩展版**: https://arxiv.org/abs/2508.08947

## 研究问题与动机

### 核心问题
交通传感器部署成本高，导致空间覆盖稀疏且不均匀。现有模型在**有观测数据的区域**表现良好，但无法泛化到**完全没有交通观测数据的连续区域**（unobserved regions）。

### 现有方法的不足
1. **Kriging/外推方法**（如IGNNK、INCREASE、KITS）：主要针对零散的未观测点（scattered unobserved locations），在面对大面积连续无观测区域时泛化能力差。
2. **STSM**（当前SOTA）：通过遮蔽观测区域中与未观测区域"相似"的位置来训练，但相似度仅基于**静态辅助特征**（POI类别、地理坐标），无法捕获交通模式的**动态特性**。
3. **KITS**：在随机位置创建虚拟节点进行训练，隐式假设了已知密度的零散未观测点，引入了结构性先验，限制了向大连续未观测区域的泛化。

### 本文思路
利用两类更具通用性的引导信号来增强泛化能力：
1. **物理原理**（Physics Principles）：编码跨区域通用的内在交通动力学规律；
2. **动态外部信号**（Dynamic External Signals）：如天气数据，具有全域可用性且与交通模式强相关。

## 核心贡献

1. 提出 **GenCast** 模型，专门针对无观测区域的交通预测泛化问题。
2. 设计 **物理原理引导的损失函数** + **连续可微空间嵌入**，使PINN可在离散交通图上应用；以及 **跨域融合模块** 将动态天气信号与交通观测融合。
3. 提出 **空间分组模块（Spatial Grouping Module）**，过滤局部噪声信号，保留区域不变模式以增强泛化。
4. 在多个真实数据集上实验，GenCast一致优于SOTA基线，预测误差最高降低 **3.1%**，$R^2$ 最高提升 **125.6%**。

## 方法概述

### 整体架构
GenCast 基于对比学习骨架（backbone），包含四个核心模块：

```
观测图 Go → 随机遮蔽子图 → 掩码图 Gm_o
                                    ↓
            ┌──────────────────────────────────────┐
            │  1. 时空编码器 (Spatial-Temporal Encoder) │
            │     - 时间嵌入 TE_enc (正弦编码)          │
            │     - 空间嵌入 SE-L (LLM) / SE-H (GeoHash)│
            │     - STE层融合 → H0                      │
            ├──────────────────────────────────────┤
            │  2. 外部信号编码器 (External Signal Encoder)│
            │     - 天气数据按地理位置匹配               │
            │     - Cross-Attention + Gated Fusion → H0_fuse│
            ├──────────────────────────────────────┤
            │  3. ST Model + 空间分组模块               │
            │     - GCN + TCN 捕获时空依赖              │
            │     - 软分组 + 熵正则化                    │
            ├──────────────────────────────────────┤
            │  4. 物理信息模块 (Physics-informed Module) │
            │     - LWR残差 R = ∂X̂/∂t + (2X̂-X_fspd)·∂X̂/∂l│
            │     - Huber Loss 约束                     │
            └──────────────────────────────────────┘
```

### 训练目标
总损失函数由四项组成：

$$\mathcal{L} = \mathcal{L}_{pred} + \lambda \mathcal{L}_{cl} + \mu \mathcal{L}_{spg} + \theta \mathcal{L}_{phy}$$

- $\mathcal{L}_{pred}$：遮蔽位置的预测损失
- $\mathcal{L}_{cl}$：对比学习损失（原始视图 vs 掩码视图的一致性）
- $\mathcal{L}_{spg}$：空间分组熵正则化损失
- $\mathcal{L}_{phy}$：基于LWR方程的物理约束损失

除 $\mathcal{L}_{cl}$ 外，其余损失均在掩码图 $G^m_o$ 上计算，模拟未观测场景以增强泛化。

### 推理过程
1. 为未观测位置生成伪观测（pseudo-observations）；
2. 将 $G = G_o + G_u$（含伪观测）作为输入 $G^m$；
3. 前向传播生成预测，提取未观测位置的预测结果。

## 关键技术细节

### 1. 时空编码器（Spatial and Temporal Encoder）

**时间嵌入**：利用正弦函数编码每日周期位置：

$$TE_{enc} = \left[\sin\left(\frac{2\pi \cdot TE}{T_d}\right), \cos\left(\frac{2\pi \cdot TE}{T_d}\right)\right]$$

其中 $T_d$ 为每天的时间间隔数。

**空间嵌入（两种策略）**：

| 策略 | 名称 | 输入 | 处理方式 | 特点 |
|------|------|------|----------|------|
| SE-L | LLM-based | 地理坐标+POI+道路属性→文本描述 | 冻结LLaMA3-8B-Instruct取最后token嵌入 | 语义丰富但受时效性影响 |
| SE-H | GeoHash-based | 地理坐标→GeoHash字符串 | char-BERT + Transformer Encoder + Mean Pooling | 简单可训练，适应性强 |

> **关键设计意图**：这两种嵌入都是**连续可微的**，使得后续物理损失中的自动微分成为可能——这是将PINN应用于离散交通图的核心创新。

### 2. 外部信号编码器（External Signal Encoder）

- **数据源**：ECMWF ERA5-Land，9km×9km网格，每小时，4个变量（2m温度、地表净太阳辐射、地表径流、总降水量）。
- **匹配方式**：每个交通节点按最近邻匹配天气站点。
- **时间窗口**：$T_w = 12$小时历史天气上下文。
- **融合机制**：Cross-Attention + Gated Fusion：

$$H^0_{fuse} = \text{ReLU}\left(FC\left(z \odot H^0 + (1-z) \odot H_{wx}\right)\right)$$

$$z = \sigma\left(FC_s(H^0) + FC_t(H_{wx})\right)$$

### 3. 空间分组模块（Spatial Grouping Module）

**目的**：过滤个体位置的局部特异性信号（如交通事故引起的异常），保留组级别共享模式以增强泛化。

**机制**：
1. 对每层ST模型的输出 $H^l \in \mathbb{R}^{N_o \times T \times D}$ 做时间平均池化得 $H'^l \in \mathbb{R}^{N_o \times D}$；
2. 将 $D$ 通道分为 $c_g$ 组，reshape为 $(N_o \cdot c_g) \times d'$ 的样本；
3. 学习 $s_g$ 个空间组的中心表示 $C = \text{Softmax}(W_c) Z^l$；
4. 基于欧氏距离的softmax计算软分配权重：

$$W_s = \text{Softmax}(-\text{cdist}(Z^l, C)) \in \mathbb{R}^{(N_o \cdot c_g) \times (s_g \cdot c_g)}$$

5. **熵最小化损失**鼓励尖锐（接近one-hot）的分配：

$$\mathcal{L}_{spg} = \frac{1}{N \cdot c_g} \sum_{i=1}^{N \cdot c_g} \mathcal{H}(W_s[i,:])$$

### 4. 物理信息模块（Physics-informed Module）

**LWR方程**（交通流守恒定律）：

$$\frac{\partial \rho}{\partial t} + \frac{\partial (\rho x)}{\partial l} = 0$$

由于密度 $\rho$ 通常不可观测，利用Greenshields速度-密度关系 $x = x_{fspd}(1 - \rho/\rho_{max})$ 重写为纯速度形式：

$$R = \frac{\partial x}{\partial t} + (2x - x_{fspd})\frac{\partial x}{\partial l}$$

训练时使用模型预测值 $\hat{X}$ 和可微嵌入计算：

$$R = \frac{\partial \hat{X}}{\partial TE_{enc}} + (2\hat{X} - X_{fspd}) \cdot \frac{\partial \hat{X}}{\partial L_{enc}}$$

物理损失采用 **Huber Loss**：$\mathcal{L}_{phy} = \text{Huber}(R, \delta)$

> **自适应阈值策略**：先用RMSE warm-up一个epoch，计算残差的 $\tau$-分位数设定 $\delta$，然后重新初始化模型用Huber Loss训练。

## 实验

### 数据集
| 数据集 | 类型 | 采样间隔 | 说明 |
|--------|------|----------|------|
| PEMS-Bay | 高速公路 | 5min | 旧金山湾区 |
| PEMS07 | 高速公路 | 5min | 加州 |
| PEMS08 | 高速公路 | 5min | 加州 |
| METR-LA | 高速公路 | 5min | 洛杉矶 |
| Melbourne | 城市交通 | 15min | 墨尔本CBD |

- 额外域泛化测试：**NREL太阳能发电数据集**
- 天气数据：ERA5-Land（9km×9km，每小时）
- 路网/区域数据：OpenStreetMap

### 实验设置
- **任务**：过去2小时 → 预测未来2小时（$T = T' = 2h$）
- **空间划分**：4:1:5（训练:验证:测试），确保空间相邻性；每个数据集4种划分取平均
- **时间划分**：前70%训练，后30%测试
- **评估指标**：RMSE, MAE, MAPE, $R^2$

### 基线方法
- **GE-GAN**：传导式Kriging
- **IGNNK**：归纳式Kriging
- **INCREASE**：归纳式Kriging，迭代遮蔽重建
- **STSM**：无观测区域预测SOTA
- **KITS**：增量训练策略的归纳式Kriging

### 主要结果

| 数据集 | 最佳基线RMSE | GenCast RMSE | 误差降低 | $R^2$提升 |
|--------|-------------|--------------|---------|-----------|
| PEMS07 | 8.390 (STSM) | 8.253 (GenCast-L) | 1.64% | 16.57% |
| PEMS08 | 7.925 (STSM) | 7.863 (GenCast-L) | 0.79% | 10.51% |
| PEMS-Bay | 8.773 (STSM) | 8.683 (GenCast-H) | 1.03% | 8.43% |
| METR-LA | 12.952 (STSM) | 12.720 (GenCast-H) | 1.79% | 79.58% |
| Melbourne | 9.175 (STSM) | 9.009 (GenCast-H) | 1.81% | 125.56% |

- 统计显著性检验：paired t-test 和 Wilcoxon signed-rank test，$p \ll 10^{-8}$
- **SE-L vs SE-H**：SE-L在PEMS07/08更优；SE-H在PEMS-Bay/METR-LA/Melbourne更优（因历史数据与当前OSM不匹配、小区域同质性等）
- **消融实验**：所有模块移除均导致性能下降；物理约束对GenCast-L尤为重要；空间分组在各数据集上一致有效
- **未观测比例鲁棒性**：未观测比例从0.2到0.8，GenCast始终最优
- **环形划分**（模拟城市布局）：$R^2$ 提升高达27.5%
- **域泛化**（NREL太阳能数据）：GenCast-w/wx仍优于所有基线

## 与本项目的关联

### 直接相关性 ⭐⭐⭐⭐⭐
本文研究的**无观测区域泛化**问题与我们构建的**异常客流预测系统**高度相关：

1. **节假日/极端天气场景下的数据稀疏问题**：在极端天气或特殊节假日期间，部分区域的正常观测模式被打破，相当于这些区域暂时处于"类无观测"状态。GenCast利用外部天气信号补偿缺失信息的思路可直接借鉴。

2. **天气信号融合**：GenCast使用ERA5天气数据通过Cross-Attention+Gated Fusion与交通数据融合的方法，为我们处理极端天气场景下的客流预测提供了成熟的技术范式。我们的系统中天气是关键的异常触发因素，该融合机制值得直接采用。

3. **物理约束思想**：虽然LWR方程针对车辆交通，但其核心理念——**用领域知识约束模型以避免过拟合到观测数据的局部模式**——完全适用于客流预测。我们可以探索客流守恒、容量约束等类似的物理/领域先验。

4. **空间分组过滤局部噪声**：在异常场景中，个别站点的突发事件（如临时封站、活动散场）会产生不可泛化的局部信号。空间分组模块的思路可用于区分系统性异常与局部扰动。

5. **空间划分评估策略**：4:1:5的空间划分+多种切分方式取平均的评估方法，适合评估我们系统在未见区域的泛化能力。

### 可迁移技术
- 连续可微空间嵌入（使PINN适用于离散图）
- 门控融合机制（Gated Fusion）用于多模态信号整合
- 熵正则化的软分组策略
- Huber Loss自适应阈值策略

## 关键启发

1. **"外部知识补偿缺失观测"是泛化的核心思路**：当目标区域缺乏直接观测时，物理规律和外部动态信号（天气等）可作为桥梁连接已观测与未观测区域。这比单纯依赖静态特征（POI、坐标）更有效。

2. **让离散图结构支持自动微分是关键技术创新**：通过设计连续可微的空间/时间嵌入，使得原本只能在连续空间使用的PINN可以应用于离散交通图。这一思路在其他需要将物理约束引入图神经网络的场景中同样适用。

3. **过滤局部信号比捕获更多信号更重要**：空间分组模块的设计哲学是"少即是多"——通过熵正则化鼓励sharp assignment，主动抑制个体级别的特异性信号，保留组级别的可泛化模式。这对异常检测场景尤为重要，避免模型被偶发事件误导。

4. **动态外部信号 > 静态辅助特征**：论文的实验明确表明，天气等动态信号比POI等静态特征更能帮助泛化。对于我们的异常客流预测系统，应优先接入实时气象、大型活动日程等动态信号源。

5. **LLM生成的空间嵌入并非总是最优**：SE-L在老旧数据集和小范围同质区域表现不如简单的SE-H。提醒我们在实际应用中不要盲目追求复杂方案，需根据数据特性选择。

6. **Huber Loss + 自适应阈值的鲁棒训练策略**：先用简单loss warm-up估计残差分布，再据此设定Huber阈值，这种两阶段策略在处理异常值时非常实用，可直接用于我们系统中应对异常客流数据的训练。
