# ST13: Seeing the Unseen: Learning Basis Confounder Representations for Robust Traffic Prediction

## 基本信息

- **会议/期刊**: KDD 2025
- **作者**: Jiahao Ji, Wentao Zhang, Jingyuan Wang (Beihang University), Chao Huang (University of Hong Kong)
- **关键词**: 交通预测, 混杂因素, 因果推断, 分布偏移, 自监督学习

## 研究问题与动机

交通预测的核心是建立历史交通数据 $X$ 与未来交通状态 $Y$ 之间的关系。然而 $X \to Y$ 的关系经常受到外部混杂因素（天气、事故、节假日等）的干扰——这些因素同时影响 $X$ 和 $Y$。

**现有方法的局限性**：
- 前门/后门调整方法依赖预定义的离散混杂因素值，无法处理连续或未定义的混杂因素
- 现实场景中混杂因素往往不可完全观测，难以穷举

**核心思想**：不直接枚举天气、节假日等具体混杂因素，而是学习一个"基础混杂因素库"（Base Confounder Bank），任意混杂因素可表示为基础向量的线性组合。

## 核心贡献

1. 提出 STEVE（Spatial-Temporal sElf-superVised confoundEr learning）模型
2. 设计基础混杂因素库机制，用线性组合表示任意连续/未定义混杂因素
3. 引入自监督辅助任务（空间位置分类、时间索引识别、交通负荷预测）增强混杂因素表示
4. 通过对抗解耦（GRL）+ 互信息最小化（CLUB）实现混杂因素与不变表示的分离

## 方法概述

### 整体架构

```
历史交通流量 X_t + 交通图 A
        ↓
  双路时空编码器 TSRL
        ↓
  Z_t (变化表示)          H_t (不变表示)
        ↓                      ↓
  Confounder Bank         GRL + 对抗辅助任务
  + Cross-Attention       + 互信息最小化
        ↓                      ↓
  C_t (混杂因素表示)     H_t (去混杂表示)
        ↓                      ↓
  混杂相关预测 Ŷ_c        混杂无关预测 Ŷ_i
        ↓                      ↓
     异质性感知融合 → 最终预测 Ŷ_{t+1}
```

### 关键模块

#### 1. 基础混杂因素库（Base Confounder Bank）

- 维护可学习的基础向量库 $B = \{b^{(1)}, ..., b^{(K)}\}$
- 用动量更新机制维护：$B_{new} = \gamma B_{old} + (1-\gamma) \tilde{B}$
- 通过 Cross-Attention 从库中加权聚合：$C_t = \sum_{k=1}^{K} \alpha^{(k)} b^{(k)}$

#### 2. 自监督辅助任务（COSSL）

| 辅助任务 | 目标 | 标签来源 |
| :--- | :--- | :--- |
| 空间位置分类 | 让 $C$ 感知区域环境差异 | 节点 ID（one-hot） |
| 时间索引识别 | 让 $C$ 感知时间分布变化 | 48 类时间槽（区分工作日/节假日）|
| 交通负荷预测 | 让 $C$ 包含交通语义 | 历史最大流量近似负载容量 |

#### 3. 混杂因素解耦

**对抗解耦**：对不变表示 $H$ 使用 Gradient Reversal Layer (GRL)，使 $H$ 无法预测时间/空间/负荷等混杂因素。前向传播时 $GRL(H) = H$，反向传播时梯度反转 $\frac{\partial GRL(H)}{\partial H} = -\lambda I$。

**互信息最小化**：使用 CLUB（Contrastive Log-ratio Upper Bound）估计并最小化 $I(H; C)$，从统计依赖角度降低耦合。采用两阶段优化——先训练 MI 估计器（detach），再用 MI 上界约束主模型。

#### 4. 异质性感知融合

考虑混杂因素对不同区域/特征的影响不同：
$$\Lambda = \text{Sigmoid}(C_t W_c)$$
$$\hat{Y}_{t+1} = \text{Fusion}(\hat{Y}^{(c)}_{t+1}, \hat{Y}^{(i)}_{t+1}, \Lambda)$$

### 总损失函数

$$\mathcal{L} = \lambda_p \mathcal{L}_{pred} + \lambda_c \mathcal{L}_{C}^{conf} + \lambda_h \mathcal{L}_{H}^{adv} + 0.1 \mathcal{L}_{MI}$$

- $\mathcal{L}_{pred}$：预测损失
- $\mathcal{L}_{C}^{conf}$：混杂表示正向监督（variant_loss）
- $\mathcal{L}_{H}^{adv}$：不变表示对抗监督（invariant_loss + GRL）
- $\mathcal{L}_{MI}$：互信息最小化（CLUB）

## 实验

### 数据集

| 数据集 | 类型 | 时间粒度 | 时间范围 |
| :--- | :--- | :--- | :--- |
| NYCTaxi | 出租车订单 | 30 min | 2015.01.01 - 2015.03.01 |
| NYCBike1 | 自行车租赁 | 60 min | 2014.04.01 - 2014.09.30 |
| NYCBike2 | 自行车租赁 | 30 min | 2016.07.01 - 2016.08.29 |
| BJTaxi | 出租车订单 | 30 min | 北京 |

训练/验证/测试划分比例 7:1:2。

### 分布偏移评估设计

- **时间分布偏移 (TDS)**：将训练集中工作日:节假日 ≈ 5:2 的比例分别偏移至 1:0 和 0:1
- **空间分布偏移 (SDS)**：用 k-means 聚类交通实体，不同簇代表不同流量水平区域

### 基线方法（13 种）

1. **GNN 时空预测**：STGCN, GMAN, ASTGNN, HimNet
2. **解耦时空方法**：COST, ST-Norm, STWA, SCNN
3. **分布偏移方法**：AdaRNN, CIGA, STNSCM, CauSTG, CaST

### 主要结果

- STEVE 在四个数据集上的所有任务中均优于所有基线方法
- 在时间分布偏移（尤其是节假日场景）和空间分布偏移场景中表现突出
- 消融实验验证了各模块的有效性：混杂因素库 > 自监督任务 > GRL > 互信息最小化

## 与本项目的关联

**高度相关**。本项目的核心目标正是异常客流预测（节假日/极端天气），而 STEVE 的核心贡献就是处理这类场景下的分布偏移问题：

1. **混杂因素表示方法可借鉴**：用基础向量库表示未定义混杂因素的思路，可扩展到极端天气等场景
2. **解耦框架可复用**：GRL + CLUB 的解耦范式可作为基础架构
3. **评估方案可参考**：TDS/SDS 的分布偏移评估设计可用于本项目的异常场景评估
4. **数据集重叠**：使用了 NYCBike/NYCTaxi 数据集，与本项目的数据基础一致

## 关键启发

1. 不必穷举所有外部因素（天气/节假日/事故），可用基础向量库隐式表示
2. 自监督信号（时间/空间/负荷）可有效引导混杂因素表示的学习
3. 对抗解耦 + 互信息最小化是实现表示分离的有效组合
4. 异常场景（节假日/极端天气）的评估应关注分布偏移鲁棒性
