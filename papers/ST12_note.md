# ST12: CAMEF: Causal-Augmented Multi-Modality Event-Driven Financial Forecasting by Integrating Time Series Patterns and Salient Macroeconomic Announcements

## 基本信息
- **会议/期刊**: KDD '25 (31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining)
- **年份**: 2025
- **作者**: Yang Zhang, Wenbo Yang, Jun Wang, Qiang Ma, Jie Xiong（西南财经大学 & 京都工艺纤维大学）
- **关键词**: Multimodal learning, Causal Learning, Financial dataset, Time-series Forecasting, Counterfactual Augmentation, LLM

## 研究问题与动机

**核心问题**：如何准确预测显著宏观经济事件（如货币政策决议、就业报告等）对金融市场的影响？

**现有方法的三大局限性**：

1. **数据局限（Data Limitation）**：现有方法大多聚焦于单一事件类型（如FOMC会议），忽略了其他关键宏观经济事件（失业救济申请、CPI、PPI、GDP等）；且多使用日频数据，无法满足实时交易中高频数据的需求。
2. **模态局限（Modality Limitation）**：多数研究依赖单一模态——要么纯文本分析（情感分类），要么纯时间序列建模（回归预测），未能整合两种模态的互补优势。已有多模态方法缺乏有效的特征融合机制和因果学习。
3. **因果性局限（Causality Limitation）**：现有方法未纳入因果推理框架，忽略了"事件→市场反应"之间的因果关系，导致模型无法充分捕捉金融市场行为的驱动因素，限制了预测鲁棒性。

**动机示例**：2020年3月16日FOMC紧急降息至0-0.25%，S&P 500、NASDAQ、道琼斯单日暴跌超10%，为1987年以来最大单日跌幅。这类事件与市场波动之间存在明确的因果关系。

## 核心贡献

1. **新颖数据集**：构建了首个整合政策文本、高频交易数据和因果增强内容的开源合成数据集。包含6类宏观经济事件脚本（2008年至2024年4月），以及5个关键美国金融资产（SPX、INDU、NDX、USGG1M、USGG5YR）的5分钟级高频交易数据。还通过LLM生成反事实事件脚本用于因果学习。
2. **新颖多模态模型CAMEF**：提出整合时间序列与文本特征的多模态框架，包含专门设计的多特征融合网络、时间序列解码机制和因果学习策略。
3. **反事实生成与因果学习**：提出基于LLM的反事实数据增强策略，通过修改关键数值和情感相关短语生成不同情感水平的反事实事件脚本，同时保持原始格式和中性内容不变。结合Triplet Loss进行因果对比学习。

## 方法概述

CAMEF架构由四个核心组件构成：

```
┌─────────────────┐    ┌──────────────────┐
│ Textual Encoder  │    │ Time-Series      │
│ (RoBERTa +      │    │ Encoder          │
│  Projection)     │    │ (MOMENT +        │
│                  │    │  Multi-Residual) │
└────────┬────────┘    └────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌─────────────────────┐
         │ Feature Fusion      │
         │ (Concat + FFN)      │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │ GPT-2 Decoder       │
         │ (Auto-regressive)   │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │ Post-Regressor      │
         │ (Linear + GELU +    │
         │  Dropout)           │
         └──────────┬──────────┘
                    ▼
              Predictions Ŷ
```

训练目标为两部分之和：$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Time}} + \mathcal{L}_{\text{Causal-TL}}$

## 关键技术细节

### 1. 文本模态编码器（CAMEF_Textual）
- 使用RoBERTa编码事件脚本 $E_i = \{w_1, w_2, \ldots, w_m\}$，得到上下文token嵌入 $\{h_1, h_2, \ldots, h_m\}$，其中 $h_j \in \mathbb{R}^{1 \times 768}$
- 每个嵌入经过三层线性投影网络（含GELU激活）：
  $$e_j = W^{(3)} \cdot \text{GELU}(W^{(2)} \cdot \text{GELU}(W^{(1)} h_j + b^{(1)}) + b^{(2)}) + b^{(3)}$$
  其中 $W^{(1)} \in \mathbb{R}^{768 \times 1024}$, $W^{(2)} \in \mathbb{R}^{1024 \times 1024}$, $W^{(3)} \in \mathbb{R}^{1024 \times 768}$
- 最终文本编码向量为所有变换后嵌入的平均值：$\mathbf{E}_i = \frac{1}{m}\sum_{j=1}^{m} e_j$

### 2. 时间序列模态编码器（CAMEF_Time-Series）
- 使用预训练的MOMENT模型编码时间序列段：$\mathbf{X}_i = \text{MOMENT}(\{X_1, X_2, \ldots, X_n\})$，$\mathbf{X}_i \in \mathbb{R}^d$
- 引入多残差投影层增强表示：
  $$Z_i = \mathbf{X}_i + f_{\text{residual}}(\mathbf{X}_i)$$
  其中 $f_{\text{residual}}$ 由多层线性层+GELU组成，$W_1, W_2 \in \mathbb{R}^{1024 \times 1024}$, $W_3 \in \mathbb{R}^{1024 \times 768}$

### 3. 特征融合与解码
- 拼接两种模态编码：$\mathbf{E}_{\text{combined}} = \text{Concat}(\mathbf{E}_i, Z_i)$
- 通过两层前馈网络融合：$\mathbf{E}_{\text{fused}} = W^{(f2)} \cdot \text{GELU}(W^{(f1)} \cdot \mathbf{E}_{\text{combined}} + b^{(f1)}) + b^{(f2)}$
- 使用GPT-2（12层）作为自回归解码器：$H^{(l)} = f^{(l)}_{\text{GPT2\_layer}}(H^{(l-1)})$，其中 $H^{(0)} = \mathbf{E}_{\text{fused}}$
- 输出经LayerNorm后送入Post-Regressor（4层线性+GELU+Dropout(0.1)）生成预测 $\hat{Y}$

### 4. 时间序列预测损失
$$\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(\hat{Y}_i - Y_i)^2, \quad \mathcal{L}_{\text{MAE}} = \frac{1}{n}\sum_{i=1}^{n}|\hat{Y}_i - Y_i|$$
$$\mathcal{L}_{\text{Time}} = \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{MAE}}$$

### 5. 反事实事件生成（LLM-based）
- 形式化定义：给定事件脚本 $E_i = E^{\text{neutral}}_i \cup E^{\text{sentiment}}_i$，生成反事实版本：
  $$E'_i = E^{\text{neutral}}_i \cup f_{\text{LLM}}(E^{\text{sentiment}}_i | S'_i)$$
- 使用LLaMA-3 8B模型，三步提示策略：
  1. **摘要提示**：将长事件脚本压缩为简洁摘要，保留关键数值变量
  2. **情感分析提示**：对原始脚本打分（1-10分），提供基准
  3. **反事实生成提示**：修改情感相关短语和数值，生成不同情感水平的反事实版本，保持中性内容和格式不变

### 6. 多样化反事实采样机制
两类反事实事件：
- **同类型采样（Identical Type Sampling）**：同类型事件的反事实版本（修改情感和数值），默认10个
- **多样类型采样（Diverse Type Sampling）**：用日期最近的5种其他类型事件替换

### 7. 因果学习目标（Triplet Loss）
- 文本编码器分别编码真实事件 $E_{GT}$ 和反事实事件 $E_{CF}$
- 时间序列编码器编码历史时间序列段得到嵌入 $T$
- Triplet Loss确保真实事件嵌入比反事实事件嵌入更接近时间序列嵌入：
  $$\mathcal{L}_{\text{Causal-TL}} = \max(0, \ d(P_{GT}, T) - d(P_{CF_i}, T) + \alpha)$$
  其中margin $\alpha = 1.0$

### 8. 两阶段训练策略
- **阶段一**：预训练MOMENT模型学习时间序列表示
- **阶段二**：训练完整CAMEF模型。冻结MOMENT全部参数、RoBERTa（除最后一层）、GPT-2 token嵌入层；其余组件可训练
- 各组件采用差异化学习率（MOMENT: $1\times10^{-6}$, RoBERTa: $5\times10^{-7}$, GPT2: $1\times10^{-5}$等）

## 实验

### 数据集
- **宏观事件数据**：6类事件（FOMC会议纪要、失业救济申请、就业情况报告、GDP先行报告、CPI报告、PPI报告），覆盖2008-2024年，共计约2,569个事实事件 + 22,660个反事实事件
- **时间序列数据**：5个金融资产（S&P500、道琼斯工业指数、NASDAQ、美国1个月国债、美国5年期国债），5分钟级高频数据，数据点从75万到33万不等
- **划分比例**：训练集:验证集:测试集 = 6:2:2

### 基线方法
- **单模态基线**：ARIMA, DLinear, AutoFormer, FEDformer, iTransformer, PatchTST
- **多模态基线**：TEST (ICLR 2024), GPT4MTS (AAAI 2024)

### 主要结果
- CAMEF在30个实验设置中**24个排名第一**，其余6个排名第二
- 相比最佳单模态模型iTransformer，CAMEF平均MSE降低**62.55%**
- 相比最佳多模态模型TEST，CAMEF平均MSE降低**33.55%**
- 在股票市场指数（SPX、INDU、NDX）上全面最优（除INDU短期预测外）
- 在1个月国债（USGG1M）上全面最优；5年期国债（USGG5YR）排第二（长期国债可能对事件驱动因素敏感度较低）

### 消融实验关键发现
1. **全模型最优**：移除任何组件均导致性能下降，验证了多模态融合的必要性
2. **因果学习的增量价值**：移除因果学习后性能下降，确认其在捕获因果关系中的作用
3. **GPT-2解码器和特征融合的关键作用**：移除后某些资产（如INDU、USGG1M）性能急剧恶化
4. **Post-Regressor不可或缺**：移除后平均MSE大幅上升

### 事件类型分析（S&P500）
- **FOMC会议纪要**预测效果最好（MSE/MAE最低），确认其最关键的市场影响力
- **失业救济申请**和**就业情况报告**也有较好预测效果
- **CPI和PPI报告**预测能力较弱，PPI误差最高

### 超参数敏感性
- 反事实样本数量 $\alpha$：增加到10左右性能饱和
- Post-Regressor深度 $k$：$k=4$ 后收益递减

## 与本项目的关联

本项目关注**节假日/极端天气场景下的异常客流预测**，与CAMEF有以下关联点和可借鉴之处：

1. **事件驱动预测范式的高度相似性**：CAMEF解决的是"宏观事件→金融市场反应"的因果预测问题，本项目解决的是"节假日/极端天气事件→客流异常"的预测问题。两者本质上都是**外部事件驱动的时序异常预测**。CAMEF将事件文本与历史时序对齐的方式（$[X_{i-\tau:i+\tau} \mapsto E_i]$）可直接迁移到本项目的"事件-客流"对齐建模中。

2. **多模态融合思路**：CAMEF融合文本模态（事件描述）和时间序列模态（历史价格）的方法对本项目有直接参考价值。本项目可类似地融合：
   - 文本/结构化信息：节假日类型、天气预报文本、预警等级等
   - 时间序列：历史客流数据、天气数值数据等

3. **因果学习与反事实增强的启发**：CAMEF通过LLM生成反事实事件来增强因果理解的做法非常有启发性。在本项目中，可以：
   - 生成反事实场景（如"如果没有暴雨预警，客流会如何？""如果春节提前一周，客流模式如何变化？"）
   - 通过对比真实与反事实场景，让模型更好地区分"事件导致的异常"与"自然波动"

4. **高频数据的处理经验**：CAMEF使用5分钟级高频金融数据，其时间序列编码器（MOMENT + 多残差层）和解码器（GPT-2自回归）的设计可为本项目处理高频客流数据提供参考。

5. **事件重要性分析的参考**：CAMEF发现FOMC和失业相关事件对市场影响最大。类似地，本项目可通过消融实验分析不同类型事件（春节vs国庆vs台风vs暴雪等）对客流异常的预测贡献度。

## 关键启发

1. **因果关系建模是事件驱动预测的核心**：单纯的相关性学习不够，需要显式建模"事件→结果"的因果链。Triplet Loss + 反事实对比是一种实用的因果学习手段。

2. **LLM作为反事实数据生成器的创新用法**：利用LLM的语言理解和生成能力，系统性地创建"假设性场景"，这在数据稀缺的异常事件预测中尤其有价值。对于本项目中罕见的极端天气事件，同样可以用LLM生成多样化的反事实/增强样本。

3. **预训练模型的组合策略值得借鉴**：CAMEF组合了三个预训练模型（RoBERTa + MOMENT + GPT-2），各自负责一个子任务，并通过选择性冻结实现高效微调。这种"预训练组件拼装+差异化学习率"的策略适合资源有限的场景。

4. **特征融合设计的重要性**：简单的concat + FFN就能有效融合多模态信息，关键在于融合后的解码器设计（GPT-2自回归解码）。这提示我们在设计多模态客流预测模型时，应重视跨模态交互和解码策略。

5. **事件类型差异化处理**：不同事件对目标的驱动力差异很大。在实际系统中，应对不同类型的事件（节假日vs天气vs突发事件）建立差异化的特征工程和权重机制，而非一视同仁。

6. **高频时序预测的工程实践**：预测窗口设为35/70/140步（对应半天/一天/两天），这种多尺度评估方式值得在本项目中采用，以验证模型在不同预测horizon下的表现。
