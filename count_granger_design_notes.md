# Count-Granger 方法设计与实验说明

本文档整理近期关于 Count-Granger 独立实验线的讨论内容，重点说明当前项目中的数据表示、数据流转、Granger 建模方式、VAR 与 Ridge-VAR 的关系、消融实验设计、与参考文献的关系，以及当前数据划分和关键参数设置。

## 1. 项目中的“多变量”含义

Count-Granger 中的“多变量”不是指多个数据集，也不是指 raw log 中的多个字段，而是指固定时间窗口内多个日志模板簇的计数变量。

经过日志模板归一化和无 LLM 语义聚类后，每一个模板簇都对应一个变量。将日志按照固定时间窗口切分后，每个窗口会统计所有模板簇的出现次数。

例如，假设系统中有 5 个模板簇：

```text
C1 = 内存相关日志
C2 = 网络连接日志
C3 = 磁盘 I/O 日志
C4 = 认证失败日志
C5 = 超时日志
```

第 `t` 个时间窗口可以表示为：

```text
x_t = [3, 0, 12, 1, 5]
```

含义是：

```text
C1 在该窗口出现 3 次
C2 在该窗口出现 0 次
C3 在该窗口出现 12 次
C4 在该窗口出现 1 次
C5 在该窗口出现 5 次
```

随着时间推进，得到多变量时间序列：

```text
X ∈ R^(T×K)
```

其中：

- `T` 表示时间窗口数量；
- `K` 表示模板簇数量；
- 每一行表示一个时间窗口；
- 每一列表示一个模板簇的计数序列。

因此，本文中的多变量时间序列可以表述为：

> 多个日志模板簇在固定时间窗口内的出现次数构成的时间序列，每个模板簇对应一个变量。

## 2. 当前数据流转过程

Count-Granger 的完整数据流如下：

```text
raw log 文件
  → 日志消息抽取
  → 模板归一化
  → source + target 共享模板词表
  → 无 LLM 语义聚类
  → 固定时间窗口计数
  → source / target 数据划分
  → 正常窗口筛选
  → Ridge-VAR Granger 建模
  → residual / edge / score 评分
  → 验证集选择评分分量与阈值
  → 测试集异常检测
```

### 2.1 raw log 到 template

原始日志行通常包含标签、时间戳、其他字段和日志消息。预处理阶段会识别日志消息起始位置，并对消息中的动态字段进行归一化。

例如：

```text
connection from 10.1.2.3 failed at port 8080
```

会被归一化为类似：

```text
connection from <IP> failed at port <NUM>
```

归一化的目的是去除 IP、数字、路径、ID 等动态值，保留稳定的日志事件模式。

### 2.2 source 和 target 共享模板空间

当前跨系统任务中，source 和 target 不分别建立独立模板词表，而是共同建立共享模板空间。

这样做是为了保证：

```text
source 的第 k 个特征
和 target 的第 k 个特征
表示同一种或相近的日志语义
```

如果 source 与 target 分别建词表，那么两个系统的第 `k` 维可能表示完全不同的事件，跨系统建模就没有意义。

### 2.3 template 到 semantic cluster

模板会进一步进入无 LLM 语义聚类流程：

```text
template 文本
  → 字符级 n-gram TF-IDF
  → MiniBatchKMeans
  → template cluster id
```

语义聚类的作用包括：

- 降低模板维度；
- 合并语义相近但表面文本不同的模板；
- 降低计数序列稀疏性；
- 提高 Granger 图结构稳定性；
- 改善验证集阈值向测试集迁移时的稳定性。

### 2.4 semantic cluster 到 count time series

日志根据时间戳被放入固定时间窗口。当前默认时间窗口为 60 秒。

对于第 `t` 个窗口和第 `k` 个模板簇：

```text
x_t[k] = 模板簇 k 在时间窗口 t 内的出现次数
```

最终得到：

```text
counts: [num_time_bins, num_clusters]
labels: [num_time_bins]
```

窗口标签由行级标签聚合得到：如果一个时间窗口中至少包含一条异常日志，则该窗口标签为异常；否则为正常。

## 3. VAR、Ridge-VAR 与 Granger 的关系

### 3.1 VAR 是什么

VAR 全称是 Vector AutoRegression，即向量自回归模型。它用多个变量的历史状态预测多个变量的当前状态。

标准形式为：

```text
z_t = A_1 z_(t-1) + A_2 z_(t-2) + ... + A_p z_(t-p) + b + ε_t
```

在当前项目中：

- `z_t` 是第 `t` 个时间窗口的模板簇计数向量；
- `p` 是最大滞后阶数，即 `max_lag`；
- `A_l` 是第 `l` 个滞后步的影响矩阵；
- `ε_t` 是预测误差。

如果配置为：

```yaml
time_bin_seconds: 60
max_lag: 5
```

那么模型等价于：

```text
用过去 5 个 60 秒窗口的日志计数状态，预测当前 60 秒窗口的日志计数状态。
```

### 3.2 VAR 中的方向性

VAR 系数矩阵中的元素：

```text
A_l[j, i]
```

表示：

```text
过去第 l 个时间步中模板簇 i 的变化
对当前模板簇 j 的预测贡献
```

如果模板簇 `i` 的历史项能够有效预测模板簇 `j` 的当前值，就可以形成 Granger 风格的方向关系：

```text
i → j
```

其含义是：

```text
模板簇 i 的过去变化有助于预测模板簇 j 的当前变化。
```

### 3.3 Ridge-VAR 是什么

标准 VAR 可以看作多输出线性回归：

```text
Y = X B + ε
```

其中：

- `X` 是由过去多个窗口拼接而成的历史输入；
- `Y` 是当前窗口的计数状态；
- `B` 是需要学习的系数矩阵。

由于日志模板簇数量较多、变量之间相关性强、计数序列稀疏，普通 VAR 容易出现过拟合和系数不稳定问题。因此当前项目使用 Ridge 正则化：

```text
min ||Y - X B||² + λ ||B||²
```

这就是 Ridge-VAR。

Ridge 正则项会抑制过大的系数，使模型在高维日志计数数据上更加稳定。

### 3.4 Ridge-VAR 和 Granger 的关系

Granger causality 的基本思想是：

```text
如果变量 X 的历史信息能够提高变量 Y 的预测能力，
则认为 X 对 Y 存在 Granger 意义上的定向影响。
```

传统 Granger 检验通常比较两个模型：

```text
不包含 X 历史的模型
包含 X 历史的模型
```

如果加入 `X` 历史后显著降低预测误差，就认为 `X Granger-causes Y`。

当前项目没有对每一对模板簇执行传统统计显著性检验，而是使用 Ridge-VAR 系数近似刻画 Granger 风格的定向时序依赖：

```text
如果 A_l[j, i] 的绝对值较大，
说明模板簇 i 的历史对模板簇 j 的当前值有较大预测贡献，
因此形成 i → j 的定向边。
```

因此，当前项目中的表述应为：

> 基于 Ridge-VAR 的 Granger 风格定向时序依赖建模。

不宜过度表述为严格统计意义上的物理因果发现。

## 4. 当前 Granger 部分如何用于检测

当前 Granger 部分输出两类分数。

### 4.1 residual score

残差分数衡量当前窗口是否难以被正常 VAR 模型预测：

```text
residual_score(t) = ||z_t - z_hat_t||
```

它回答的问题是：

```text
当前整体日志计数状态是否偏离正常预测？
```

### 4.2 edge score

edge score 基于训练得到的 Granger 风格定向边，衡量当前窗口中的定向边贡献是否异常。

它回答的问题是：

```text
正常状态下模板簇之间稳定的定向依赖关系是否被破坏？
```

近期消融结果表明，当前任务中真正有效的是 edge score，而不是 residual score。`edge_only` 与 `full` 结果一致，而 `residual_only` 表现明显较差。这说明异常主要体现在日志模板簇之间的定向依赖结构变化上，而不只是单个计数变量的预测误差增大。

## 5. 当前消融实验为什么是这些

当前消融设计遵循“基线配置 + 单点改动”的原则，用来验证 Count-Granger 方法链条中各个关键模块的贡献。

### 5.1 full

完整模型作为所有对比的基线，包含：

- 语义聚类；
- 特征筛选；
- Ridge-VAR Granger 建模；
- residual / edge / score 候选分数；
- 验证集自动选择评分分量；
- Precision 约束阈值选择；
- source + target 联合训练。

### 5.2 no_cluster

关闭无 LLM 语义聚类，直接对归一化模板计数。该消融用于回答：

```text
语义聚类是否真的改善检测效果和阈值稳定性？
```

### 5.3 residual_only

只使用 VAR 预测残差进行异常检测。该消融用于回答：

```text
单纯的时间序列预测误差能否完成日志异常检测？
```

近期结果表明，该分数单独使用效果很差。

### 5.4 edge_only

只使用 Granger 定向边异常分数。该消融用于回答：

```text
Granger 定向依赖结构本身是否是核心检测信号？
```

近期结果中，`edge_only` 与 `full` 完全一致，说明完整模型实际主要依赖 edge score。

### 5.5 no_score_selection

不进行验证集评分分量选择，强制使用固定组合分数 `score`。该消融用于回答：

```text
验证集自动选择 residual / edge / score 是否必要？
```

近期结果表明，固定组合分数不稳定，验证集选择是必要的。

### 5.6 no_precision_threshold

取消 Precision 约束，只使用普通 F1 阈值。该消融用于回答：

```text
Precision 约束是否实际影响最终阈值？
```

近期结果中，该项与 full 一致，说明当前 `min_precision: 0.85` 没有触发，但并不能说明 Precision 约束无用。

### 5.7 chronological

将验证集和测试集改为时间顺序划分。该消融用于回答：

```text
模型在时间分布漂移下是否仍然稳定？
```

近期结果显示，chronological 下 ROC-AUC 仍高，但 F1 明显下降，说明模型排序能力仍在，但固定阈值受时间漂移影响较大。

### 5.8 target_only

只使用目标域训练，不使用 source 数据。该消融用于回答：

```text
source 域正常行为先验是否有助于目标域异常检测？
```

近期结果显示，target_only 弱于 full，说明 source 信息对当前跨系统检测有帮助。

## 6. 当前项目与参考文献的关系

当前 Count-Granger 主要受到 `Feature Selection for Fault Detection and Prediction based on Event Log Analysis` 的启发。

### 6.1 相似之处

两者的相似点包括：

1. 都将 event log 转换为时间序列特征；
2. 都关注事件日志特征选择；
3. 都利用 Granger 思想分析事件之间的时序关系；
4. 都服务于故障检测、故障预测或异常检测任务。

从思想层面看，当前项目与该文献的相似度较高，大约可以认为是 `60% - 70%`。

### 6.2 不同之处

当前项目与该文献也有明显差异：

1. 当前任务是 BGL、Thunderbird、Spirit 上的跨系统日志异常检测；
2. 当前项目处理的是大规模 raw log，需要模板归一化和共享模板空间；
3. 当前项目加入无 LLM 语义聚类，降低模板稀疏性；
4. 当前项目用 Ridge-VAR 作为可扩展 Granger 近似，而不是传统两两 Granger 检验；
5. 当前最终检测信号主要来自定向边异常分数，而不是仅用 Granger 做特征选择；
6. 当前项目设计了完整的模块消融实验。

因此更准确的论文表述是：

> 受该文献启发，本文将日志模板在固定时间窗口内的出现次数表示为多变量时间序列，并进一步引入基于 Ridge-VAR 的 Granger 风格定向依赖建模，用于跨系统日志异常检测。

不建议写成“复现该方法”，因为当前方法已经在任务、数据表示、模型和评分方式上做了较大改造。

## 7. 当前数据划分方式

当前默认配置为：

```yaml
split:
  source_train_ratio: 0.5
  target_train_ratio: 0.4
  val_ratio: 0.3
  test_ratio: 0.3
  eval_mode: stratified_original
  eval_anomaly_ratio: 0.5  # ?? stratified_random ?????
```

### 7.1 source 的使用方式

source 只用于训练，不参与验证和测试。

流程为：

```text
source raw log
  → count time series
  → 取前 50% 时间窗口
  → source train
```

在训练 Ridge-VAR 时，并不是所有 source train 窗口都会进入模型。当前训练样本需要满足：

```text
当前窗口正常
并且过去 max_lag 个历史窗口也正常
```

如果当前窗口或历史窗口中包含异常标签，则该 lagged training sample 会被跳过。

### 7.2 target 的使用方式

target 是最终评估对象。默认流程为：

```text
target raw log
  → count time series
  → 前 40% 时间窗口作为 target train
  → 剩余 60% 作为 validation / test 候选池
```

其中：

- `target train` 用于和 source train 一起拟合模型，并用于目标域正常分数校准；
- `target val` 用于选择评分分量和阈值；
- `target test` 用于最终报告 Precision、Recall、F1、ROC-AUC 和 PR-AUC。

### 7.3 stratified_random 划分

当前默认 `eval_mode` 是 `stratified_random`，因此 target 的验证集和测试集不是严格按时间连续切分，而是从 target 训练段之后的剩余窗口中按标签分层随机采样。

这样做的好处是：

- validation 和 test 的异常比例更稳定；
- Precision、Recall、F1 更稳定；
- 验证集阈值更容易迁移到测试集。

但它也有局限：

- 不完全等价于真实在线检测；
- 无法充分反映时间分布漂移；
- 因此需要 `chronological` 消融作为补充。

### 7.4 以 BGL → Thunderbird 为例

在近期实验中，预处理后大致得到：

```text
source BGL:        309,198 个 60 秒窗口
 target Thunderbird: 37,225 个 60 秒窗口
```

按照当前默认配置，理论上：

```text
source train ≈ 309,198 × 0.5 = 154,599 个窗口
 target train ≈ 37,225 × 0.4 = 14,890 个窗口
 target val/test 候选池 ≈ 22,335 个窗口
```

最终真正进入 Ridge-VAR 训练的是：

```text
source train 正常 lagged samples
+
target train 正常 lagged samples
```

最终报告指标只来自：

```text
target test
```

## 8. max_lines 参数说明

当前配置中：

```yaml
preprocess:
  max_lines: 10000000
```

该参数表示：

```text
每个数据集最多读取 10,000,000 行原始日志。
```

它作用在 raw log 读取阶段，不是训练样本数，也不是时间窗口数。

例如：

```text
BGL 原始文件约 4,747,963 行，小于 max_lines，因此基本全部读取；
Thunderbird 原始文件可能更大，但当前最多读取前 10,000,000 行。
```

设置 `max_lines` 的主要原因是控制预处理成本，避免超大日志导致：

- 模板提取时间过长；
- 内存占用过高；
- 语义聚类过慢；
- count series 过长；
- 实验迭代不可控。

## 9. 为什么 source_train_ratio 不默认设为 1

虽然 source 不参与目标测试，理论上可以全部用于训练，但当前默认使用：

```yaml
source_train_ratio: 0.5
```

而不是：

```yaml
source_train_ratio: 1.0
```

主要原因如下。

### 9.1 避免 source 后期异常和漂移污染训练

source 数据中也包含异常窗口。即使训练时过滤异常标签，异常附近仍可能存在：

- 异常前兆窗口；
- 异常恢复窗口；
- 标签不完整的异常窗口；
- 系统状态切换窗口；
- 运行负载漂移窗口。

这些窗口可能被标为正常，但统计模式已经偏离稳定正常状态。只使用 source 前段可以降低后期异常和漂移对模型的影响。

### 9.2 保持与 target 前段训练设定一致

target 当前使用前 40% 时间窗口训练。如果 source 使用全部生命周期，而 target 只使用早期生命周期，训练数据来源会变成：

```text
source: 完整时间跨度
target: 早期时间跨度
```

这可能导致 source 的后期状态过多影响模型，使其与 target 训练阶段不匹配。

### 9.3 避免 source 过度主导 target

以 BGL → Thunderbird 为例：

```text
BGL source 总窗口约 309,198
Thunderbird target train 约 14,890
```

如果 source 全部使用，则 source 与 target train 的比例约为：

```text
309,198 : 14,890 ≈ 20.8 : 1
```

即使 source 只使用 50%，比例仍约为：

```text
154,599 : 14,890 ≈ 10.4 : 1
```

这说明 source 本身已经很强。如果使用全部 source，模型可能主要学习 source 的 Granger 结构，而不是 target 的目标域结构。

### 9.4 控制训练时间

source 全量使用会增加：

- 特征筛选成本；
- 冗余相关计算成本；
- lagged 样本构造成本；
- Ridge-VAR 拟合成本；
- edge score 统计成本。

因此，默认使用 50% source 也是一种训练成本和迁移稳定性的折中。

### 9.5 建议作为消融实验

`source_train_ratio: 1.0` 并不是不能使用，而是更适合作为一个单独消融。

建议新增：

```yaml
# configs/count_granger_ablations/source_full.yaml
ablation:
  name: source_full
  description: Use all source count bins for Count-Granger training.

split:
  source_train_ratio: 1.0
```

然后比较：

```text
full:         source_train_ratio = 0.5
source_full:  source_train_ratio = 1.0
target_only:  不使用 source
```

如果结果为：

```text
source_full > full > target_only
```

说明 source 数据越多越好。

如果结果为：

```text
full > source_full > target_only
```

说明 source 有帮助，但过多 source 会引入分布偏移。

如果结果为：

```text
target_only > full
```

说明 source 迁移存在负作用。

## 10. 当前结论总结

当前 Count-Granger 可以概括为：

```text
将 raw log 转换为模板簇计数多变量时间序列，
使用 Ridge-VAR 学习正常状态下模板簇之间的 Granger 风格定向依赖，
并通过定向边结构异常检测目标系统中的异常窗口。
```

当前最重要的实验观察包括：

1. “多变量”指模板簇计数变量，而不是多个数据集或日志字段；
2. Ridge-VAR 是当前 Granger 风格定向依赖建模的核心；
3. 当前方法更准确地说是 Granger 风格的定向时序依赖，而不是严格物理因果发现；
4. edge score 是目前最有效的异常检测信号；
5. residual score 单独使用效果较差；
6. 语义聚类有助于提升检测稳定性；
7. source 信息对 target 检测有帮助，但 source 数据不一定越多越好；
8. `max_lines` 控制 raw log 最大读取行数；
9. `source_train_ratio` 控制 source count series 中参与训练的时间窗口比例；
10. `chronological` 消融暴露了时间漂移问题，后续需要动态阈值或滚动校准。