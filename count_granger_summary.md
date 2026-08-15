# Count-Granger 日志异常检测实验总结

## 1. 方法定位

Count-Granger 是项目中独立于原始 LogGCA 序列预测模型的一条实验路线。它不直接把日志表示为事件 ID 序列，而是先统计固定时间窗口内各类日志模板的出现次数，将原始日志转换为多变量计数时间序列，然后在计数序列上建模日志模板之间的定向影响关系。

该方法主要参考了 `Feature Selection for Fault Detection and Prediction based on Event Log Analysis` 中“事件日志特征选择 + 时间序列因果分析”的思想，并结合本项目的 BGL、Thunderbird 和 Spirit 数据集进行改造。当前实现不使用 LLM，语义聚类采用字符级 n-gram TF-IDF 和 `MiniBatchKMeans`。

Count-Granger 的基本流程如下：

```text
原始日志
  → 日志模板归一化
  → 共享模板词表
  → 无 LLM 语义聚类
  → 固定时间窗口计数
  → 特征筛选与冗余删除
  → Ridge-VAR Granger 建模
  → 定向边异常评分
  → 验证集阈值选择
  → 日志窗口异常检测
```

## 2. 数据表示

原始日志行通常可以抽象为：

```text
标签 时间戳 其他字段 日志消息
```

对于第 `t` 个时间窗口和第 `k` 个模板簇，定义：

```text
x_t[k] = 模板簇 k 在时间窗口 t 内的出现次数
```

因此，一个数据集被转换为多变量时间序列：

```text
X ∈ R^(T×K)
```

其中：

- `T` 表示时间窗口数量；
- `K` 表示模板簇数量；
- 每一行表示一个时间窗口内的日志状态；
- 每一列表示一个模板簇随时间变化的计数序列。

当前默认使用 60 秒时间窗口。此前对 120 秒和 300 秒窗口的实验表明，较大的窗口会平滑掉短时异常突发，导致检测性能明显下降。因此，当前 Count-Granger 默认优先保留 60 秒粒度。

## 3. 模板与语义聚类

预处理阶段首先在源域和目标域日志上建立共享模板空间，避免两个系统分别生成不可比较的特征维度。模板处理包括：

1. 根据日志格式识别消息起始位置；
2. 对数字、地址、路径、标识符等动态字段进行归一化；
3. 统计模板频率并过滤极低频模板；
4. 使用字符级 n-gram TF-IDF 表示模板文本；
5. 使用 `MiniBatchKMeans` 将语义相近的模板合并为模板簇；
6. 在每个时间窗口中统计模板簇出现次数。

语义聚类的作用不是直接识别异常，而是：

- 减少模板维度；
- 合并语义相近但表面文本略有差异的日志；
- 降低计数序列稀疏性；
- 提高 Granger 图结构和异常分数的稳定性；
- 改善验证集阈值向测试集迁移时的表现。

## 4. Granger 建模

当前实现使用带 Ridge 正则的 VAR 模型作为可扩展的 Granger 近似：

```text
z_t = A_1 z_(t-1) + ... + A_p z_(t-p) + b + ε_t
```

其中：

- `z_t` 是经过 `log1p` 变换和稳健归一化后的模板簇计数向量；
- `p` 由 `max_lag` 指定；
- `A_l[j, i]` 表示模板簇 `i` 在第 `l` 个滞后步对模板簇 `j` 的定向影响；
- 多个滞后步上的系数被聚合为定向影响矩阵；
- Ridge 正则用于缓解模板簇数量较多时的共线性和过拟合问题。

当前模型不是对每一对模板进行独立的传统显著性检验，而是使用 Ridge-VAR 系数构造可扩展的 Granger 风格定向影响图。因此，在论文中更准确的表述是：

> 基于正则化 VAR 系数提取 Granger 风格的定向影响关系。

不宜将当前实现表述为对所有模板对执行了严格的小样本统计显著性检验。

## 5. 特征筛选与图构建

为了避免在数千个模板上直接拟合高维 VAR，当前流程依次执行：

1. 删除总出现次数过低的模板簇；
2. 删除活跃窗口数量过少的模板簇；
3. 删除方差过低的模板簇；
4. 使用秩相关近似删除高度冗余的模板簇；
5. 根据特征重要性保留不超过 `max_features` 个模板簇；
6. 根据 Granger 系数阈值构造定向边。

当前 BGL → Thunderbird 实验中，完整配置最终保留：

```text
特征数量：86
定向边数量：5138
边密度：0.694700
```

目前边密度相对较高，说明 `edge_threshold` 仍然偏宽松。后续可以增加每个目标节点只保留 Top-k 父节点的稀疏化实验，以进一步提升图结构的可解释性和跨时间稳定性。

## 6. 异常评分

模型支持两类异常信号：

### 6.1 预测残差分数

预测残差表示当前计数状态偏离正常 VAR 动力学的程度：

```text
residual_score(t) = ||z_t - z_hat_t||
```

该分数反映的是“当前日志状态是否难以被正常模型预测”。

### 6.2 定向边异常分数

定向边分数表示当前窗口中重要 Granger 影响关系的贡献是否异常。该分数反映的是：

```text
模板之间原本稳定的定向依赖关系是否发生了异常变化
```

### 6.3 当前评分分量选择

实现支持以下评分方式：

- `score`：残差分数和边分数的固定加权组合；
- `residual`：仅使用预测残差；
- `edge`：仅使用定向边分数；
- `validation_best`：在验证集上从上述分量中选择效果最好的分数。

当前完整实验中，验证集自动选择了 `edge` 分量。因此当前最有效的实际检测器是定向边异常检测，而不是残差与边分数的固定相加。

## 7. 最新消融实验

实验任务为：

```text
BGL → Thunderbird
```

使用命令：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python run_count_granger_ablations.py --source_data dataset/BGL/BGL.log --target_data dataset/Thunderbird/Thunderbird.log
```

最新结果如下：

| 消融设置 | Precision | Recall | F1 | ROC-AUC | PR-AUC | 选择的评分分量 | 特征数 | 边数 | 边密度 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 完整模型 `full` | 0.961228 | 0.979693 | 0.970372 | 0.970150 | 0.924058 | `edge` | 86 | 5138 | 0.694700 |
| 去除语义聚类 `no_cluster` | 0.848974 | 0.953348 | 0.898139 | 0.966060 | 0.963907 | `edge` | 75 | 2780 | 0.494222 |
| 仅预测残差 `residual_only` | 0.800000 | 0.008782 | 0.017372 | 0.795334 | 0.690267 | `residual` | 86 | 5138 | 0.694700 |
| 仅定向边分数 `edge_only` | 0.961228 | 0.979693 | 0.970372 | 0.970150 | 0.924058 | `edge` | 86 | 5138 | 0.694700 |
| 不进行评分分量选择 `no_score_selection` | 0.800000 | 0.008782 | 0.017372 | 0.846662 | 0.738859 | `score` | 86 | 5138 | 0.694700 |
| 不使用 Precision 约束 `no_precision_threshold` | 0.961228 | 0.979693 | 0.970372 | 0.970150 | 0.924058 | `edge` | 86 | 5138 | 0.694700 |
| 按时间顺序划分 `chronological` | 0.888514 | 0.396682 | 0.548488 | 0.986218 | 0.846719 | `edge` | 86 | 5138 | 0.694700 |
| 仅使用目标域 `target_only` | 0.835828 | 0.919319 | 0.875588 | 0.901820 | 0.818461 | `edge` | 58 | 3333 | 0.990785 |

## 8. 消融结果分析

### 8.1 定向边分数是当前核心检测机制

完整模型和 `edge_only` 的结果完全一致：

```text
Precision = 0.961228
Recall    = 0.979693
F1        = 0.970372
```

这是因为完整模型在验证集上自动选择了 `edge` 分量。因此，该结果说明当前任务中最有效的检测信号来自日志模板之间的定向依赖结构变化。

相比之下，仅使用预测残差时，F1 下降到 `0.017372`，Recall 下降到 `0.008782`。这表明异常并不总是表现为单个计数变量无法被预测，而更可能表现为多个日志模板之间原有定向关系的变化。

该结果支持以下方法结论：

> Count-Granger 的主要价值不只是将日志转化为计数时间序列，而是从计数时间序列中提取模板之间的定向影响结构，并利用结构异常进行检测。

### 8.2 固定组合分数不稳定

`no_score_selection` 的 F1 只有 `0.017372`，远低于完整模型的 `0.970372`。这说明残差分数和定向边分数不能未经校准直接进行固定加权。

可能原因包括：

- 两类分数的数值尺度不同；
- 残差分数包含较多与异常无关的波动；
- 固定权重无法适应不同数据集和不同时间窗口；
- 残差分数可能破坏定向边分数的排序结构。

因此，当前配置保留验证集评分分量选择：

```yaml
detect:
  score_component: validation_best
```

### 8.3 语义聚类有助于提高阈值稳定性

去除语义聚类后，PR-AUC 反而提高到 `0.963907`，但最终 Precision 和 F1 分别下降到 `0.848974` 和 `0.898139`。

这说明语义聚类的作用不一定是单纯提高全局排序指标，而是改善：

- 计数序列的稳定性；
- 正常窗口分数分布；
- 定向图结构的完整性；
- 阈值从验证集向测试集迁移时的可靠性。

换言之，未聚类模型可能具有较好的局部排序能力，但分数更稀疏、更不稳定，最终阈值下的 Precision 和 F1 较差。当前结果支持保留无 LLM 语义聚类模块。

### 8.4 时间顺序划分揭示了分布漂移问题

Chronological 设置的 ROC-AUC 为 `0.986218`，说明模型仍然具备较好的异常排序能力；但是 Recall 只有 `0.396682`，F1 只有 `0.548488`。

这说明当前模型面临明显的时间分布漂移：

- 验证集和测试集的日志活跃度可能不同；
- 模板计数分布可能随时间变化；
- 固定阈值难以直接迁移到后续时间段；
- 异常类型或异常密度可能发生变化。

因此，当前的分层随机划分更适合评估离线检测能力，而时间顺序划分更适合评估在线部署泛化能力。论文中应同时报告两类结果，不能将随机划分结果直接等同于真实在线性能。

### 8.5 源域信息能够增强跨系统检测

仅使用目标域训练数据时，F1 从完整模型的 `0.970372` 下降到 `0.875588`，ROC-AUC 从 `0.970150` 下降到 `0.901820`。

这说明 BGL 源域提供的正常行为先验有助于 Thunderbird 目标域的 Granger 结构估计。但当前结果不能直接说明源系统对目标系统存在物理因果影响，更准确的解释是：

> 源域日志提供了额外的正常计数模式、模板筛选信息和定向依赖先验，从而提高了目标域模型的稳定性。

后续需要进一步拆分“源域样本增强”和“跨系统图结构迁移”两种作用，避免将数据共享效果误写成跨系统因果传播。

### 8.6 Precision 约束在本次实验中没有触发

`no_precision_threshold` 与完整模型完全一致，说明当前配置中的：

```yaml
detect:
  min_precision: 0.85
```

没有改变最终阈值。原因是验证集上用于最大化 F1 的阈值本身已经满足 Precision 不低于 `0.85`。

因此，本次结果不能说明 Precision 约束无效，只能说明 `0.85` 这一约束在当前数据划分中不够严格。后续可以比较：

```text
min_precision = 0.85
min_precision = 0.90
min_precision = 0.95
```

以观察 Precision、Recall 和 F1 之间的权衡关系。

## 9. 当前方法结论

根据最新消融结果，Count-Granger 当前最合理的方法解释是：

1. 将日志模板转换为固定窗口内的计数时间序列；
2. 使用无 LLM 语义聚类降低模板稀疏性；
3. 使用特征筛选保留与故障状态相关的模板簇；
4. 使用 Ridge-VAR 提取 Granger 风格的定向影响关系；
5. 使用定向边异常分数识别日志状态变化；
6. 在验证集上选择更适合当前数据分布的评分分量和阈值。

当前 BGL → Thunderbird 结果表明，定向边结构比单纯预测残差更适合作为异常检测依据。语义聚类和源域正常数据则主要改善结构稳定性和跨系统泛化能力。

## 10. 当前局限与后续实验

### 10.1 图结构过密

完整模型边密度为 `0.694700`，偏高。后续应增加边稀疏化实验，例如：

- 每个目标节点保留 Top-3 父节点；
- 每个目标节点保留 Top-5 父节点；
- 每个目标节点保留 Top-10 父节点；
- 全局保留 Top-10% 或 Top-20% 的定向边。

### 10.2 残差和边分数需要稳健归一化

当前固定组合分数表现较差，后续应分别使用目标域正常训练窗口估计中位数和 MAD：

```text
normalized_score = (score - median_normal) / MAD_normal
```

然后再组合：

```text
score = α × normalized_residual
      + β × normalized_edge
```

### 10.3 需要处理时间漂移

对于 chronological 和在线场景，应加入：

- 滚动正常状态校准；
- 基于历史正常窗口的动态阈值；
- walk-forward 评估；
- 分时间段重新估计分数中心和尺度。

### 10.4 需要在多个数据集对上重复消融

当前表格是 BGL → Thunderbird 单个数据集对的结果。正式论文中应至少补充：

- BGL → Spirit；
- Thunderbird → BGL；
- Thunderbird → Spirit；
- Spirit → BGL；
- Spirit → Thunderbird。

只有在多个数据集对上重复观察到类似趋势，才能将“定向边分数优于残差分数”“语义聚类提高稳定性”等结论作为一般性结论。

## 11. 关键文件

| 文件 | 作用 |
|---|---|
| `count_series_preprocessing.py` | 将原始日志转换为模板簇计数时间序列 |
| `count_granger_model.py` | 特征筛选、Ridge-VAR 拟合和定向影响评分 |
| `count_granger_main.py` | 单数据集及源域到目标域的训练和评估入口 |
| `configs/count_granger_config.yaml` | Count-Granger 默认配置 |
| `run_count_granger_ablations.py` | 批量运行 Count-Granger 消融实验 |
| `configs/count_granger_ablations/` | 各项消融实验配置 |

## 12. 运行命令

运行单个数据集：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python count_granger_main.py --data dataset/BGL/BGL.log --config configs/count_granger_config.yaml --mode train
```

运行源域到目标域：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python count_granger_main.py --source_data dataset/BGL/BGL.log --target_data dataset/Thunderbird/Thunderbird.log --config configs/count_granger_config.yaml --mode train
```

运行全部消融：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python run_count_granger_ablations.py --source_data dataset/BGL/BGL.log --target_data dataset/Thunderbird/Thunderbird.log
```

只运行部分消融：

```powershell
python run_count_granger_ablations.py --source_data dataset/BGL/BGL.log --target_data dataset/Thunderbird/Thunderbird.log --ablations full residual_only edge_only no_cluster
```

消融结果默认保存到：

```text
results/count_granger_ablations/<时间戳>_<源域>_to_<目标域>/summary.csv
results/count_granger_ablations/<时间戳>_<源域>_to_<目标域>/summary.md
results/count_granger_ablations/<时间戳>_<源域>_to_<目标域>/summary.json
```
## 13. 图稀疏度完整对比

下面汇总当前已经跑完的 `dense / Top-k` 图稀疏度消融。所有结果均来自 `eval_mode: stratified_random`、`eval_anomaly_ratio: 0.5` 的 balanced split，因此这一节只回答 balanced split 下的图稀疏度问题。

| 设置 | edge_selection | top_k | 特征数 | 边数 | 边密度 | Precision | Recall | F1 | ROC-AUC | PR-AUC | 选择分数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | none | N/A | 86 | 5052 | 0.683072 | 0.956032 | 0.978058 | 0.966920 | 0.967231 | 0.917854 | edge |
| top2 | top_k_per_target | 2 | 86 | 170 | 0.022985 | 0.936027 | 0.762479 | 0.840387 | 0.912815 | 0.863693 | edge |
| top3 | top_k_per_target | 3 | 86 | 254 | 0.034343 | 0.916509 | 0.794844 | 0.851351 | 0.901279 | 0.834824 | edge |
| top4 | top_k_per_target | 4 | 86 | 338 | 0.045700 | 0.910985 | 0.791552 | 0.847080 | 0.897514 | 0.825137 | edge |
| top5 | top_k_per_target | 5 | 87 | 426 | 0.056282 | 0.904282 | 0.787713 | 0.841982 | 0.890896 | 0.815291 | edge |
| top10 | top_k_per_target | 10 | 86 | 834 | 0.112764 | 0.885206 | 0.791004 | 0.835458 | 0.886589 | 0.817988 | edge |
| top20 | top_k_per_target | 20 | 86 | 1629 | 0.220254 | 0.875386 | 0.778387 | 0.824042 | 0.889139 | 0.818814 | edge |

从这组结果可以直接得到结论：

- **如果只看 balanced split 的检测性能，dense 最优**，F1 达到 `0.966920`，明显高于任何 Top-k 稀疏版本。
- 在稀疏图里，`Top-3` 是目前最强的折中点，F1 为 `0.851351`，略优于 `Top-4/Top-5/Top-10/Top-20`。
- `Top-2` 进一步提高了 Precision，但 Recall 明显下降，说明过度稀疏会切断一部分有效传播边。
- 因此，**balanced split 下真正最优的是 dense，而不是 Top-3**；如果论文更重视解释性与结构可读性，`Top-3` 是当前最好的稀疏版本。

这也意味着后续论文叙事可以分成两层：

1. **性能上限**：dense Granger 图在 balanced split 下表现最好，说明密集定向依赖中确实包含更多可用于检测的信号。
2. **可解释性折中**：Top-3 以约 `3.43%` 的边密度保留了部分检测能力，是更适合作为稀疏可解释版本的配置。

需要注意的是，本节结论只适用于 balanced split。由于 `eval_anomaly_ratio: 0.5` 会改变测试集类别先验，dense 是否仍然优于 Top-3，还需要在 `stratified_original` 和 `chronological` 评估下继续验证。

## 14. 2026-08-11 校准保护与阈值防退化后的 full 六方向结果

本轮实验只启用了 `full` 消融项，用来验证两项关键修改的整体效果：

1. `target_normal_calibration` 增加正常窗口数量保护；
2. `validation_best` 和阈值选择增加防退化约束，避免出现 `Precision=1.0` 但 `Recall` 极低的伪最优。

### 14.1 六个 source-target 方向的结果

| Source -> Target | 校准策略 | Target 正常窗口数 | 选中分量 | Precision | Recall | F1 | ROC-AUC | PR-AUC | 约束满足 |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| BGL -> Spirit | pooled | 2 | edge | 0.956 | 0.880 | 0.916 | 0.919 | 0.858 | 是 |
| BGL -> Thunderbird | target | 6543 | edge | 0.956 | 0.978 | 0.967 | 0.967 | 0.918 | 是 |
| Spirit -> BGL | target | 122806 | score | 0.558 | 0.923 | 0.695 | 0.630 | 0.610 | 否 |
| Spirit -> Thunderbird | target | 6543 | edge | 0.939 | 0.983 | 0.960 | 0.949 | 0.872 | 是 |
| Thunderbird -> BGL | target | 122806 | residual | 0.561 | 0.929 | 0.699 | 0.593 | 0.582 | 否 |
| Thunderbird -> Spirit | pooled | 2 | edge | 0.652 | 0.950 | 0.773 | 0.819 | 0.777 | 否 |

平均 Precision 约为 `0.770`，Recall 约为 `0.941`，F1 约为 `0.835`。

### 14.2 校准保护的作用

`BGL -> Spirit` 和 `Thunderbird -> Spirit` 的 target 端正常窗口一共只有 `2` 个，如果直接用 target 进行校准，分数尺度会非常不稳定。现在这两个方向都回退到 `pooled` 校准，其中 `BGL -> Spirit` 已从原来的低召回率情况提升为 `F1=0.916`。

### 14.3 阈值防退化的作用

修改前，`Thunderbird -> BGL` 可能出现 `Precision=1.000` 但 `Recall=0.004` 的极端结果。现在加入约束后，该方向会在不能同时满足 Precision 和 Recall 约束时回退到更平衡的 F1 最大解。这使得 `validation_best` 不再以极端高 Precision 作为唯一目标。

### 14.4 评分分量选择结果

本轮中，`edge` 仍然是最重要的定向分量，在 `BGL -> Spirit`、`BGL -> Thunderbird`、`Spirit -> Thunderbird` 和 `Thunderbird -> Spirit` 中都被选为最优分量。仅有两个以 BGL 为 target 的方向选了 `score` 和 `residual`，说明 BGL 作为 target 时，单纯的 Granger 边结构区分度仍然有限，需要结合残差或组合分数。

### 14.5 总体结论

本次修改已经把原来的“阈值影响导致召回率崩溃”问题明显缓解，但现在的重点已经转为部分 target 方向 Precision 不足。这意味着，接下来应该主要通过 `edge_only`、`residual_only`、`combined_score`、`target_only` 和 `no_cluster` 等消融继续定位问题来源。


## 18. 原版主模型不变的弱方向专用方案

目标是**不改变原版主模型的默认效果**，只针对 `Thunderbird → Spirit`、`Thunderbird → BGL`、`Spirit → BGL` 三个弱方向做单独的配置路由和评估分支。

### 18.1 基本原则

- 主模型保持原版 baseline，不把 `target_priority`、`robust_residual`、`edge_consistency` 设成默认主线；
- 强方向继续使用原版主配置，避免原有效果回退；
- 弱方向只在评估策略、阈值选择、图稀疏度上做单独分支；
- 每个方向单独保存 `effective_config.yaml` 和 `summary.json`，便于比较。

### 18.2 推荐的方向级路由

| 方向 | 主体模型 | 先试的调整 | 目的 |
|---|---|---|---|
| Thunderbird → Spirit | 原版主模型 | `validation_best` + `stratified_original` | 让阈值和评分分量随该方向的验证集自动选择，减少分布漂移影响 |
| Thunderbird → BGL | 原版主模型 | `validation_best` + `stratified_original` + dense / top-k 小网格 | 该方向通常更依赖结构稳定性，先从评估和稀疏度找增益，不动主模型 |
| Spirit → BGL | 原版主模型 | `validation_best` + `stratified_original` + dense / top-k 小网格 | 该方向模板空间更大，先验证是否需要更密或更疏的图结构 |

### 18.3 建议的实验顺序

1. **先冻结主模型**：恢复原版默认配置，确认 `BGL → Thunderbird` 不再回退；
2. **再做方向级评估路由**：把三个弱方向切到 `validation_best`，并统一使用 `stratified_original`；
3. **最后调图稀疏度**：只在弱方向上比较 `dense`、`top3`、`top5`、`top10`，不要全局改默认；
4. **保留方向级配置文件**：每个方向一个覆盖配置，避免“全局最优”把强方向拖坏。

### 18.4 为什么这条路更稳

- 它不会改动主模型的训练逻辑，因此强方向结果更容易保持；
- 弱方向通常不是“模型完全无效”，而是“阈值、分布、稀疏度和校准不匹配”；
- 先做方向级分流，比直接改特征选择或评分机制更不容易把整体性能打坏；
- 后续如果某个弱方向确实有效，再把那组配置单独固化成专用实验分支。


## 19. 方向专用配置文件

已经新增三份弱方向专用配置草案，建议直接用于单独试跑：

- `configs/count_granger_directions/Thunderbird_to_Spirit.yaml`
- `configs/count_granger_directions/Thunderbird_to_BGL.yaml`
- `configs/count_granger_directions/Spirit_to_BGL.yaml`

这三份配置都保持主模型不变，只覆盖了：

- `split.eval_mode: stratified_original`
- `detect.threshold_strategy: f1`
- `detect.score_component: validation_best`
- `detect.candidate_components: [score, residual, edge]`

推荐的运行方式仍然是：

```powershell
python count_granger_main.py --source_data <source.log> --target_data <target.log> --config <direction_config.yaml> --mode train
```

如果后续要继续提弱方向，再在这三份方向配置上分别做 `dense / top-k` 的小网格，而不要先改主模型默认值。
## 20. BGL 作为目标域的后处理尝试

针对 `Thunderbird -> BGL` 和 `Spirit -> BGL` 这两个弱方向，曾尝试通过时间窗口和后处理提升 Precision/F1。

新增过以下方向专用配置：

- `configs/count_granger_directions/Thunderbird_to_BGL_120s_post.yaml`
- `configs/count_granger_directions/Thunderbird_to_BGL_300s_post.yaml`
- `configs/count_granger_directions/Spirit_to_BGL_120s_post.yaml`
- `configs/count_granger_directions/Spirit_to_BGL_300s_post.yaml`

主要思路包括：

1. 将 `preprocess.time_bin_seconds` 从 `60` 调整为 `120` 或 `300`，希望减少 BGL 目标域中异常窗口过稀的问题；
2. 加入简单后处理，例如 `rolling_mean_window: 2` 和 `min_consecutive: 2`，希望过滤孤立误报。

实验结果表明，这类简单后处理没有稳定提升弱方向效果，甚至会明显降低 Recall 和 F1。因此它更适合作为诊断实验，不建议作为最终 baseline。

## 21. Cache 复用策略

当前批量实验和手动命令统一复用 pair-level cache，默认根目录为：

```text
results/cache/count_granger
```

`count_granger_main.py` 中每个 source-target 方向的 cache 路径为：

```text
results/cache/count_granger/<source>_to_<target>
```

`run_count_granger_pairs.ps1` 和 `run_count_granger_ablations.py` 已调整为复用同一个 pair cache，而不是为每个 ablation 单独创建 cache。这样 `full`、`no_cluster`、`target_only`、`edge_only` 等实验只要 source-target 和预处理参数一致，就可以复用同一份 count series。

需要注意：如果修改了 `time_bin_seconds`、`semantic_clustering`、`max_lines`、`max_templates` 等预处理参数，cache 会重新生成，这是合理的。

## 22. 非归一化 alpha/beta 加权搜索

曾增加过 `weighted_search`，用于自动搜索组合分数：

```text
score_beta = 1.0 * normalized_residual + beta * normalized_edge
```

其中 beta 搜索集合为：

```text
{0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0}
```

实验目标是让 validation 自动选择最优 beta，然后观察 test Precision、Recall 和 F1 是否提升。

相关配置包括：

- `configs/count_granger_directions/Thunderbird_to_Spirit_weighted_search.yaml`
- `configs/count_granger_directions/Thunderbird_to_BGL_weighted_search.yaml`
- `configs/count_granger_directions/Spirit_to_BGL_weighted_search.yaml`

实验结论：非归一化 beta 搜索对三个弱方向帮助很小，validation 通常仍偏向选择单独的 `edge` 或 `residual`，说明当前 residual 和 edge 的互补性不足。

## 23. 归一化 alpha/beta 加权搜索

随后又尝试归一化加权：

```text
score_alpha = alpha * normalized_residual + (1 - alpha) * normalized_edge
```

其中 alpha 搜索集合为：

```text
{0.0, 0.25, 0.5, 0.75, 1.0}
```

该实验希望避免非归一化组合中 residual 尺度过强或 edge 尺度过弱的问题。

实验结论：归一化 alpha/beta 搜索仍然没有明显提升 `Thunderbird -> Spirit`、`Thunderbird -> BGL`、`Spirit -> BGL` 三个弱方向。因此，继续调线性权重不是优先方向。

## 24. 强异常窗口标签实验

为了分析 BGL 作为目标域时 Precision/F1 偏低的问题，预处理阶段新增了 `anomaly_counts`，并在划分配置中加入：

```text
split.eval_label_min_anomaly_lines
```

评价标签定义为：

```text
label_eval = 1(anomaly_counts >= eval_label_min_anomaly_lines)
```

也就是说，一个时间窗口内异常日志行数达到阈值时，才将该窗口视为异常窗口。

测试过 `ge3`、`ge5` 和 `ge10` 等强异常标签设置。结果显示，强异常标签可以略微提升 BGL target 下的 Precision/F1，但提升幅度有限：

- `Thunderbird -> BGL` 在 `ge10` 下 F1 约为 `0.710`；
- `Spirit -> BGL` 在 `ge10` 下 F1 约为 `0.701`。

因此，强异常窗口标签可以作为诊断实验保留，但不应作为主要提升方案。

## 25. 六方向 Source/Target Granger 图结构分析

本轮新增 `analyze_granger_transferability.py`，用于在 BGL、Thunderbird、Spirit 三个数据集之间做六个有向 source-target 方向的 Granger 图结构可迁移性诊断。分析结果保存在：

```text
results/granger_transferability/20260812_152633/graph_transfer_summary.csv
results/granger_transferability/20260812_152633/graph_transfer_summary.md
```

### 25.1 分析目的

当前弱方向主要集中在 `Thunderbird -> BGL`、`Spirit -> BGL`、`Thunderbird -> Spirit`。继续修改模型之前，需要先判断问题到底来自哪里：

1. 源域和目标域是否选择了相似的日志簇节点；
2. 源域和目标域是否具有相似的 Granger 边；
3. 目标节点的 Top-k 父节点是否可迁移；
4. 共享边的权重排序是否一致；
5. 弱方向是否是因为 source-target 图结构本身不可迁移。

因此，该脚本分别拟合 source 图和 target 图，然后比较两张图的节点、边、Top-k 父节点、边权相关性和方向一致性。

### 25.2 当前六方向结构对比结果

| source | target | node_jaccard | edge_jaccard | topk_edge_jaccard | mean_topk_parent_jaccard | shared_edge_weight_spearman | direction_consistency | transferability_score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BGL | Thunderbird | 0.050000 | 0.002259 | 0.000000 | 0.000000 | 0.285714 | 0.500000 | 0.172143 |
| Thunderbird | BGL | 0.050000 | 0.002259 | 0.000000 | 0.000000 | 0.285714 | 0.500000 | 0.172143 |
| BGL | Spirit | 0.059701 | 0.005317 | 0.002132 | 0.019737 | -0.086937 | 0.500000 | 0.118550 |
| Spirit | BGL | 0.059701 | 0.005317 | 0.002132 | 0.019737 | -0.086937 | 0.500000 | 0.118550 |
| Thunderbird | Spirit | 0.293103 | 0.110022 | 0.010101 | 0.023564 | 0.131909 | 0.500000 | 0.217343 |
| Spirit | Thunderbird | 0.293103 | 0.110022 | 0.010101 | 0.023564 | 0.131909 | 0.500000 | 0.217343 |

### 25.3 关键观察

1. **图结构相似度是 pair-level 的，不是 direction-level 的。** 例如 `BGL -> Thunderbird` 和 `Thunderbird -> BGL` 的图相似度完全相同，但检测效果差异很大。这说明仅靠 source/target 两张 Granger 图的静态相似度，无法解释方向不对称问题。
2. **BGL 与 Thunderbird 的节点和边重合极低。** `node_jaccard=0.05`，`edge_jaccard=0.002259`，`topk_edge_jaccard=0`。这表明两者可共享的 Granger 结构很少，但 `BGL -> Thunderbird` 仍能取得较好效果，说明强方向并不完全依赖图结构重合。
3. **Thunderbird 与 Spirit 的节点/边相似度最高，但 `Thunderbird -> Spirit` 仍然偏弱。** 这进一步说明弱方向不只是“图不像”，而更可能是目标域异常比例、目标正常校准、阈值选择、训练分布和评价采样共同造成的。
4. **BGL 作为 target 时存在特殊困难。** `Thunderbird -> BGL` 和 `Spirit -> BGL` 的问题更像是目标域 BGL 的异常窗口稀疏、正常窗口占比极高、validation/test 采样敏感，以及 residual/edge 分数对正常和异常的间隔不足。
5. **Top-k 父节点 Jaccard 普遍很低。** 说明“直接迁移源域 Granger 父节点结构”不是稳定方案。如果要利用图结构，应更偏向目标域自适应建图，而不是强行让 source 图指导 target 图。

### 25.4 对后续修改的启示

目前结果不支持继续做简单的全局模型修改，也不支持继续只调 `top_k_parents`、`alpha/beta` 或窗口后处理。更合理的方向是：

1. **保留原版主模型作为 full baseline。** 避免为了提升弱方向而损坏 `BGL -> Thunderbird`、`Spirit -> Thunderbird` 等强方向。
2. **对弱方向做 target-aware 处理。** 尤其是 BGL 作为 target 时，应单独处理阈值、评价采样和目标正常校准。
3. **不要把 source 图结构硬迁移到 target。** 当前 Top-k 边和父节点重合度太低，硬迁移容易引入错误先验。
4. **优先做目标域分数分布诊断。** 比较 normal/anomaly 的 residual、edge 分布重叠程度，确认弱方向到底是分数不可分，还是阈值选择失败。
5. **后续可以考虑 target 图重估计或图结构置信度加权。** 也就是让 source 只提供弱先验，最终边结构仍由 target normal windows 决定。

### 25.5 当前结论

六方向 Granger 图结构分析是必要的，但当前结果说明：弱方向的核心问题不是简单的“source-target 图结构相似度不足”，而是 **target 侧分布、异常窗口定义、校准和阈值选择** 与图分数之间不匹配。下一步不建议继续做全局模型改造，而应在保持原始 full 效果不变的前提下，对 `Thunderbird -> BGL`、`Spirit -> BGL`、`Thunderbird -> Spirit` 做方向专用的 target-aware 诊断和配置。
## 26. Source Granger 关系迁移到 Target 的 Top-K 保留率分析

### 26.1 分析目的

为了直接检查 source 学到的重要 Granger 关系能否迁移到 target，本轮新增：

`	ext
analyze_granger_edge_retention.py
`

该脚本在每个数据集的正常窗口上分别拟合 Granger 图，然后执行以下检查：

1. 从 source 图中按边权取 Top-K 有向边；
2. 在 target 图中查找完全相同的有向边；
3. 计算原始保留率、端点条件保留率和权重保留率；
4. 记录这些 source 边在 target 图中的排名位置。

其中一条边统一表示为：

`	ext
parent -> child
`

在模型邻接矩阵中对应：

`	ext
adjacency[child, parent]
`

结果保存于：

`	ext
results/granger_edge_retention/20260812_195149/edge_retention_summary.csv
results/granger_edge_retention/20260812_195149/edge_retention_summary.md
`

### 26.2 指标定义

**原始保留率**：

`	ext
raw_retention = source Top-K 边中同时出现在 target 图中的边数 / K
`

**端点条件保留率**：

`	ext
endpoint_conditioned_retention =
同时出现在 target 图中的 source Top-K 边数 /
两个端点都被 target 选中的 source Top-K 边数
`

端点条件保留率用于区分两种情况：

- 关系确实没有迁移；
- target 没有选择对应节点，因此无法判断该关系是否迁移。

**加权保留率**：

`	ext
weighted_retention = 被保留 source 边的 source 权重之和 /
source Top-K 边的 source 权重总和
`

该指标比单纯计数更关注重要边是否保留。

### 26.3 Top-50 结果

| source | target | 原始保留率 | 端点条件保留率 | 加权保留率 | source Top-50 中端点可比边数 | 保留边数 |
|---|---|---:|---:|---:|---:|---:|
| BGL | Thunderbird | 0.020 | 1.000 | 0.018 | 1 | 1 |
| Thunderbird | BGL | 0.000 | 无可比边 | 0.000 | 0 | 0 |
| BGL | Spirit | 0.020 | 1.000 | 0.024 | 1 | 1 |
| Spirit | BGL | 0.020 | 0.500 | 0.015 | 2 | 1 |
| Thunderbird | Spirit | 0.020 | 1.000 | 0.019 | 1 | 1 |
| Spirit | Thunderbird | 0.000 | 无可比边 | 0.000 | 0 | 0 |

### 26.4 不同 K 的保留率变化

| source | target | K=20 | K=50 | K=100 | K=200 |
|---|---|---:|---:|---:|---:|
| BGL | Thunderbird | 0.050 | 0.020 | 0.010 | 0.015 |
| Thunderbird | BGL | 0.000 | 0.000 | 0.000 | 0.000 |
| BGL | Spirit | 0.050 | 0.020 | 0.010 | 0.015 |
| Spirit | BGL | 0.000 | 0.020 | 0.010 | 0.005 |
| Thunderbird | Spirit | 0.000 | 0.020 | 0.050 | 0.120 |
| Spirit | Thunderbird | 0.000 | 0.000 | 0.050 | 0.065 |

### 26.5 结论

1. **Source 的 Top-K Granger 边整体不能直接迁移到 Target。** 大多数方向的 Top-50 原始保留率只有  % 或 2%，远低于可以支撑强 source-guided 迁移的水平。
2. **Thunderbird -> BGL 的 source 边迁移性最差。** Top-20、Top-50、Top-100 和 Top-200 的保留率均为  。这说明 Thunderbird 学到的重要关系在 BGL 中几乎没有对应关系，不能把 Thunderbird 图作为 BGL 的直接结构先验。
3. **Spirit -> BGL 也不适合直接迁移 source 图。** Top-50 仅保留 1/50 条边，Top-200 只有 1/200 条边；这与该方向检测效果较弱相吻合。
4. **BGL -> Thunderbird 的保留率也很低，但该方向检测效果仍然较好。** 因此，source 边迁移不是取得高检测效果的必要条件。强方向可能主要依赖 target 自身的训练和目标域异常分数分离，而不是 source 图边的直接复用。
5. **Thunderbird -> Spirit 在较大的 K 下保留率有所上升。** Top-200 保留率为  .12，但 Top-50 仍只有  .02，说明只有部分中等重要关系可以迁移，最核心的 Top-20/Top-50 关系并不稳定。
6. **Spirit -> Thunderbird 在 Top-100 和 Top-200 才出现部分保留。** 这说明两者存在一定中等重要关系重合，但核心边仍没有稳定重合。
7. **不能只看端点条件保留率。** 例如 BGL -> Thunderbird 的端点条件保留率为 1.0，但只有 1 条 Top-50 边的两个端点同时出现在 target 图中。因此实际原始保留率仍只有  .02，端点条件结果不能解释为“迁移性很强”。

### 26.6 对模型设计的影响

当前结果不支持以下做法：

- 将 source Top-K Granger 边直接硬编码到 target 模型；
- 使用 source 图强制约束 target 图的边集合；
- 假设 source 和 target 之间存在大量可复用的核心因果关系。

更合理的设计是：

1. 以 target normal windows 学到的图作为主要结构；
2. source 图只作为低权重的软先验；
3. 只有满足端点共享、边权稳定和 target 中排名较高的关系，才允许迁移；
4. 对迁移边设置置信度门控，而不是直接复制 source 边；
5. 在实验中把 aw_retention@K 作为迁移性诊断指标，而不是把静态图相似度直接当成模型性能依据。

### 26.7 当前最终判断

Source-to-Target Granger 关系迁移检查是必要的，而且本轮结果已经给出明确结论：**当前三个数据集之间的核心 Granger 边迁移性整体较弱，特别是迁移到 BGL 时几乎不存在稳定的 source 边保留。**

因此，后续不应继续围绕“如何把 source 图迁移得更强”做全局修改。更有价值的方向是研究 target-aware 图建模、迁移边置信度筛选，以及 source 图仅作为辅助信息时是否能带来增益。
## 27. 六方向分数可分性与阈值迁移诊断

### 27.1 分析目的

在确认 source Top-K Granger 边整体难以直接迁移之后，本轮继续检查弱方向到底是：

1. 分数本身无法区分 normal/anomaly；
2. 还是 validation 阈值迁移到 test 时失效。

新增脚本：

`	ext
analyze_score_separability.py
`

输出结果保存在：

`	ext
results/score_separability/20260812_203437/score_separability_components.csv
results/score_separability/20260812_203437/score_separability_selected.csv
results/score_separability/20260812_203437/score_separability_components.md
`

该脚本对六个方向分别统计 score、esidual、edge 三个评分分量的：

- test Precision、Recall、F1；
- ROC-AUC、PR-AUC；
- normal/anomaly 分数分布间隔；
- anomaly 落在 normal 95 分位以下的比例；
- test oracle F1 与 validation 阈值 F1 的差距。

### 27.2 当前 selected component 结果

| source | target | selected component | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| BGL | Thunderbird | edge | 0.956 | 0.978 | 0.967 | 0.967 | 0.918 |
| BGL | Spirit | edge | 0.956 | 0.880 | 0.916 | 0.919 | 0.858 |
| Thunderbird | BGL | residual | 0.561 | 0.929 | 0.699 | 0.593 | 0.582 |
| Thunderbird | Spirit | edge | 0.652 | 0.950 | 0.773 | 0.819 | 0.777 |
| Spirit | BGL | score | 0.558 | 0.923 | 0.695 | 0.630 | 0.610 |
| Spirit | Thunderbird | edge | 0.939 | 0.983 | 0.960 | 0.949 | 0.872 |

### 27.3 组件级可分性结果

| source | target | component | test F1 | oracle F1 | ROC-AUC | PR-AUC | separation_gap_p05_p95 | anomaly_below_normal_p95_rate |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BGL | Thunderbird | edge | 0.967 | 0.967 | 0.967 | 0.918 | 0.481 | 0.019 |
| BGL | Spirit | edge | 0.916 | 0.918 | 0.919 | 0.858 | -0.474 | 0.118 |
| Thunderbird | BGL | residual | 0.699 | 0.702 | 0.593 | 0.582 | -14.100 | 0.884 |
| Thunderbird | Spirit | edge | 0.773 | 0.774 | 0.819 | 0.777 | -1.011 | 0.775 |
| Spirit | BGL | score | 0.695 | 0.702 | 0.630 | 0.610 | -13.569 | 0.882 |
| Spirit | Thunderbird | edge | 0.960 | 0.961 | 0.949 | 0.872 | -0.236 | 0.438 |

其中：

`	ext
separation_gap_p05_p95 = anomaly_p05 - normal_p95
`

如果该值为正，说明异常分数的低分位仍高于正常分数的高分位，可分性很好；如果该值大幅为负，说明 normal/anomaly 分数高度重叠。

`	ext
anomaly_below_normal_p95_rate
`

表示异常样本中有多少比例低于 normal 的 95 分位。该值越高，说明异常越容易被正常高分尾部淹没。

### 27.4 关键结论

1. **弱方向主要不是阈值迁移失败，而是分数本身不可分。** 所有方向的 	est oracle F1 与 validation 阈值得到的 	est F1 差距都很小。例如 Thunderbird -> BGL 的 residual 分量只差约  .0024，Spirit -> BGL 的 score 分量只差约  .0072。这说明即使直接在 test 上选最优阈值，也只能小幅提升，问题不在阈值选择。
2. **BGL 作为 target 时，分数可分性明显不足。** Thunderbird -> BGL 的最佳 residual 分量 ROC-AUC 只有  .593，PR-AUC 只有  .582；Spirit -> BGL 的最佳 score 分量 ROC-AUC 只有  .630，PR-AUC 只有  .610。这不是一个可通过调阈值解决的问题。
3. **BGL target 的 normal/anomaly 分数重叠非常严重。** Thunderbird -> BGL 中约 88.4% 的异常样本低于 normal 95 分位；Spirit -> BGL 中约 88.2% 的异常样本低于 normal 95 分位。这说明大量异常窗口在当前模型下得分并不高。
4. **强方向依赖 edge 分量，而不是 residual/combined score。** BGL -> Thunderbird 和 Spirit -> Thunderbird 的 edge 分量明显优于 residual 和 score，说明 Granger 边结构分数在 Thunderbird target 上有效。
5. **BGL -> Spirit 虽然 separation gap 略为负，但 edge 分量仍有较好的排序能力。** 其 edge ROC-AUC 为  .919，PR-AUC 为  .858，说明分数排序足够好，阈值也基本稳定。
6. **Thunderbird -> Spirit 是中等可分，不是完全失败。** edge ROC-AUC 为  .819、PR-AUC 为  .777，说明还有提升空间，但问题不是简单阈值，而是 edge 分量对部分异常窗口区分不足。

### 27.5 对后续修改的影响

当前诊断基本否定了继续优先做以下事情：

- 单纯继续调 min_precision、min_recall 或 threshold strategy；
- 继续搜索 lpha/beta 线性组合；
- 继续调 	op_k_parents 期待 BGL target 大幅提升；
- 继续做简单平滑或连续窗口后处理。

因为这些方法主要影响阈值或分数组合，而现在 BGL target 的核心问题是 **score separability 不足**。

更值得做的是：

1. **重新设计 BGL target 的特征或异常标签粒度。** 当前 60s count window 对 BGL 异常的刻画可能过粗或过稀，导致异常窗口得分接近正常窗口。
2. **做 target-only 与 full 的分数组件对比。** 如果 target-only 在 BGL 上更好，说明 source 数据反而干扰了目标域建模；如果 full 更好，说明 source 仍有辅助价值。
3. **引入局部变化型分数。** BGL 异常可能不是全局 Granger 边异常，而是局部突变、短时频率突增或事件簇组合变化，需要增加 change-score 或 burst-score。
4. **针对 Thunderbird -> Spirit 做 edge 分量增强。** 该方向不是完全不可分，可以考虑目标域 edge calibration、目标图重估计或中等重要边筛选。
5. **论文叙事上不要强调 source 图硬迁移。** 更稳妥的说法是：source 提供辅助统计先验，异常检测效果主要取决于 target-aware 的分数可分性和目标域校准。

### 27.6 当前最终判断

六方向分数可分性诊断给出的结论比单纯调参更明确：**弱方向的主要问题不是阈值选择失败，而是当前模型在这些 target 上没有产生足够可分的异常分数。**

因此，后续如果要实质提升 Thunderbird -> BGL、Spirit -> BGL 和 Thunderbird -> Spirit，应该优先改特征表示或增加新的 target-aware anomaly score，而不是继续围绕现有 residual/edge/score 做小范围调参。
## 28. Target-only 与 full 的 edge 可分性对比

### 28.1 分析目的

前一轮分数可分性诊断表明，弱方向的主要问题不是阈值选错，而是分数本身不可分。为了进一步确认 source 是否在干扰 target 的 edge 分量，本轮新增：

`	ext
analyze_target_only_edge_separability.py
`

该脚本对每个方向比较两种模式：

1. ull：source + target 一起拟合；
2. 	arget_only：只用 target 拟合，source 不参与训练。

同时保持相同的 target calibration 和相同的评分分量 edge，从而判断 source 是否真正帮助了 edge 分数，还是反而造成干扰。

结果保存于：

`	ext
results/target_only_edge_separability/20260813_145952/target_only_edge_modes.csv
results/target_only_edge_separability/20260813_145952/target_only_edge_comparison.csv
results/target_only_edge_separability/20260813_145952/target_only_edge_comparison.md
`

### 28.2 结果概览

| source | target | full F1 | target_only F1 | Delta F1 | full ROC-AUC | target_only ROC-AUC | Delta ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| BGL | Thunderbird | 0.967 | 0.955 | -0.012 | 0.967 | 0.950 | -0.018 |
| BGL | Spirit | 0.916 | 0.757 | -0.160 | 0.919 | 0.787 | -0.133 |
| Thunderbird | BGL | 0.664 | 0.664 | 0.000 | 0.547 | 0.536 | -0.011 |
| Thunderbird | Spirit | 0.773 | 0.765 | -0.009 | 0.819 | 0.790 | -0.029 |
| Spirit | BGL | 0.660 | 0.660 | 0.000 | 0.544 | 0.544 | 0.000 |
| Spirit | Thunderbird | 0.960 | 0.960 | 0.000 | 0.949 | 0.949 | 0.000 |

### 28.3 关键观察

1. **source 并没有普遍提升 edge 可分性。** 在 six directions 里，只有 BGL -> Thunderbird 仍然保持接近高水平，其他方向中 	arget_only 并没有比 ull 更差到足以说明 source 带来明显正收益。
2. **BGL -> Spirit 是最明显的负面案例。** 	arget_only 的 F1 从  .916 直接掉到  .757，ROC-AUC 也从  .919 降到  .787。这说明 source 并不是简单干扰，而是在这个方向上对 edge 分数确实有明显帮助。
3. **Thunderbird -> BGL 和 Spirit -> BGL 基本不靠 source 改善 edge。** 这两个方向里 full 和 target_only 的 F1 几乎相同，说明 source 不是提升边可分性的核心因素。
4. **Spirit -> Thunderbird 也几乎没有差异。** full 和 target_only 的 edge 表现一致，说明 source 不会实质改变 target 的 edge separability。
5. **BGL -> Spirit 与前两轮诊断共同说明：source 的作用不稳定，而且是方向相关的。** 有些方向 source 帮助 edge，有些方向 source 基本无效。

### 28.4 对前面结论的修正

这轮结果说明，不能简单下结论说 “source 一定在干扰 target”。 更准确的说法是：

- source 对 edge 分数的影响是 **方向相关** 的；
- 在 BGL -> Spirit 上 source 有明显帮助；
- 在 Thunderbird -> BGL、Spirit -> BGL、Spirit -> Thunderbird 上 source 对 edge 的增益很弱或几乎没有；
- 所以当前问题不是“source 一定坏”，而是 **source 不是稳定的跨域增益源**。

### 28.5 当前最终判断

	arget_only edge separability 是有价值的，而且这次实验给出了一个比前两轮更细的判断：**source 对 edge 分量的作用是方向敏感的，不应被假设为总是正向，也不应被假设为总是干扰。**

因此，后续如果要继续提升弱方向，不能简单地关闭 source 或保留 source，而应针对每个方向单独判断：

1. source 是否真的提升了 edge separability；
2. 若提升，提升来自哪些图边或哪些节点簇；
3. 若没有提升，再考虑 target-only 或 target-aware 图建模。

## 29. Target-aware level/change/burst 分数实验

### 29.1 实验动机

前几轮诊断表明，Thunderbird -> BGL、Spirit -> BGL 和 Thunderbird -> Spirit 等弱方向的主要问题不是阈值选择错误，而是现有 residual/edge/score 分数在目标域上可分性不足。尤其当 BGL 作为 target 时，source 图或 edge 分量并不能稳定提供跨域增益。因此，本轮不再继续围绕 source Granger edge 做细粒度筛选，而是新增 target-aware 的局部计数偏离分数，用目标域正常窗口自身的统计基线刻画异常。

本轮新增的核心思想是：

```text
target train normal windows
  -> robust target baseline
  -> level / change / burst local deviation scores
  -> validation-best component selection
  -> target test evaluation
```

该方案不替代原有 Granger edge 分数，而是把新的 target-aware 分数组件与 residual、edge 并列，由验证集选择最终用于测试集的分数组件。

### 29.2 新增分数组件

本轮在 `count_granger_model.py` 中新增三类 target-aware 分数：

1. **level score**：度量当前窗口中模板簇计数相对 target 正常基线的绝对偏离。对每个选中特征使用 `log1p(count)`，在 target train normal 上估计 median 和 robust scale，然后对当前窗口计算 robust z-score，并取 Top-k 特征均值。
2. **change score**：度量当前窗口相对前一窗口的一阶变化是否异常。正常基线只使用当前窗口和前一窗口均为正常的 target train 样本估计，避免异常边界污染变化统计。
3. **burst score**：度量短期均值相对长期中位数是否出现局部爆发。当前默认使用 `burst_short_window=3`、`burst_long_window=30`。

同时新增两个固定组合分数：

```text
max_edge_change = max(edge, change)
max_change_burst = max(change, burst)
```

本轮默认 Top-k 聚合参数为：

```text
target_score_top_ratio = 0.05
target_score_top_min = 3
```

即每个窗口只聚合偏离最大的少数模板簇，避免局部异常被全维平均稀释。

### 29.3 代码与结果位置

新增或修改的主要文件包括：

```text
count_granger_model.py
count_granger_main.py
configs/count_granger_ablations/target_scores.yaml
```

实验结果保存于：

```text
results/target_scores_probe/20260813_191802_Thunderbird_to_BGL
results/target_scores_probe/20260813_191858_BGL_to_Thunderbird
results/target_scores_probe/20260813_191917_BGL_to_Spirit1G
results/target_scores_probe/20260813_191936_Thunderbird_to_Spirit1G
results/target_scores_probe/20260813_191948_Spirit1G_to_BGL
results/target_scores_probe/20260813_192009_Spirit1G_to_Thunderbird
```

每个方向下包含 `full/` 和 `target_scores/` 两个子目录。重点结果文件包括：

```text
summary.csv
summary.md
summary.json
target_scores/ablation_summary.json
target_scores/target_val_scores.csv
target_scores/target_test_scores.csv
target_scores/count_granger_model.json
```

### 29.4 六方向结果概览

| source | target | full selected | full Precision | full Recall | full F1 | target_scores selected | target_scores Precision | target_scores Recall | target_scores F1 | Delta F1 | target_scores ROC-AUC | target_scores PR-AUC | level ROC-AUC | edge ROC-AUC |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Thunderbird | BGL | residual | 0.561 | 0.929 | 0.699 | level | 0.919 | 0.775 | 0.841 | +0.141 | 0.886 | 0.883 | 0.886 | 0.547 |
| Spirit | BGL | score | 0.558 | 0.923 | 0.695 | level | 0.916 | 0.722 | 0.807 | +0.112 | 0.858 | 0.847 | 0.858 | 0.544 |
| BGL | Thunderbird | edge | 0.956 | 0.978 | 0.967 | level | 0.996 | 0.991 | 0.993 | +0.026 | 0.996 | 0.983 | 0.996 | 0.967 |
| Spirit | Thunderbird | edge | 0.939 | 0.983 | 0.960 | level | 0.996 | 0.993 | 0.995 | +0.034 | 0.996 | 0.989 | 0.996 | 0.949 |
| Thunderbird | Spirit | edge | 0.652 | 0.950 | 0.773 | level | 0.705 | 0.947 | 0.808 | +0.035 | 0.822 | 0.755 | 0.822 | 0.819 |
| BGL | Spirit | edge | 0.956 | 0.880 | 0.916 | edge | 0.956 | 0.880 | 0.916 | +0.000 | 0.919 | 0.858 | 0.942 | 0.919 |

### 29.5 关键观察

1. **target-aware level score 是本轮真正有效的新增分量。** 六个方向中，除 BGL -> Spirit 仍由 edge 被验证集选中外，其余五个方向均选择 level。尤其在两个 BGL target 弱方向上，level 明显提升了异常分数可分性。
2. **BGL target 的提升最明显。** Thunderbird -> BGL 的 F1 从 0.699 提升到 0.841，Spirit -> BGL 的 F1 从 0.695 提升到 0.807。这说明 BGL 异常更可能表现为目标域局部计数水平偏离，而不是稳定的 Granger edge 异常。
3. **强方向未被破坏。** BGL -> Thunderbird 和 Spirit -> Thunderbird 原本已经较强，本轮 level 仍进一步提升 F1；BGL -> Spirit 中 validation-best 保持选择 edge，因此整体结果没有退化。
4. **change 和 burst 暂未成为主力。** 从本轮结果看，短时变化和短期爆发分数并未稳定超过 level。当前最有价值的新增信号是 target normal level deviation，而不是一阶变化或 burst 结构。
5. **不要用均值单独论证 Thunderbird -> BGL。** 该方向 test 中存在极端异常窗口，会显著拉高 anomaly mean。但分位数仍支持 level 的可分性，例如 normal p95 为 0.434，anomaly median 为 1.599，anomaly p95 为 14.363。因此论文叙事应优先报告 F1、ROC-AUC、PR-AUC 和稳健分位数，而不是依赖均值差异。

### 29.6 对前面结论的修正

前面第 27 节提出“弱方向应优先增加新的 target-aware anomaly score”，本轮结果支持这一判断。更具体地说，当前最有效的 target-aware score 并不是复杂的 source-edge transfer confidence，也不是 change/burst，而是简单但稳健的 **target normal level deviation**。

因此，后续方法叙事可以从“source 图硬迁移”调整为：

- Granger edge 分数用于捕获跨模板定向依赖异常；
- target-aware level 分数用于捕获目标域局部计数水平偏离；
- validation-best 机制在每个方向上选择更适合目标域异常形态的分数组件；
- source 信息可以作为辅助统计先验，但最终检测效果依赖 target-aware 校准和目标域分数可分性。

### 29.7 当前最终判断

本轮实验是目前最值得保留的改进方向。与 transfer-confidence edge 相比，target-aware level score 在弱方向上产生了实质性提升，而且没有破坏强方向表现。

后续建议将主方法收敛为较简洁的候选组件集合：

```text
residual
edge
level
```

其中 `change` 和 `burst` 可以暂时作为消融组件保留，但不宜作为主方法重点叙述。论文中更稳妥的表述是：Count-Granger 不仅利用正则化 VAR 提取 Granger 风格的定向影响异常，还引入目标域正常分布校准的局部计数偏离分数，以弥补纯 edge 分数在部分 target 上可分性不足的问题。

## 30. Top-k weighted level 分数实验

### 30.1 实验目的

第 29 节中的 target-aware level score 使用 Top-k mean 聚合窗口内偏离最大的模板簇。该做法比较稳健，但会把最强局部异常和其他较弱偏离一起平均，可能稀释尖峰模板的贡献。因此，本轮尝试将 Top-k mean 改为 Top-k weighted mean，以增强 level 分数对局部异常的表达能力。

本轮实验不改变 target-aware level 的基本定义，只改变 Top-k 聚合方式。代码中保留 `mean` 作为默认值，同时新增三种加权聚合：

```text
rank_weighted_mean
score_weighted_mean
softmax_weighted_mean
```

其中重点比较的是 `rank_weighted_mean`。该方法按照 Top-k 内部排名给更靠前的模板更高权重，默认 `target_score_rank_weight_power=1.0`。与 score-weighted 或 softmax-weighted 相比，rank-weighted 不直接依赖分数幅值，因此更保守，也更不容易被极端值主导。

### 30.2 代码与配置

本轮主要修改如下：

```text
count_granger_model.py
count_granger_main.py
configs/count_granger_config.yaml
configs/count_granger_ablations/target_scores_rank_weighted_level.yaml
configs/count_granger_ablations/target_scores_score_weighted_level.yaml
configs/count_granger_ablations/target_scores_softmax_weighted_level.yaml
```

新增配置项包括：

```text
target_score_aggregation: mean | rank_weighted_mean | score_weighted_mean | softmax_weighted_mean
target_score_rank_weight_power: 1.0
target_score_softmax_temperature: 1.0
```

默认配置仍为：

```text
target_score_aggregation: mean
```

因此旧配置下的 target-aware level 结果保持不变。

### 30.3 结果位置

实验结果保存于：

```text
results/count_granger_weighted_level/20260814_111917_Thunderbird_to_Spirit1G
results/count_granger_weighted_level/20260814_111940_Thunderbird_to_BGL
results/count_granger_weighted_level/20260814_112003_Spirit1G_to_BGL
results/count_granger_weighted_level/20260814_112048_BGL_to_Thunderbird
results/count_granger_weighted_level/20260814_112107_BGL_to_Spirit1G
results/count_granger_weighted_level/20260814_112125_Spirit1G_to_Thunderbird
```

额外的 `score_weighted_mean` 和 `softmax_weighted_mean` 对比结果保存于：

```text
results/count_granger_weighted_level/20260814_112341_Thunderbird_to_Spirit1G
results/count_granger_weighted_level/20260814_112350_Thunderbird_to_BGL
results/count_granger_weighted_level/20260814_112412_Spirit1G_to_BGL
results/count_granger_weighted_level/20260814_112428_BGL_to_Thunderbird
results/count_granger_weighted_level/20260814_112448_BGL_to_Spirit1G
results/count_granger_weighted_level/20260814_112505_Spirit1G_to_Thunderbird
```

### 30.4 mean 与 rank-weighted 六方向对比

| source | target | mean selected | mean Precision | mean Recall | mean F1 | rank-weighted selected | rank-weighted Precision | rank-weighted Recall | rank-weighted F1 | Delta F1 | rank-weighted ROC-AUC | rank-weighted PR-AUC |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Thunderbird | Spirit | level | 0.704901 | 0.946935 | 0.808186 | level | 0.743117 | 0.972758 | 0.842571 | +0.034385 | 0.811438 | 0.727086 |
| Thunderbird | BGL | level | 0.918782 | 0.775161 | 0.840883 | level | 0.918782 | 0.775161 | 0.840883 | +0.000000 | 0.884760 | 0.878243 |
| Spirit | BGL | level | 0.915761 | 0.721627 | 0.807186 | level | 0.915761 | 0.721627 | 0.807186 | +0.000000 | 0.857560 | 0.841740 |
| BGL | Thunderbird | level | 0.995592 | 0.991223 | 0.993403 | level | 0.996145 | 0.992320 | 0.994229 | +0.000826 | 0.995490 | 0.982789 |
| BGL | Spirit | edge | 0.955905 | 0.879682 | 0.916211 | edge | 0.955905 | 0.879682 | 0.916211 | +0.000000 | 0.919236 | 0.857921 |
| Spirit | Thunderbird | level | 0.995602 | 0.993417 | 0.994509 | level | 0.997785 | 0.988481 | 0.993111 | -0.001397 | 0.996822 | 0.992347 |

### 30.5 三种 weighted 聚合对比

| source | target | mean F1 | rank-weighted F1 | score-weighted F1 | softmax-weighted F1 | 最优 F1 |
|---|---|---:|---:|---:|---:|---:|
| Thunderbird | Spirit | 0.808186 | 0.842571 | 0.838198 | 0.836560 | 0.842571 |
| Thunderbird | BGL | 0.840883 | 0.840883 | 0.840883 | 0.840883 | 0.840883 |
| Spirit | BGL | 0.807186 | 0.807186 | 0.807186 | 0.807186 | 0.807186 |
| BGL | Thunderbird | 0.993403 | 0.994229 | 0.992572 | 0.992026 | 0.994229 |
| BGL | Spirit | 0.916211 | 0.916211 | 0.916211 | 0.916211 | 0.916211 |
| Spirit | Thunderbird | 0.994509 | 0.993111 | 0.993943 | 0.993119 | 0.994509 |

### 30.6 关键观察

1. **rank-weighted level 对 Thunderbird -> Spirit 的提升最明显。** F1 从 `0.808186` 提升到 `0.842571`，Precision 从 `0.704901` 提升到 `0.743117`，Recall 从 `0.946935` 提升到 `0.972758`。这说明该方向中确实存在被普通 Top-k mean 稀释的局部高偏离模板。
2. **BGL target 两个方向没有提升。** Thunderbird -> BGL 和 Spirit -> BGL 的 Precision、Recall、F1 均保持不变，同时 ROC-AUC 和 PR-AUC 略有下降。这说明 BGL target 的瓶颈不在 Top-k 内部加权，而更可能仍是异常窗口本身的分数可分性问题。
3. **BGL -> Thunderbird 有小幅提升。** rank-weighted 的 F1 从 `0.993403` 提升到 `0.994229`，但提升幅度很小，不应作为主要论据。
4. **Spirit -> Thunderbird 不适合替换为 rank-weighted。** rank-weighted 提高了 Precision，但 Recall 下降更多，F1 从 `0.994509` 降到 `0.993111`。
5. **score-weighted 和 softmax-weighted 没有超过 rank-weighted。** 在 Thunderbird -> Spirit 上二者分别为 `0.838198` 和 `0.836560`，均低于 rank-weighted；在 Thunderbird target 的强方向上也没有稳定优势。

### 30.7 当前判断

Top-k weighted level 值得作为 target-aware level 的增强实验保留，但不适合直接替换默认 Top-k mean。当前更稳妥的做法是：默认 `target_score_aggregation=mean`，同时保留 `target_scores_rank_weighted_level` 作为候选消融配置。

从六方向结果看，`rank_weighted_mean` 是三种加权方式中最值得继续跟进的一种。它对 Thunderbird -> Spirit 这类局部异常较明显的方向有实质提升，但对 BGL target 没有解决作用。因此后续若继续优化弱方向，BGL target 仍需要新的分数表达或评价校准思路，而不是继续只调 Top-k 聚合方式。


## 31. Target-aware level 特征范围实验

### 31.1 实验目的

BGL false negative 诊断显示，BGL 目标域的漏检窗口中，异常行很多但落在原 `selected_features` 内的覆盖率很低。因此本轮不改变 Granger edge 的建模特征，只放宽 target-aware level score 的打分特征范围，用来验证漏检是否来自 level 可见模板过窄。

本轮比较四种 level 特征范围：

```text
原 selected
Top-50 target features
Top-100 target features
全部 target features
```

Top-50/Top-100 按 target 训练集模板总次数排序，活跃窗口数和方差用于打破并列；level 的正常基线仍只用 target normal 窗口校准。

### 31.2 代码调整

新增 `granger.target_score_feature_scope` 配置，支持 `selected`、`top_50`、`top_100`、`all`。`selected_features` 仍用于 Granger VAR、residual、edge 和 edge-level fusion 的 edge 部分；新增 `target_score_features` 仅用于 target-aware level 的正常基线拟合与 level 打分。模型保存文件中也写入 `target_score_features`，便于复查 Top-50/Top-100/all 实际覆盖了哪些模板。

新增四个消融配置：

```text
configs/count_granger_ablations/target_scores_scope_selected.yaml
configs/count_granger_ablations/target_scores_scope_top50.yaml
configs/count_granger_ablations/target_scores_scope_top100.yaml
configs/count_granger_ablations/target_scores_scope_all.yaml
```

### 31.3 六方向结果

| source | target | scope | selected component | Precision | Recall | F1 |
|---|---|---|---|---:|---:|---:|
| Thunderbird | BGL | selected | level | 0.918782 | 0.775161 | 0.840883 |
| Thunderbird | BGL | Top-50 | level | 0.962472 | 0.933619 | 0.947826 |
| Thunderbird | BGL | Top-100 | level | 0.962472 | 0.933619 | 0.947826 |
| Thunderbird | BGL | all | level | 0.962472 | 0.933619 | 0.947826 |
| Spirit | BGL | selected | level | 0.915761 | 0.721627 | 0.807186 |
| Spirit | BGL | Top-50 | level | 0.930804 | 0.892934 | 0.911475 |
| Spirit | BGL | Top-100 | level | 0.931567 | 0.903640 | 0.917391 |
| Spirit | BGL | all | level | 0.933884 | 0.967880 | 0.950578 |
| Thunderbird | Spirit | selected | edge_level_fusion_level0p5 | 0.860934 | 0.957435 | 0.906624 |
| Thunderbird | Spirit | Top-50 | edge_level_fusion_level0p75 | 0.927681 | 0.950057 | 0.938735 |
| Thunderbird | Spirit | Top-100 | edge_level_fusion_level0p75 | 0.922152 | 0.958002 | 0.939736 |
| Thunderbird | Spirit | all | edge_level_fusion_level0p25 | 0.654695 | 0.951759 | 0.775760 |
| BGL | Spirit | selected | edge_level_fusion_level0p25 | 0.963043 | 0.887344 | 0.923645 |
| BGL | Spirit | Top-50 | edge_level_fusion_level0p5 | 0.909091 | 0.953462 | 0.930748 |
| BGL | Spirit | Top-100 | edge_level_fusion_level0p25 | 0.923549 | 0.939274 | 0.931345 |
| BGL | Spirit | all | edge_level_fusion_level0p25 | 0.908543 | 0.941544 | 0.924749 |
| BGL | Thunderbird | selected | level | 0.996145 | 0.992320 | 0.994229 |
| BGL | Thunderbird | Top-50 | level | 1.000000 | 0.991223 | 0.995592 |
| BGL | Thunderbird | Top-100 | level | 1.000000 | 0.991223 | 0.995592 |
| BGL | Thunderbird | all | level | 1.000000 | 0.989578 | 0.994762 |
| Spirit | Thunderbird | selected | level | 0.997785 | 0.988481 | 0.993111 |
| Spirit | Thunderbird | Top-50 | level | 1.000000 | 0.991223 | 0.995592 |
| Spirit | Thunderbird | Top-100 | level | 1.000000 | 0.991223 | 0.995592 |
| Spirit | Thunderbird | all | level | 0.999447 | 0.991223 | 0.995318 |

### 31.4 结果判断

1. **BGL target 的弱方向明显提升。** Thunderbird -> BGL 从 `0.840883` 提升到 `0.947826`；Spirit -> BGL 从 `0.807186` 提升到 `0.950578`。这基本验证了前面的 false negative 诊断：原 selected 范围过窄，导致许多 BGL 异常模板没有进入 level score 的观察范围。
2. **强方向没有被破坏。** BGL -> Thunderbird 和 Spirit -> Thunderbird 均从约 `0.993`/`0.994` 提升到约 `0.996`；BGL -> Spirit 也从 `0.923645` 小幅提升到 `0.931345`。
3. **all target features 不适合作为默认主线。** 它在 Spirit -> BGL 上最好，但在 Thunderbird -> Spirit 上 F1 从 `0.906624` 降到 `0.775760`。Spirit 目标域训练正常窗口只有 2 个，all 会让 level 基线受低频模板和不稳定模板影响，泛化风险明显更高。
4. **Top-50/Top-100 更适合纳入主候选。** 两者在六个方向上均不低于 selected，其中 Top-100 在 Thunderbird -> Spirit、BGL -> Spirit、Spirit -> BGL 上略优，Top-50 在 Thunderbird target 上与 Top-100 持平。当前更稳妥的主候选是 `selected + top_50 + top_100`，`all` 作为诊断或备用消融保留。

### 31.5 结果保存位置

六方向实验结果保存在：

```text
results/count_granger_level_feature_scope/20260814_135012_Thunderbird_to_BGL
results/count_granger_level_feature_scope/20260814_135130_Spirit1G_to_BGL
results/count_granger_level_feature_scope/20260814_135307_Thunderbird_to_Spirit1G
results/count_granger_level_feature_scope/20260814_135350_BGL_to_Spirit1G
results/count_granger_level_feature_scope/20260814_135458_BGL_to_Thunderbird
results/count_granger_level_feature_scope/20260814_135609_Spirit1G_to_Thunderbird
```

汇总表保存在：

```text
results/count_granger_level_feature_scope/summary_scope_comparison.csv
```


## 32. 多 scope level 单次主线实验

### 32.1 实验目的

上一轮分别运行 `selected`、`Top-50`、`Top-100`、`all target features` 后发现，Top-50/Top-100 能显著增强 BGL target 弱方向，同时不会破坏强方向；但 `all` 在 Thunderbird -> Spirit 上明显退化。因此本轮将 `selected/top_50/top_100` 纳入同一次运行的 validation-best 候选，不再通过 test 结果事后选择 scope。

### 32.2 代码调整

新增 `granger.target_score_feature_scopes`，可在一次 detector 中同时拟合多个 target-aware level baseline。当前主线配置为：

```text
selected
top_50
top_100
```

一次运行会输出以下候选组件：

```text
level_selected
level_top50
level_top100
edge_level_fusion_selected_level0p25 / 0p5 / 0p75
edge_level_fusion_top50_level0p25 / 0p5 / 0p75
edge_level_fusion_top100_level0p25 / 0p5 / 0p75
```

`target_scores.yaml` 已更新为当前主线配置；同时保留 `target_scores_multi_scope.yaml` 作为显式实验配置。

### 32.3 六方向结果

| source | target | selected component | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|
| Thunderbird | BGL | level_top50 | 0.962472 | 0.933619 | 0.947826 |
| Spirit | BGL | level_top100 | 0.931567 | 0.903640 | 0.917391 |
| Thunderbird | Spirit | edge_level_fusion_top50_level0p75 | 0.927681 | 0.950057 | 0.938735 |
| BGL | Spirit | edge_level_fusion_top50_level0p5 | 0.909091 | 0.953462 | 0.930748 |
| BGL | Thunderbird | level_top50 | 1.000000 | 0.991223 | 0.995592 |
| Spirit | Thunderbird | level_top50 | 1.000000 | 0.991223 | 0.995592 |

### 32.4 结果判断

1. **多 scope 单次主线达到了预期。** Thunderbird -> BGL 从原 selected 的 `0.840883` 提升到 `0.947826`；Spirit -> BGL 从 `0.807186` 提升到 `0.917391`。弱方向明显增强。
2. **强方向保持。** BGL -> Thunderbird 和 Spirit -> Thunderbird 均达到 `0.995592`，比原 selected 主线略高。
3. **edge + level 融合仍有价值。** Thunderbird -> Spirit 和 BGL -> Spirit 均选择了 edge-level fusion，而不是单独 level，说明融合候选应该保留在主线中。
4. **不纳入 all 是合理的。** Spirit -> BGL 的 all 曾达到 `0.950578`，但 Thunderbird -> Spirit 的 all 降到 `0.775760`。当前主线选择 `selected/top50/top100` 是更稳妥的折中。

### 32.5 结果保存位置

```text
results/count_granger_multi_scope_mainline/20260814_152733_Thunderbird_to_BGL
results/count_granger_multi_scope_mainline/20260814_152824_Spirit1G_to_BGL
results/count_granger_multi_scope_mainline/20260814_152856_Thunderbird_to_Spirit1G
results/count_granger_multi_scope_mainline/20260814_152923_BGL_to_Spirit1G
results/count_granger_multi_scope_mainline/20260814_152953_BGL_to_Thunderbird
results/count_granger_multi_scope_mainline/20260814_153023_Spirit1G_to_Thunderbird
```

汇总表：

```text
results/count_granger_multi_scope_mainline/summary_multi_scope_mainline.csv
```


## 33. 当前模型主线总结

### 33.1 总体定位

当前 Count-Granger 的代码主线可以概括为：

```text
Count-Granger edge/residual
        +
Target-aware multi-scope rank-weighted level
        +
Edge-level fusion candidates
        +
Validation-best score selection
```

也就是说，模型仍然保留原来的 Granger 动态关系建模，同时加入 target 自己的局部计数异常分数，并让验证集在不同分数组件中选择最适合当前 source-target 方向的检测信号。

### 33.2 Granger 主干

日志首先被转换为窗口级模板计数序列：

```text
raw logs
  ↓
log template / semantic template
  ↓
window count series
```

随后模型基于 source 和 target 的训练窗口进行跨域特征筛选，得到 `selected_features`。这些特征仍然是 Granger 主干的唯一建模特征集合：

```text
selected_features
  ↓
lagged count series
  ↓
Ridge VAR / regularized Granger model
  ↓
residual score
edge score
```

其中：

- `residual score` 表示当前窗口的预测误差是否异常；
- `edge score` 表示 Granger 边上的动态影响是否异常。

注意，Top-50/Top-100 只用于后面的 level 分数，不改变 Granger VAR、residual 和 edge 的建模特征。

### 33.3 Target-aware multi-scope level

当前主线新增 target-aware level，用于捕捉目标域自身的局部计数异常。它的含义是：以 target normal 窗口为基准，判断当前 target 窗口中模板计数水平是否偏离正常。

一次运行中同时拟合三个 level scope：

```text
selected
top_50 target features
top_100 target features
```

对应输出三个分数组件：

```text
level_selected
level_top50
level_top100
```

每个 scope 的 level 计算过程为：

```text
target normal windows
  ↓
拟合每个模板的正常计数水平
  ↓
current target window
  ↓
计算每个模板相对 target normal 的偏离
  ↓
取偏离最大的 top-k 个模板
  ↓
rank-weighted mean 聚合
  ↓
level score
```

当前默认聚合方式为：

```text
target_score_aggregation = rank_weighted_mean
```

这样做的原因是异常往往只体现在少数模板上，简单 mean 容易被大量正常模板稀释；rank-weighted mean 会让偏离最大的模板获得更高权重。

### 33.4 Edge-level fusion

当前主线还保留 edge + level 同次融合候选。模型会将 `edge` 分别与不同 scope 的 level 分数融合：

```text
edge + level_selected
edge + level_top50
edge + level_top100
```

每组融合尝试三个 level 权重：

```text
level weight = 0.25
level weight = 0.50
level weight = 0.75
```

融合前会基于 validation scores 做 robust normalization，避免 edge 和 level 因量纲不同而互相压制。输出候选形式例如：

```text
edge_level_fusion_selected_level0p25
edge_level_fusion_top50_level0p5
edge_level_fusion_top100_level0p75
```

这部分用于处理 edge 和 level 互补的方向，例如 Thunderbird -> Spirit 和 BGL -> Spirit。

### 33.5 Validation-best 最终选择

最终检测分数不是固定为某一个组件，而是由 validation-best 在同一次运行中选择。当前主线候选包括：

```text
score
residual
edge
level_selected
level_top50
level_top100
edge_level_fusion_selected_*
edge_level_fusion_top50_*
edge_level_fusion_top100_*
```

选择逻辑仍沿用当前检测配置中的 validation-best / F1-at-precision 机制。验证集选中某个 score component 和 threshold 后，再在 test set 上评估。

最近一次六方向主线结果中，各方向选中的组件为：

| source | target | selected component | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|
| Thunderbird | BGL | level_top50 | 0.962472 | 0.933619 | 0.947826 |
| Spirit | BGL | level_top100 | 0.931567 | 0.903640 | 0.917391 |
| Thunderbird | Spirit | edge_level_fusion_top50_level0p75 | 0.927681 | 0.950057 | 0.938735 |
| BGL | Spirit | edge_level_fusion_top50_level0p5 | 0.909091 | 0.953462 | 0.930748 |
| BGL | Thunderbird | level_top50 | 1.000000 | 0.991223 | 0.995592 |
| Spirit | Thunderbird | level_top50 | 1.000000 | 0.991223 | 0.995592 |

### 33.6 当前不纳入 all target features 的原因

`all target features` 仍作为消融和诊断配置保留，但不进入默认主线。原因是它的效果不稳定：

```text
Spirit -> BGL:
all F1 = 0.950578，优于 Top-100

Thunderbird -> Spirit:
all F1 = 0.775760，明显低于 selected/top50/top100
```

因此当前主线采用更稳妥的 `selected + top_50 + top_100`，不默认加入 all。这个选择符合当前目标：弱方向更强，强方向至少保持，同时避免单个方向的大幅退化。

### 33.7 当前代码主线配置

当前主线配置文件为：

```text
configs/count_granger_ablations/target_scores.yaml
```

显式 multi-scope 实验配置为：

```text
configs/count_granger_ablations/target_scores_multi_scope.yaml
```

最近一次主线结果保存于：

```text
results/count_granger_multi_scope_mainline/summary_multi_scope_mainline.csv
```

### 33.8 一句话总结

当前模型主线是：

```text
edge 看“关系是否异常”；
level 看“target 自己的模板数量是否异常”；
multi-scope 让 level 不再只看过窄的 selected features；
edge-level fusion 捕捉二者互补；
validation-best 决定当前方向最终相信哪个分数。
```


## 34. Target-aware level 不同 Top-K 范围消融

### 34.1 实验目的

上一轮主线采用 `selected/top50/top100` 三个 level scope。为了判断 Top-K 是否只是偶然选择，还是存在稳定有效区间，本轮加入更多 K 值进行消融：

```text
selected
top25
top50
top75
top100
top150
all
```

本轮仍采用单次 validation-best 选择，即同一次运行中同时提供不同 K 的 level 及 edge-level fusion 候选，由 validation set 选择最终 score component。

### 34.2 代码与配置

`count_granger_model.py` 已将 `target_score_feature_scope` 从固定 `top50/top100` 扩展为通用 Top-K 解析，支持：

```text
top_25 / top25
top_75 / top75
top_150 / top150
```

新增消融配置：

```text
configs/count_granger_ablations/target_scores_topk_sweep.yaml
```

### 34.3 六方向结果

| source | target | selected component | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|
| Thunderbird | BGL | level_top50 | 0.962472 | 0.933619 | 0.947826 |
| Spirit | BGL | level_all | 0.933884 | 0.967880 | 0.950578 |
| Thunderbird | Spirit | edge_level_fusion_top50_level0p75 | 0.927681 | 0.950057 | 0.938735 |
| BGL | Spirit | edge_level_fusion_top25_level0p5 | 0.918328 | 0.954030 | 0.935839 |
| BGL | Thunderbird | level_top25 | 1.000000 | 0.991223 | 0.995592 |
| Spirit | Thunderbird | level_top25 | 1.000000 | 0.991223 | 0.995592 |

### 34.4 与当前主线对比

当前主线为 `selected/top50/top100`。Top-K sweep 与当前主线的 F1 对比如下：

| source | target | 当前主线组件 | 当前主线 F1 | Top-K sweep 组件 | Top-K sweep F1 | Delta F1 |
|---|---|---|---:|---|---:|---:|
| Thunderbird | BGL | level_top50 | 0.947826 | level_top50 | 0.947826 | +0.000000 |
| Spirit | BGL | level_top100 | 0.917391 | level_all | 0.950578 | +0.033187 |
| Thunderbird | Spirit | edge_level_fusion_top50_level0p75 | 0.938735 | edge_level_fusion_top50_level0p75 | 0.938735 | +0.000000 |
| BGL | Spirit | edge_level_fusion_top50_level0p5 | 0.930748 | edge_level_fusion_top25_level0p5 | 0.935839 | +0.005091 |
| BGL | Thunderbird | level_top50 | 0.995592 | level_top25 | 0.995592 | +0.000000 |
| Spirit | Thunderbird | level_top50 | 0.995592 | level_top25 | 0.995592 | +0.000000 |

### 34.5 结果判断

1. **Top-K 有必要做，但不宜当作无边界调参。** 本轮证明 K 的选择会影响结果，尤其 BGL -> Spirit 从 Top-50 融合切到 Top-25 融合后，F1 从 `0.930748` 提升到 `0.935839`。
2. **Top-50 对 Thunderbird -> BGL 已经足够。** 加入 Top-25、Top-75、Top-150 和 all 后，validation 仍选择 `level_top50`，说明该方向在 Top-50 附近已经进入平台区。
3. **Spirit -> BGL 的最高值来自 all。** 该方向 F1 从当前主线的 `0.917391` 提升到 `0.950578`。但这不能直接说明 all 应进入默认主线，因为 all 曾在单独 scope 实验中使 Thunderbird -> Spirit 明显退化。更稳妥的做法是后续设计 normal-bin gate，只在 target normal bins 充足时允许 all 进入候选。
4. **Top-25 值得纳入下一版主线候选。** Top-25 在 BGL -> Spirit 上有小幅提升，在 Thunderbird target 两个方向与 Top-50 持平，未观察到负面影响。因此相比 all，Top-25 是更低风险的主线扩展。
5. **不是 K 越大越好。** 六方向被选中的 K 包括 Top-25、Top-50、Top-100 和 all，说明不同 target 的有效特征范围不同。继续盲目扩大 K 没有充分依据。

### 34.6 当前建议

建议下一步将主线候选从：

```text
selected / top50 / top100
```

扩展为：

```text
selected / top25 / top50 / top100
```

`all` 暂时不直接纳入默认主线。更稳妥的后续方案是增加一个 target normal bins gate：当 target normal bins 足够多时，才允许 `level_all` 参与 validation-best；否则仍限制在 Top-K 中等范围内。

### 34.7 结果保存位置

```text
results/count_granger_topk_sweep/summary_topk_sweep.csv
results/count_granger_topk_sweep/20260814_163041_Thunderbird_to_BGL
results/count_granger_topk_sweep/20260814_163137_Spirit1G_to_BGL
results/count_granger_topk_sweep/20260814_163227_Thunderbird_to_Spirit1G
results/count_granger_topk_sweep/20260814_163308_BGL_to_Spirit1G
results/count_granger_topk_sweep/20260814_163359_BGL_to_Thunderbird
results/count_granger_topk_sweep/20260814_163447_Spirit1G_to_Thunderbird
```

## 35. 固定 Top-25 与固定 Top-50 消融

### 35.1 实验目的

上一轮 Top-K sweep 表明，不同方向被选中的有效 level scope 并不完全一致。本轮进一步固定单一 K 值，分别只允许 `top25` 或 `top50` 参与 target-aware level 与 edge-level fusion 候选，用来判断是否有必要将 Top-K 固定为某个统一值。

本轮配置为：

```text
configs/count_granger_ablations/target_scores_fixed_top25.yaml
configs/count_granger_ablations/target_scores_fixed_top50.yaml
```

结果汇总保存为：

```text
results/count_granger_fixed_topk/summary_fixed_top25_top50.csv
```

### 35.2 六方向结果

| source | target | fixed scope | selected component | Precision | Recall | F1 |
|---|---|---|---|---:|---:|---:|
| Thunderbird | BGL | top25 | level | 0.928899 | 0.867238 | 0.897010 |
| Thunderbird | BGL | top50 | level | 0.962472 | 0.933619 | 0.947826 |
| Spirit | BGL | top25 | level | 0.921833 | 0.732334 | 0.816229 |
| Spirit | BGL | top50 | level | 0.930804 | 0.892934 | 0.911475 |
| Thunderbird | Spirit | top25 | edge_level_fusion_level0p5 | 0.900510 | 0.952894 | 0.925962 |
| Thunderbird | Spirit | top50 | edge_level_fusion_level0p75 | 0.927681 | 0.950057 | 0.938735 |
| BGL | Spirit | top25 | edge_level_fusion_level0p5 | 0.918328 | 0.954030 | 0.935839 |
| BGL | Spirit | top50 | edge_level_fusion_level0p5 | 0.909091 | 0.953462 | 0.930748 |
| BGL | Thunderbird | top25 | level | 1.000000 | 0.991223 | 0.995592 |
| BGL | Thunderbird | top50 | level | 1.000000 | 0.991223 | 0.995592 |
| Spirit | Thunderbird | top25 | level | 1.000000 | 0.991223 | 0.995592 |
| Spirit | Thunderbird | top50 | level | 1.000000 | 0.991223 | 0.995592 |

### 35.3 结果判断

1. **不建议把 Top-K 固定为 Top-25。** Thunderbird -> BGL 的 F1 从 `0.947826` 降到 `0.897010`，Spirit -> BGL 从当前多 scope 主线的 `0.917391` 降到 `0.816229`。Top-25 对 BGL -> Spirit 有小幅提升，但无法覆盖它在 BGL target 方向上的明显损失。
2. **固定 Top-50 可以作为强基线，但不应替代多 scope 主线。** Top-50 在 Thunderbird -> BGL、Thunderbird -> Spirit 以及两个 Thunderbird target 方向上与当前主线持平；但在 Spirit -> BGL 上低于当前主线的 `level_top100`，在 BGL -> Spirit 上低于固定 Top-25。
3. **固定单一 K 的方向适配性不足。** 六个方向中，Top-50 更稳，但不是所有方向最优；Top-25 对局部异常更敏感，但在 BGL target 上召回损失较大。
4. **当前更合理的主线仍是多 scope validation-best。** 即 `selected/top25/top50/top100` 作为候选，让 validation set 根据目标域表现选择最终 component。`all` 暂时不建议无条件纳入默认主线，需要后续 normal-bin gate 或其他约束后再测试。

### 35.4 与当前主线的关系

固定 Top-50 的结果说明 `top50` 是一个可靠的中心尺度，但实验不支持“固定 Top-50 就足够”。当前主线应保留多尺度候选，并将 Top-25 纳入候选集，而不是把 Top-K 固定为某一个值。

建议下一版默认候选为：

```text
selected / top25 / top50 / top100
```

不建议默认候选为：

```text
top25 only
top50 only
all without gate
```