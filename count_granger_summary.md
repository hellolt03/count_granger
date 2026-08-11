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
