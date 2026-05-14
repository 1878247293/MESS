# SAGEM 代码解读

本文按"一次完整运行"的实际执行顺序，把每个代码文件实现了什么、跟哪些文件协作，一行行串清楚。读完之后再去看任意一个文件，都能知道它在整体里站哪个位置。

> 阅读顺序建议：
> 1. **入口**（第 1 节）：先看启动脚本，知道整个系统从哪里跑起来。
> 2. **主流程**（第 2 节）：从读数据到出最终 F1，逐阶段拆。
> 3. **对比学习**（第 3 节）：可选的微调链路，独立成段。
> 4. **LLM 数据生成**（第 4 节）：为对比学习提供训练正例的离线工具。
> 5. **Web 前端**（第 5 节）：包在以上三条链路外面的 Gradio 工作台。
> 6. **工具与日志**（第 6 节）：跨流程复用的小组件。

---

## 1. 三个入口

整个项目有 **三个独立 CLI** 加 **一个 Web 前端**。

| 入口文件 | 干什么 | 典型调用 |
|---|---|---|
| `main.py` | 跑一次完整的实体匹配流水线，输出 P/R/F1 | `python main.py --data-name Geo` |
| `train_contrastive.py` | 独立训对比学习模型，把权重缓存到 `finetuned_models/` | `python train_contrastive.py --data-name Geo --cl-epochs 10` |
| `llm_data_generator/main.py` | 用大模型给指定数据集合成 `labeled_pairs.json` | `python -m llm_data_generator.main --dataset Geo` |
| `web/app.py` | 起 Gradio 把上面三个串成图形界面 | `python web/app.py` |

> 三个 CLI 都是独立可用的，没有强依赖。Web 前端只是用子进程调它们。

### `main.py`
主流程编排者。从命令行参数解析开始，按顺序跑：

1. `build_main_args()` 解析命令行 + 应用数据集默认参数（来自 `dataset_configs.py`）。
2. `init_logger()` 起 loguru，控制台只看关键阶段，文件落 DEBUG 全量。
3. `ResultLogger` 收集每阶段输入输出，最后一次性写 JSON / TXT。
4. **属性选择阶段** —— 调 `auto_selection` 或读 `selector_cache` 缓存。
5. **数据读取 + 编码** —— `read_all_tables` 重读（这次只读选中的属性），`SentenceTransformer.encode` 出向量。
6. **合并阶段** —— 走 `merge` / `merge_parallel` / `merge_with_smart_pairing` 之一，根据 `--use-smart-pairing` 和 `--run-in-parallel` 组合决定。
7. **评估** —— `evaluate_log_with_output` 算 P/R/F1 + 写出 4 类错误分组。
8. **落盘汇总** —— `result_logger.save_results()` 写 `results/<dataset>_<timestamp>.json/.txt`。

阶段切换处都包了 `ResourceMonitor.start/stop`，跟踪每段的 wall-clock + RAM/VRAM 峰值。

### `train_contrastive.py`
独立训练脚本。流程比主流程短：

1. 解析参数 → 加载 SentenceTransformer。
2. 调 `contrastive_finetune(model, args, cache_dir=...)`：
   - 命中缓存就直接灌权重返回；
   - 否则从 `llm_training_data/<dataset>/labeled_pairs.json` 读正对，跑 InfoNCE 训练，写缓存。
3. 打训练摘要。

**与主流程的关系**：训完的模型权重写到 `finetuned_models/<dataset>_..._supervised/` 下，下次 `main.py` 跑同样数据集时，可以通过 `--lm-model-or-path` 指过去。

### `llm_data_generator/main.py`
4 阶段数据生成 CLI：

1. **Stage 1** — `sampler.load_tables` + `sampler.sample_records` 读 `data/<dataset>/table_*.csv` 各采样若干行。
2. **Stage 2** — `analyzer.analyze_dataset` 把样本喂给 LLM，让它吐一份 Markdown 规范（字段格式 / 差异分布 / 噪声类型）。结果落 `analysis_cache_<model>.json`，下次可以 `--reuse-analysis` 复用。
3. **Stage 3** — `generator.generate_entity_groups` 按规范多线程批量生成 entity groups + variants。
4. **Stage 4** — `formatter.build_pairs` 组内 variant 两两配对，写到 `llm_training_data/<dataset>/labeled_pairs.json`。

`token_tracker.TokenTracker` 全程累计每个阶段的 prompt / completion token 用量。

### `web/app.py`
Gradio 前端。`CUSTOM_CSS` / `HERO_HTML` / `FOOTER_HTML` 三块构成蓝白系视觉。4 个 tab：
- **主流程** → `tabs/tab_main_flow.py`
- **对比学习** → `tabs/tab_contrastive.py`
- **数据生成** → `tabs/tab_llm_gen.py`
- **结果分析** → `tabs/tab_results.py`

每个 tab 实际运行靠 `web/runner.py` 起子进程跑对应 CLI，stdout 实时回流到日志框。

---

## 2. 主流程链路（`main.py` 一次运行）

```
            数据                                           评估
table_*.csv ─► read_all_tables ─► auto_selection ─► read_all_tables(再读一次，只读选中列)
                                       │                                  │
                                       ▼                                  ▼
                              selector_cache 写/读      textify_table → SentenceTransformer.encode
                                                                          │
                                                                          ▼
                              ┌──────────────────────────────────────────┘
                              ▼
                          (可选) SmartTablePairing.get_smart_pairing
                              │
                              ▼
              merge / merge_parallel / merge_with_smart_pairing / merge_parallel_with_smart_pairing
                              │
                              ▼
                       Table.get_tuples()
                              │
                              ▼
                  evaluate_log_with_output  → P / R / F1 + 错误分类
                              │
                              ▼
                  ResultLogger.save_results → results/ JSON & TXT
```

### 2.1 数据层 — `src/data_chuli/`

#### `data.py`
- **`Table`** dataclass：`idx`（表号字符串）/ `tids`（实体在全局向量数组里的下标列表）/ `tuple_ids`（每个 tid 属于哪个分组）。
  - `get_tuples()` 把 `(tid, tuple_id)` 按 `tuple_id` 聚合，输出形如 `[(tid, tid, ...), ...]` 的预测分组（≥2 个成员才算）。
- **`read_table` / `read_all_tables`**：读 `table_0.csv table_1.csv ...`。支持 `selected_attrs` 只读特定列、`sample_rate` 大表下采样（采样时 `tid` 会重新映射成 `0..n` 连续整数，下游靠 `tid` 直接做数组下标）。
- **`read_ground_truth`**：读 `ground_truth.txt`，每行一个 tuple。
- **`textify_table`**：表的每一行拼成一句字符串（除 `tid` 列），用于编码。

#### `dataset_configs.py`
按数据集存"最优超参 + 描述 + F1"的注册表。`main.py` 启动时通过 `args.use_dataset_config=True` 自动套——只覆盖那些用户没在命令行显式给的字段。注册表里目前有 Geo / Geo+CL / Geo+HardNeg / Music / Shopee 等组合。

### 2.2 属性选择 — `src/core/selector.py` & `selector_cache.py`

属性选择的目的：把那些**对实体身份没影响的列**（如随机 ID）剔掉，只保留真正能区分实体的属性。

#### `selector.py` · `auto_selection`
1. 把所有表 concat 到一起，按 `selection_rate` 下采样。
2. 整张表 `textify_table` 后编码出 `embeddings_before`。
3. 对每个非 `tid` 列：
   - 把这一列**整列打乱**，重新 textify + 编码，得 `embeddings_after`。
   - 算逐行 cosine 相似度均值 `mean_sim`。
4. `mean_sim ≤ col_sim_threshold`（γ，默认 0.8）说明这列被打乱后行向量变了很多 ⇒ **对身份重要**，保留。

#### `selector_cache.py` · `AttributeSelectionCacheManager`
属性选择跑一次几十秒到几分钟，所以缓存 key 是 `(dataset, γ, rate, model_name)`，命中直接读 `cache/attribute_selection/*.json`。**model_name 必须进 key**——不同模型的相似度分布完全不一样，曾经共享缓存吃过亏。

### 2.3 编码 — `main.py` 直接用 `SentenceTransformer.encode`
不在单独文件里。流程：
- 重读 `read_all_tables(selected_attrs=…)`，这次只读选中的列。
- 实例化 `SentenceTransformer(args.lm_model_or_path, local_files_only=True)`。
- 可选 `_wrap_data_parallel` 包一层 `DataParallel`（要 `--multi-gpu`）。
- 每张表 `textify_table` + `encode`，最后 `np.concatenate` 成 `all_embeddings`。

### 2.4 表配对 — `src/core/smart_table_pairing.py`（可选）

控制开关：`--use-smart-pairing`。

`SmartTablePairing` 只有一种策略：**按表语义中心相似度从高到低做贪心配对**。
- `compute_table_centers`：每张表所有 embedding 取均值再 L2 归一化。
- `compute_similarity_matrix`：表中心两两 cosine。
- `greedy_pairing_by_similarity`：所有表对按 sim 降序排，从高到低顺次匹配，已配过的跳过。剩 1 张就是 unpaired。

不开 smart pairing 就走 `utils.shuffle` 随机配。

### 2.5 分层合并 — `src/core/merger.py`

合并是个 **层次化两两合并** 过程：N 张表 → N/2 张 → N/4 张 → … → 1 张。每一层都按当前层的表数挑配对方式：

- `merge` —— 串行 + 随机配对（基线）
- `merge_parallel` —— joblib 并行 + 随机配对
- `merge_with_smart_pairing` —— 串行 + SmartTablePairing
- `merge_parallel_with_smart_pairing` —— 并行 + SmartTablePairing

#### 关键函数 `merge_ij(table_i, table_j, all_embeddings, args)`
两张表合一张：
1. 各取每个 tuple 的 embedding 均值（`get_table_embeddings`）。
2. **双向 mutual KNN**：
   - 默认走 `search_ij` 老路子：双向各跑一次 HNSW，取交集。
   - `--use-efficient-matching` 切到 `efficient_matcher.efficient_mutual_search`（同语义，numpy 向量化更紧）。
3. 距离 `≤ args.min_dis` 的 `(i, j)` 才保留。
4. 用 set 拼新表：匹配上的 tuple 合并，没匹配上的各自延续 tuple_id。

> 这版本没有剪枝阶段，合完直接当作最终预测。

### 2.6 高效匹配 — `src/core/efficient_matcher.py`
两个实现：
- `efficient_mutual_search` —— 跟老路子完全等价的紧凑实现，hnswlib + numpy mask。
- `efficient_mutual_search_v2` —— 单向 + 反向校验。两表大小差很多时只对命中过的 j 做反向搜，省事。

### 2.7 评估 — `src/utils/metrics.py`
- `evaluate_f1` —— 集合层面 tuple 完全匹配。
- `evaluate_pair_f1` —— tuple 拆成两两组合后做 F1（衡量"对的配上没"）。
- `evaluate_f1_with_output` —— 在算 F1 的同时把所有预测分组写进 TXT，按 **完全正确 / 漏匹配（真子集）/ 误匹配-超集（包对了但带杂质）/ 误匹配-其他** 四类落盘，方便人工排查。

### 2.8 结果记录 — `src/utils/result_logger.py`
`ResultLogger` 把整次运行的输入输出收集起来：
- `RunInfo / AttributeSelectionResult / DataLoadingResult / EncodingResult / MergingResult / EvaluationResult / ContrastiveLearningResult / ...`
- 每个 dataclass 对应主流程一个阶段。
- `save_results()` 一次性写 `results/<dataset>_<timestamp>.json` + `.txt` 两份。
- `results/` 同时是 Web "结果分析" 标签页的数据源。

---

## 3. 对比学习（可选）— `src/training/contrastive_learning.py`

主流程 **不内嵌** 对比学习。需要先单独跑 `train_contrastive.py` 把权重训出来缓存，再让 `main.py` 通过 `--lm-model-or-path` 指过去。

### 数据来源
`llm_training_data/<dataset>/labeled_pairs.json`，结构：
```json
{
  "metadata": {...},
  "entity_groups": [...],
  "pairs": [["text_a", "text_b", 1], ...]
}
```
每对都是正例（同一 entity group 的两个 variant），负例由 batch 内其它样本提供。

### `SupervisedPairDataset`
读 `pairs` 字段（也兼容老格式 `triplets` 中 `label==1` 的部分）。每个 `__getitem__` 返回 `(text_a, text_b)`。

### `ContrastiveLearner`
训练壳：
- `compute_contrastive_loss` —— InfoNCE：两个视图 normalize 后做 `[B, B]` 相似度矩阵，对角线就是正对，双向 cross-entropy 取平均。
- `train_epoch` —— 一个 epoch：tokenize → forward 拿 `sentence_embedding` → loss → backward。
- `train(data_path, data_name, epochs, batch_size, lr, warmup_steps)` —— 整个训练循环。每个 epoch 落 loss 到 `training_logs/<dataset>_loss.json`，Web 端"对比学习" tab 实时画曲线。

### `contrastive_finetune(model, args, cache_dir)`
对外入口（被 `train_contrastive.py` 调）：
1. 按 `(dataset, model_type, cl_epochs, cl_lr, cl_temperature)` 拼缓存文件名（保留 `_supervised` 后缀以兼容历史缓存目录）。
2. 命中缓存 + `--force-retrain=False` ⇒ 直接 `_load_cached_weights` 灌权重返回。
3. 没命中就构造 `ContrastiveLearner` 训练完再 `finetuned_model.save()`。

---

## 4. LLM 数据生成 — `llm_data_generator/`

这是个 **独立的离线工具**。目的：在拿不到真实标注的情况下，让大模型按数据集的分布生成 entity groups（同一实体的多种格式表示），再把组内 variant 两两配成正对，供对比学习消费。

### `main.py` 编排
4 阶段管线（见上方第 1 节）。`reuse_analysis` 开关四种组合：开/关 × 缓存有/没，分支都写在 main.py 里。

### Stage 1 · `sampler.py`
- `load_tables` —— 按 `table_*.csv` 数字顺序读完。
- `sample_records` —— 每张表随机抽 `sample_size` 行（默认 50）。
- `format_samples_for_prompt` —— 把抽样结果拼成 Markdown 喂给 LLM。

### Stage 2 · `analyzer.py`
- `ANALYSIS_SYSTEM_PROMPT` / `ANALYSIS_USER_TEMPLATE` —— 长 prompt，要求 LLM 输出一份 Markdown 规范：每张表每列的格式 / 表间差异分布 / 噪声类型 / 特别注意。
- `analyze_dataset` —— 只调一次 LLM，剥 ``` ` ``` 代码块包装，写入 `analysis_cache_<model>.json`。

### Stage 3 · `generator.py`
- `GENERATE_SYSTEM_PROMPT` —— 列了 10 条硬约束（不许占位符、不许跨实体、字段对齐、表间格式差异、大小写差异、语言一致性、虚构实体要符合常识等）。
- `GENERATE_USER_TEMPLATE` —— 把 Stage 2 的规范 + 已生成实体名（做去重）+ 多样性提示词拼进去。
- `generate_entity_groups(client, config, analysis, num_groups, batch_size, max_workers)` —— 多线程并发：
  - 每批让 LLM 一次性吐 `batch_size` 个 entity group。
  - `_validate_entity` 过黑名单（占位符 / null / 空值过多）。
  - 主线程拿 `lock` 做 canonical 字段去重。
  - 连续 30 批没产出新实体就提前停。
- `DIVERSITY_HINTS` —— 多样性提示词池，每批轮一个让 LLM 别老盯着同一类实体。

### Stage 4 · `formatter.py`
- `format_record_text` —— 单个 variant 拼成 `"name: X, latitude: Y, ..."` 这种格式。
- `build_positive_pairs` —— 组内任两个 variant 配成 `[text_a, text_b, 1]`（label 永远是 1）。
- `save_output` —— 写 `llm_training_data/<dataset>/<dataset>_<model>.json`。

> 注意：这里写出的文件叫 `<dataset>_<model>.json`；`SupervisedPairDataset` 读的是 `labeled_pairs.json`，所以训练前需要把生成结果重命名/软链。

### `config.py`
- `DatasetConfig` —— 单数据集 schema：列名、文本字段、canonical 字段、组级附加字段、variant 数。
- `DATASET_REGISTRY` —— 已知数据集（Geo / Person / Shopee / Music-2000 / Music-200 等）的 DatasetConfig 实例。
- `GeneratorConfig` —— 生成器运行参数（backend、模型名、温度、batch、并发数等）。
- `auto_detect_config` —— 没注册的数据集按列名硬塞。

### LLM 客户端
- `api_client.ApiClient` —— OpenAI 兼容 API（带 API Key 鉴权），SSE 流式优先。
- `ollama_client.OllamaClient` —— Ollama REST API。
- `vllm_client.VllmClient` —— vLLM OpenAI 兼容路径。
- 三者接口一致（`chat / chat_json / check_connection / list_models`），由 `main.py` 根据 `--backend` 选其一。

### `token_tracker.py`
`TokenTracker` 加锁，按阶段（analysis / generation）累计 prompt / completion token；`global_tracker` 跨数据集累积。报告通过 `report()` 打表 + `save()` 写 JSON。

---

## 5. Web 前端 — `web/`

### `app.py`
Gradio 入口。`create_app()` 拼出 4 个 tab，挂自定义 CSS（蓝白系，sticky header）和 HERO/FOOTER 文案。`python web/app.py` 后台跑在 `127.0.0.1:7860`。

### `runner.py` · `ProcessRunner`
**所有 tab 都通过它跑后端**——不直接 import，而是 `subprocess.Popen([python, "main.py", ...])`。优点：
- 子进程崩了不影响 Gradio 主进程。
- stdout 行流式回吐给前端日志框。
- `stop()` 通过 `os.killpg` 一次性把进程组全杀掉。

三个方法：`run_main_flow / run_contrastive / run_llm_gen`，每个对应一个 CLI 入口。

### `utils.py`
前端杂活：
- `collapse_progress_lines` —— 同一条 tqdm 反复打印只留最后一帧。
- `parse_metrics_from_log` —— 从日志里抓 `[none] P=... R=... F1=...` 末次。
- `parse_progress_from_log` —— 抓 `[phase X/Y]` 或末尾百分比。
- `parse_loss_from_log` —— 对比学习 tab 用，抓 `Epoch i/N - Loss: X` 序列。
- `scan_datasets` —— 扫 `data/` 列出可用数据集名。
- `get_dataset_stats / get_preset` —— 数据集摘要与"推荐参数"按钮。
- `load_result_files / load_result_detail` —— 结果 tab 用，读 `results/*.json`。

### 4 个 tab
- **`tab_main_flow.py`** —— 主流程：左边选数据集 / 模型 / γ / 距离阈值 / 选采样率 / 智能配对开关；右边实时日志 + P/R/F1 三块卡片。`run_btn.click → runner.run_main_flow(...)` 把字段拼成 CLI 参数交给子进程。
- **`tab_contrastive.py`** —— 对比学习：左边训练超参（epochs / batch / lr / temperature），右边 loss 折线 + 日志。`run_btn.click → runner.run_contrastive(...)`。
- **`tab_llm_gen.py`** —— LLM 数据生成：左边数据集 / 后端选择（Ollama/vLLM/API）/ 目标实体数等，右边日志 + token 使用统计。
- **`tab_results.py`** —— 结果分析：扫 `results/` 列出历次运行，可以挑一条看完整阶段拆解 + 各类错误分组。

---

## 6. 跨流程小工具 — `src/utils/`

| 文件 | 角色 |
|---|---|
| `args.py` | `MainArgs` dataclass + `tyro.cli` 解析；`build_main_args` 还做：模型类型 → 本地路径映射、HF cache 风格目录自动定位、数据集默认参数套用、模型推荐 batch/seq_length 自动调整。 |
| `log.py` | `init_logger` 配 loguru：终端 INFO+（且屏蔽 API 噪音），文件 DEBUG 全量；`log` / `log_args` / `log_time` 都是薄包装。 |
| `timer.py` | `Timer` 一个最小的 start/stop 计时器。 |
| `resource_monitor.py` | `ResourceMonitor` 后台线程定时采样进程 RSS + `torch.cuda.max_memory_reserved`，stop 后返回 `ResourceUsage`；`ensure_stage_metrics / update_stage_metrics` 维护一个跨阶段的资源累积字典。 |
| `metrics.py` | F1 / pair F1 评估 + 错误分组写出（见 2.7）。 |
| `utils.py` | `knn_search`（hnswlib 包装）、`shuffle`（带 seed）、`element_wise_cosine_sim`。 |
| `result_logger.py` | 整次运行的 JSON/TXT 汇总（见 2.8）。 |

---

## 附：典型运行命令

```bash
# A. 仅跑主流程（默认参数，Geo 数据集）
python main.py --data-name Geo

# B. 主流程 + 智能配对 + 并行
python main.py --data-name Geo --use-smart-pairing --run-in-parallel

# C. 离线生成对比学习训练数据（Ollama 后端）
python -m llm_data_generator.main --dataset Geo --num-entities 200

# D. 用生成好的正对训对比学习
python train_contrastive.py --data-name Geo --cl-epochs 10

# E. 用训好的模型再跑主流程
python main.py --data-name Geo --lm-model-or-path finetuned_models/Geo_cl_epochs10_lr1e-05_temp0.07_supervised

# F. 起 Web 工作台
python web/app.py
```

## 附：目录速查

```
PathCL-EM/
├── main.py                              # 主流程入口
├── train_contrastive.py                 # 对比学习训练入口
├── web/app.py                           # Gradio 前端入口
├── src/
│   ├── core/
│   │   ├── selector.py                  # 属性选择
│   │   ├── selector_cache.py            # 属性选择缓存
│   │   ├── merger.py                    # 层次合并（4 个版本）
│   │   ├── smart_table_pairing.py       # 语义相似度贪心配对
│   │   └── efficient_matcher.py         # 高效互 KNN
│   ├── data_chuli/
│   │   ├── data.py                      # Table / CSV 读 / textify
│   │   └── dataset_configs.py           # 各数据集最优参数
│   ├── training/
│   │   └── contrastive_learning.py      # 监督式对比学习
│   └── utils/
│       ├── args.py                      # CLI 参数 + 自动配置
│       ├── log.py                       # loguru 包装
│       ├── timer.py                     # 计时器
│       ├── resource_monitor.py          # RAM/VRAM 采样
│       ├── metrics.py                   # F1 + 错误分组
│       ├── result_logger.py             # 结果聚合落盘
│       └── utils.py                     # KNN / shuffle / 余弦
├── llm_data_generator/
│   ├── main.py                          # 4 阶段编排
│   ├── sampler.py                       # Stage 1 采样
│   ├── analyzer.py                      # Stage 2 LLM 出规范
│   ├── generator.py                     # Stage 3 LLM 出 entity groups
│   ├── formatter.py                     # Stage 4 拼正对
│   ├── config.py                        # schema + 注册表
│   ├── api_client.py                    # OpenAI 兼容后端
│   ├── ollama_client.py                 # Ollama 后端
│   ├── vllm_client.py                   # vLLM 后端
│   └── token_tracker.py                 # token 计数
└── web/
    ├── runner.py                        # subprocess 包装
    ├── utils.py                         # 日志解析等
    └── tabs/
        ├── tab_main_flow.py             # 主流程 tab
        ├── tab_contrastive.py           # 对比学习 tab
        ├── tab_llm_gen.py               # 数据生成 tab
        └── tab_results.py               # 结果浏览 tab
```
