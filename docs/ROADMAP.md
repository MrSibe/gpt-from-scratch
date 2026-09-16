# 模型、训练与推理优化学习路线

## 目标与边界

以当前字符级 GPT 为可读、可测试的基线，逐步理解：

1. **模型架构**：Transformer / GPT 的基本机制，以及 RMSNorm、RoPE、SwiGLU、GQA 等类 LLaMA 组件。
2. **训练质量**：优化器、学习率、初始化和数据如何影响收敛。
3. **训练效率**：混合精度、高效 attention、编译和显存管理。
4. **推理效率**：prefill、decode、KV cache、批处理和量化。
5. **完整语言模型流程**：子词 tokenizer、预训练、SFT，以及 MiniMind 风格的学习项目。

当前代码已经是 GPT 风格的 decoder-only Transformer，不需要先实现 encoder-decoder 才能继续。
可以先学习原始 Transformer 的 encoder、decoder 和 cross-attention 的区别；机器翻译实现作为可选支线。
本路线不是逐行复刻某个版本的 LLaMA 或 MiniMind，也不以尽快堆齐功能为目标。

硬件起点是单张 RTX 4060 Laptop GPU。优先做短小、可解释的单卡实验，再决定是否扩大规模。
以下未勾选项目是规划，不代表当前已实现。

## 实验原则

- **一次只改变一个主要因素。** 先提出假设，跑对照实验，再决定是否保留。
- 保留手写实现作为参考；优化实现通过配置切换，不要马上删除 baseline。
- 分清“质量改进”和“计算加速”：loss 更低不等于更快，吞吐更高也不等于同时间内效果更好。
- 质量实验统一数据、tokenizer、验证集、训练 token 预算和有效 batch；架构实验记录参数量与计算量差异。
- 性能实验固定设备、shape、dtype、batch、上下文长度；记录软件版本、实际后端和是否包含编译时间。
- 性能结果重复测量，报告中位数和波动；小幅质量差异用多个 seed 验证，不凭一次结果下结论。
- 每个阶段先用小配置验正确性，再上 GPU 测性能，最后才做较长训练。
- 不把小语料上的结论直接推广到大模型，也不要求每项主流技术在当前规模上都产生收益。

## 路线总览

```text
P0 闭环与正确性
  ↓
P1 最小实验记录与可信 baseline
  ↓
P2 手写 attention ↔ SDPA：第一个性能对照实验
  ↓
P3 训练质量与训练效率（两条支线，逐项实验）
  ↓
P4 GPT → 类 LLaMA 的架构替换
  ↓
P5 KV cache 与推理优化
  ↓
P6 tokenizer / 数据 / SFT / LoRA 等扩展
```

P2 提前安排，是因为代码改动集中，适合练习完整实验流程。
它不是训练策略的前置知识；P3 中的学习率实验也可以提前做。
KV cache 也可提前用现有 GPT 实现，但必须先明确绝对位置编码和窗口截断的语义。

## P0：闭环与正确性

### 已完成

- [x] 配置以字典保存，生成端能够重建模型。
- [x] 新 checkpoint 使用 `weights_only=True` 加载。
- [x] `train.py` 提供 `main()` 入口，导入不触发下载或训练。
- [x] CLI 支持小模型、短训练和指定 CPU，方便快速验证。
- [x] 最后一次参数更新后执行评估，step 表示已完成的更新次数。
- [x] 测试前向/反向、因果性、小 batch 过拟合与保存加载一致性。
- [x] 用本地临时语料测试训练→保存→生成闭环。

### 验收与边界

```bash
uv run python -m unittest discover -s tests -v
```

测试无需 GPU 或网络。闭环通过只说明代码流程正确，不代表生成质量已经达标。
当前 `ckpt.pt` 是最佳验证模型，不是完整续训状态；旧配置对象格式的兼容说明见 [README](../README.md#checkpoint-约定)。

## P1：先建立最小实验记录

**问题：如何判断下一次改动真的更好，而不是训练数据或计时方式变了？**

### 实现顺序

- [ ] 为每个实验创建独立目录，避免覆盖 checkpoint。
- [ ] 保存 `config.json`：模型/训练配置、seed、数据标识或哈希、代码版本及工作区是否有修改、软硬件信息。
- [ ] 保存 `metrics.csv`，使用标准库即可，不急着引入 pandas。
- [ ] 固定验证窗口，或使用独立评估 RNG；评估不能消耗训练使用的随机数流。
- [ ] 增加简单 Matplotlib 脚本，对比多个实验。
- [ ] 分离 `best.pt` 与 `last.pt`；后者保存 optimizer、step、best loss、RNG，以及引入后的 scheduler/scaler 状态。
- [ ] 增加续训测试：连续训练与保存后恢复训练，在受控 CPU 环境下应保持一致。

建议目录：

```text
runs/<experiment-id>/
  config.json
  metrics.csv
  best.pt
  last.pt
  plots/
```

这是未来目录约定；实现时也应将 `runs/` 中的大型实验产物加入忽略规则，不提交权重和语料。

### 最小指标

| 指标 | 用途 |
| --- | --- |
| step / tokens_seen | 标明优化器更新次数和累计训练数据量 |
| train_loss / val_loss | 观察收敛与泛化 |
| lr / grad_norm | 观察调度和训练稳定性；范数注明是否为裁剪前 |
| train_tokens_per_sec | 衡量纯训练吞吐，明确是否包括数据传输 |
| peak_memory_mb | 记录测量区间 CUDA 峰值 allocated 显存，必要时另记 reserved |
| train_time_s / eval_time_s / wall_time_s | 区分纯训练时间、验证开销和端到端时间 |

首批图表：

1. val loss vs tokens_seen：相同数据预算下的学习效果。
2. val loss vs 累计训练时间：相同计算时间下的学习效果；另外记录端到端耗时。
3. 吞吐和峰值显存对比：速度与资源成本。

### 为什么先 CSV + Matplotlib

CSV 是可重用的原始记录，Matplotlib 便于离线比较。TensorBoard 适合实时观察，等确实需要远程看曲线或分析梯度分布时再接入；它不是实验体系的前置条件。
后续可以同时写 CSV 和 TensorBoard，但只保留一套指标计算逻辑。

### 计时与评估注意事项

- 当前打印的 `elapsed` 是包含评估、保存等操作的累计时间，不能直接作为训练吞吐。
- CUDA 异步执行，专门 benchmark 时要 warmup，并使用 CUDA events 或在测量边界同步。
- 编译耗时和稳态吞吐分别报告，不在每个训练操作之间插入同步。
- 验证和训练显存分开测，按测量区间重置峰值统计。
- 当前每次评估默认跑 200 个训练 batch 和 200 个验证 batch，调试时先降低 `--eval-iters`，正式比较再统一。
- 相同 seed 并不自动保证不同设备、不同 PyTorch 版本的结果完全一致。

**验收：能复跑 baseline；仅改变评估频率，不会改变后续训练 batch 和 dropout 随机序列；能从日志重画对比图。**

## P2：第一个优化实验——SDPA

**问题：保持注意力数学定义不变，更换实现能节省多少时间和显存？**

- [ ] 保留手写 attention，增加 SDPA 开关。
- [ ] 使用 `torch.nn.functional.scaled_dot_product_attention`，正确设置因果性。
- [ ] 训练时传入 attention dropout；评估时显式传入 `dropout_p=0.0`，不要以为 SDPA 会自动读取 `model.training`。
- [ ] 在 dropout 关闭时，对比输出、输入梯度和参数梯度；低精度使用合理容差。
- [ ] 测量不同 batch、上下文长度和 dtype 下的训练吞吐、峰值显存。
- [ ] 记录实际使用的 attention 后端；SDPA 不保证总是选中 FlashAttention。

**验收：因果性与数值测试通过，产出一张对比表，说明收益及没有收益的配置。**

## P3：训练优化

### A. 训练质量：让相同数据预算更有效

保持架构不变，依次实验：

- [ ] AdamW 参数分组：明确哪些参数衰减，bias 和归一化参数通常不衰减。
- [ ] warmup + cosine 调度：先画出学习率曲线，确认 warmup 和末尾行为。
- [ ] 梯度裁剪：记录裁剪前范数和发生裁剪的频率。
- [ ] 显式初始化：研究线性层、embedding 和残差投影的初始化尺度。
- [ ] 权重共享支线：比较 embedding / lm_head 权重绑定后的参数量和验证效果。

**验收：每项都有固定预算的对照曲线；保留无收益结果，不把所有改动一次性打包。**

### B. 训练效率：更快或更省显存

- [ ] BF16 autocast：先检查设备支持；FP16 是另一条实验路径，通常需 GradScaler。
- [ ] 梯度累积：定义有效 batch = micro-batch × accumulation steps × world size。
- [ ] 正确缩放累积 loss，明确不足一个累积周期时的处理；调度按优化器更新步推进。
- [ ] `torch.compile`：单列冷启动成本和稳态表现，小模型不保证受益。
- [ ] Activation checkpointing：比较额外计算成本和显存收益。
- [ ] 用 profiler 定位瓶颈，再考虑数据采样、CPU→GPU 传输、融合算子或优化器实现。

注意：

- 梯度累积是容量工具，不是天然加速器；比较时保持有效 batch 不变。
- 使用混合精度和裁剪时，正确处理梯度缩放，不能对仍处于缩放状态的梯度直接按原阈值裁剪。
- 累积训练和直接大 batch 在存在 dropout、不同 kernel 时，不要求逐位一致。
- 当前语料较小，不能凭大数据训练经验就先引入复杂 DataLoader；用实际测量决定。

**验收：能解释每项改动是提高吞吐、减少显存还是改变训练行为，并展示代价。**

## P4：GPT → 类 LLaMA

保持字符词表和语料不变，逐个增加可配置组件：

| 顺序 | 替换 | 重点与验证 |
| --- | --- | --- |
| 1 | LayerNorm → RMSNorm | 均值/尺度处理、数值稳定性、参数差异 |
| 2 | 绝对位置 embedding → RoPE | Q/K 旋转、位置索引、旋转维度和因果性 |
| 3 | GELU FFN → SwiGLU | 门控机制、三次投影和 FFN 宽度 |
| 4 | MHA → GQA | query / KV 头数约束、KV 共享及 cache 容量 |

- [ ] 每一步补组件测试，再跑固定预算对照。
- [ ] 为最终类 LLaMA 组合单独记录配置；组合效果不能简单视为单项收益之和。
- [ ] 保持参数预算尽量接近，无法一致时明确报告差异。

公平性提醒：

- SwiGLU 有三个主要投影，不能不加说明地沿用原来的 `4 × hidden_size` 中间宽度。
  忽略 bias 时，约 `8/3 × hidden_size` 的宽度可与普通 4 倍宽 FFN 接近，再按实现需要取整。
- GQA 主要权衡模型质量与 KV cache 成本，不保证验证 loss 更低。
- 加上 RoPE 不等于模型自动具备良好的长上下文外推能力。
- 不要在同一次架构实验里同时换 tokenizer 和语料。

**验收：能解释组件公式、张量形状、参数量变化，并用实验说明质量和效率的权衡。**

## P5：推理优化

**首要目标：实现正确的 KV cache，而不是先接入复杂推理框架。**

- [ ] 给无 cache 生成建立 benchmark，固定 prompt 长度、输出长度、batch 和 dtype。
- [ ] 区分 prefill 与逐 token decode，为每层维护 KV cache。
- [ ] 正确处理位置偏移；测试单 token decode 和带历史 cache 的多 token 输入。
- [ ] 测试 cache 与无 cache 的逐位置 logits 一致性，而不是只看生成文本。
- [ ] 正确性通过后，再研究预分配 cache、批量推理和不同长度请求。
- [ ] 最后实验量化：明确量化对象、格式、kernel 支持及质量回归评估。

### 必须测量

- 首 token 延迟，并明确是否包含 tokenization、采样和数据传输。
- prefill 时间 / 吞吐。
- decode tokens/s 和每 token 延迟。
- 总生成耗时、峰值显存及 KV cache 大小。
- 不同上下文长度和 batch size 的结果。

### 两个正确性陷阱

1. 当前 GPT 截断窗口后会重新从零分配绝对位置编号。超过上下文窗口后，简单删除旧 KV 不能保证与无 cache 实现等价。第一版可以明确只支持总长度不超过窗口；不要把滑动窗口作为隐式行为。
2. 带历史 cache 时，Q 和 K 长度可能不同，因果 mask 需要考虑位置偏移。不能直接沿用方形 attention 的假设；用 logits 对照测试确认。

量化也不保证更快：小模型可能受 kernel、数据搬运或启动开销影响。
PagedAttention、连续批处理和自定义 CUDA/Triton kernel 留到确有瓶颈后再学。

**验收：cache 数值测试通过；生成质量无明显异常；能展示长上下文下时间与显存变化。**

## P6：扩展为更完整的语言模型学习项目

在架构与计量工具稳定后逐项推进：

- [ ] BPE / SentencePiece：词表训练、特殊 token、编码解码和 checkpoint 一致性。
- [ ] 更大语料：清洗、去重、训练/验证隔离和数据版本记录。
- [ ] EOS、文档边界和 packing：明确是否允许跨文档注意力。
- [ ] SFT：对话模板、assistant-only loss mask，以及 padding mask。
- [ ] LoRA：可训练参数量、显存收益、合并权重前后的一致性。
- [ ] 有多卡需求时再做 DDP：数据划分、梯度同步、全局 batch 与恢复训练。
- [ ] MoE、偏好优化、蒸馏等作为选修，而非当前主线的前置条件。

不同 tokenizer 的 token-level loss / PPL 不能直接比较；换词表后需要定义共同的评估口径，例如相同文本上的 bits-per-byte，或统一下游任务指标。
SFT / LoRA 也不能代替基本的预训练质量验证；Tiny Shakespeare 适合学习机制，不足以据此判断聊天能力。

## 下一步建议提交顺序

1. **`feat: 增加最小实验记录与固定验证集`**：独立目录、配置 JSON、指标 CSV、训练与评估 RNG 隔离。
2. **`feat: 增加实验曲线绘图`**：读取 CSV，画 loss-token、loss-time、吞吐/显存对比。
3. **`perf: 增加可切换的 SDPA attention`**：先补一致性测试，再测性能。
4. **`feat: 保存完整训练状态并支持恢复`**：为更长的训练实验打基础。
5. 从 P3 中选择一项训练质量改动和一项效率改动，分别做独立实验。

不要提前把整个项目重构成通用训练框架。配置和接口随着真实实验需求增长即可。

## 每次实验记录模板

```text
实验名称 / 代码版本：
问题与假设：
对照配置 / 实验配置：
唯一主要变量：
数据版本 / tokenizer / seed：
参数量 / 有效 batch / 训练 token 预算：
硬件 / 软件版本 / dtype / 实际后端：
正确性测试：
质量结果：
吞吐 / 显存 / 延迟结果（含计时边界、重复次数与波动）：
结论：保留、回退，或需要更多证据？
局限和下一步：
```

最终目标是形成“提出问题 → 保证正确 → 测量对照 → 解释结果”的习惯，而不是完成一份技术名词清单。
