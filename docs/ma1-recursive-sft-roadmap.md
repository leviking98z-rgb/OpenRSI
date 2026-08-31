# MA1 SFT-Only 代际自举 Roadmap

> 状态：核心实验方案与 2026-08-29/30 执行结果（链路跑通，候选未晋升）
> 目标：在不改动 OpenMLE-Gym 和 OpenMLE-Evo harness 的前提下，验证一条低成本的
> `Evo experience -> distillation -> SFT -> model promotion` 权重更新链路，是否能提高整个系统的性能。

## 1. 边界与核心假设

本阶段固定以下两层：

- **OpenMLE-Gym 不变**：继续提供任务、沙箱和可执行评价。
- **OpenMLE-Evo 不变**：继续负责 Draft、Improve、Debug、程序执行和候选选择。

本阶段只改变 **OpenMLE-ERL 的模型更新方式**：

```text
原始方式：
Gym -> Evo rollout -> RL/GSPO -> MA1

当前原型：
Gym -> Evo search -> verified experience -> continued SFT
     -> candidate MA1 -> promotion -> next generation
```

第一版明确不做：

- 在线 RL 或 GSPO；
- harness、reward 或 evaluator 自修改；
- Crossover、多岛种群和长搜索树；
- 多个候选模型并行演化；
- 搜索期间在线同步权重。

要验证的首要假设是：

```text
在相同 Gym、Evo、任务、随机种子和执行预算下：

G1 + Evo@B > G0 + Evo@B
```

其中 `G0` 是当前 MA1，`G1` 是由 G0 的 Evo 经验继续 SFT 得到的候选模型，
`B` 是每个任务固定的程序执行预算。

## 2. 顶层闭环

```text
                    generation t

             current MA1 checkpoint Gt
                         |
                    freeze weights
                         |
                         v
        lightweight Evo search on Search-Train tasks
                         |
                         v
       executable programs + scores + feedback + lineage
                         |
                         v
             selection and SFT distillation
                         |
                         v
                offline continued SFT
                         |
                         v
                 candidate G(t+1)
                         |
                         v
       held-out promotion evaluation against Gt
                  /                   \
              accept                 reject
                |                       |
                v                       v
             G(t+1)                    Gt
```

这里有两种“晋升”：

1. **程序层晋升**：Evo 中更好的 child program 替换 parent program。
2. **模型层晋升**：由这些成功修改训练出的 candidate checkpoint，通过 held-out
   验证后替换 parent checkpoint。

第一阶段只跑一代：

```text
G0 -> search -> distill -> SFT -> G1
```

只有一代链路通过核心验证后，才进入：

```text
G1 -> fresh search tasks -> distill -> SFT -> G2
```

一代实验只能证明闭环有效；`G2 > G1 > G0` 才提供更强的递归自举证据。

## 3. 一代中的模块职责

### 3.1 Evo：冻结模型后的经验制造器

每代收集数据期间冻结 `Gt`。Evo 使用一个固定的小预算 profile，例如：

```text
2 x Draft
2 x Improve or Debug
```

具体预算可根据成本调整，但同一实验内必须固定。原型只保留：

```text
Draft -> Execute -> select best
                   -> Improve when valid
                   -> Debug when invalid
                   -> Execute
```

此阶段的重点不是重现完整长程搜索，而是用少量执行找到比直接生成更好的、可验证的监督信号。

### 3.2 复用现有 Evo-to-SFT 数据链路

OpenRSI 原本的 SFT 就包含 Evo 产生的轨迹数据，因此这里不需要重新发明一套
“Evo-to-SFT bridge”。发布代码已经提供：

```text
run_evolutionary_rollout.sh
  -> program_ep_0/<task>/stat.json 和逐 step artifacts
  -> select_evolutionary.py 选择有效轨迹 step
  -> messages 数据
  -> finalize_messages.py 去重和 token filter
  -> SLIME SFT
```

原始 SFT 数据由两部分组成：

- parallel rollout 产生的 full responses；
- evolutionary rollout 产生的 trajectory steps。

因此核心实验的默认做法是复用原有 operator-conditioned 样本语义，只把经验生成模型
从原始 teacher/rollout model 换成当前冻结的 `Gt`，并增加 generation、checkpoint、
task split 和 parent-child lineage 标记。

现有 evolutionary selector 会按 Draft、Improve、Crossover 及其 Debug descendants
恢复 segment，并使用执行分数、parent 比较、medal gate 和 causal-inheritance
annotation 选择训练 step。每个被选 step 原本的 system/user prompt 与 assistant
response 可直接构成 SFT 对。

发布代码中的 selector 主要输出 `selected_steps.jsonl` manifest。如果实际训练入口
尚不能直接读取该 manifest，只需补一个很薄的 materialization adapter，把已有逐 step
prompt/response artifacts 写成 `{id, messages}`。这是工程接线，不是新的算法模块。

第一版优先保留以下已有监督形式。

#### A. 成功局部修改

```text
input:
  task + parent program + execution feedback

target:
  better child program
```

仅保留：

- Improve：child 的可执行得分严格优于 parent；
- Debug：invalid parent 被修改成 valid child；
- evaluator 方向、分数和有效性都能被可靠解释的样本。

#### B. 最佳终点重新蒸馏为 Draft（可选增强）

```text
input:
  original task

target:
  best verified program found by the search
```

这可以把“依赖数步搜索才获得的最终程序”压缩为模型的一次 Draft 能力，但它不是打通
闭环所必需的。为减少首个核心实验的变量，先复用原始 trajectory-step 配方；待基线
闭环有效后，再把 endpoint-as-Draft 作为独立消融。

不要把 Improve/Debug 中依赖 parent 的原始 reasoning 直接配给 Draft prompt。不同
operator 的输入语义必须一致；必要时只使用最终代码，或重新构造与 Draft 输入一致的
响应。

### 3.3 Continued SFT：把搜索收益压入权重

第一版采用离线 continued SFT：

```text
anchor SFT replay + newly distilled Evo experience -> candidate G1
```

初始可使用以下数据混合范围：

```text
60-70%  原始 anchor/replay SFT
30-40%  新的 Evo 蒸馏数据
```

该比例是起点，不是结论。anchor replay 用于降低遗忘；新经验比例根据 held-out
operator eval 和 promotion eval 调整。

### 3.4 Promotion gate：决定是否替换父模型

candidate 不因训练完成而自动晋升。必须在从未参与搜索数据生成和 SFT 的
Promotion Set 上，与 parent 使用完全相同的推理和搜索预算进行比较。

最小晋升条件：

```text
1. fixed-budget Evo 主指标提高；
2. task-level wins > losses；
3. 程序有效率和 anchor 能力没有超过预设容差的退化。
```

若不满足，保留 parent，并把失败归因到数据、训练或 operator/search 匹配问题，
而不是继续无条件滚动到下一代。

## 4. 数据隔离

至少划分三部分：

```text
Search-Train
  运行 G0 + Evo，生成本代 SFT 数据

Promotion
  选择 checkpoint 和执行模型晋升，永不用于训练

Final Test
  方案冻结后只运行一次，用于最终结论
```

必须按完整 task 划分；如果多个任务共享同一 Kaggle competition、dataset 或高度
相似的数据源，还应按 competition/dataset family 成组隔离，而不是按数据行随机切分。

建议的核心实验规模：

```text
Search-Train:  30-50 tasks
Promotion:      8-10 tasks
Final Test:    15-20 tasks
```

在正式核心实验前可以使用更小的 smoke split 检查链路，但不能用 smoke 结果支持性能结论。

每一代还必须：

- 使用新的 Search-Train task shard；
- 为每个 task/model/seed 清空 Program DB 和搜索状态；
- 禁止把历史 best program、搜索树或 execution feedback 带进 held-out eval；
- 保存 task split、数据样本和 checkpoint 的可追踪 manifest。

## 5. 小核心 Eval

### 5.1 比较组

当前 critical path 只保留：

```text
A. G0
B. G1 = G0 + Evo-distilled SFT
```

它只回答核心工程问题：

```text
G1 > G0
  新增权重更新链路是否带来系统提升？
```

token-matched replay control 已从默认链路移除：每代不再为它生成数据、训练 checkpoint
或运行 eval。它属于正结果出现后的归因消融，不参与当前 promotion；在 G1 尚未超过
G0 时运行该分支不会改变晋升结论，只会增加一次 SFT 和一次完整 eval。

### 5.2 Eval 1：Direct / first-candidate eval

在 held-out tasks 上比较相同采样配置下的第一个候选：

```text
G0 direct@1
G1 direct@1
```

记录：

- valid program rate；
- normalized task score；
- task-level win/tie/loss。

该测试回答：搜索经验是否已经被压缩成模型自身的一次生成能力。

### 5.3 Eval 2：固定状态的 operator eval

从 held-out tasks 固定一批：

```text
task + parent program + execution feedback
```

把完全相同的状态分别交给各模型生成一次 child，记录：

- Improve child 严格优于 parent 的比例；
- Debug 从 invalid 变成 valid 的比例；
- child valid rate；
- child 相对 parent 的 normalized score delta。

这是一个低成本的机制验证：它直接检查 MA1 在 Evo 内负责的局部修改能力是否提高。

### 5.4 Eval 3：端到端 fixed-budget Evo eval

把模型重新接回同一个 Evo harness：

```text
G0 + Evo@B
G1 + Evo@B
```

所有组固定：

- task 集合；
- Evo 代码和配置；
- prompt/template；
- sampling 参数；
- seed 集合；
- 每个任务的程序执行预算 `B`；
- evaluator、sandbox 和超时策略；
- 初始为空的 Program DB。

主指标：

```text
best normalized score @ fixed execution budget B
```

稳健配套指标：

- per-task paired score delta；
- task-level win/tie/loss；
- median normalized score；
- valid program rate；
- best-so-far score at each execution；
- search-curve AUC，即同等预算内的整体搜索效率。

不同 Kaggle 任务的原始 metric 尺度和方向可能不同，不能直接平均 raw score。优先使用
Gym 的统一 normalized score；若某类任务不能可靠归一化，则以 task-level
win/tie/loss 和任务内 paired delta 为主。

### 5.5 最小标准 Eval 入口

`OpenMLE-ERL/SFT/scripts/generational/standard_eval.py` 把上述核心检查收敛为
两个子命令。任务始终按冻结 manifest 的顺序取前 `num_tasks` 个，因此小规模
smoke 是正式 Eval 的固定子集。

```bash
# 固定上下文的 Debug / Improve operator gate
python scripts/generational/standard_eval.py operator \
  --cases operator-cases.jsonl \
  --parent g0-operator-results.jsonl \
  --candidate candidate-operator-results.jsonl \
  --num-tasks 24 \
  --operators debug improve \
  --output-dir outputs/operator

# 复用已经导出的 fixed-budget Evo execution records
python scripts/generational/standard_eval.py e2e \
  --parent g0-evo.jsonl \
  --candidate candidate-evo.jsonl \
  --task-manifest promotion.jsonl \
  --num-tasks 24 \
  --budget 16 \
  --output-dir outputs/e2e
```

两种模式都写出统一的 `standard_eval.json`。Operator gate 要求 Debug 和
Improve 各自成功数不低于 G0，且总成功数严格高于 G0；E2E gate 要求
`best@budget` 平均值提高、逐题 wins 多于 losses，且 valid rate 不低于 G0。

### 5.6 随机性与统计报告

推荐每个 held-out task 使用 3 个配对 seeds；成本极紧时，优先保留更多 task，而不是
在极少 task 上堆很多 seeds。

最终至少报告：

```text
mean and median paired delta
win / tie / loss
95% paired bootstrap confidence interval
score-vs-execution curve
```

tie 的容差应在看结果前，根据 evaluator 精度预先确定。

### 5.6 核心成功判据

**系统性能证据：**

```text
G1 + Evo@B > G0 + Evo@B
```

**权重吸收证据：**

```text
G1 direct@1 > G0 direct@1
and/or
G1 has higher Improve/Debug success than G0
```

不同结果应作不同解释：

| 观察 | 解释 |
| --- | --- |
| direct@1 和 Evo@B 都提高 | 搜索经验进入权重，并转化为系统收益 |
| direct@1 提高，Evo@B 不提高 | 模型变强，但当前搜索策略没有利用该提升 |
| direct@1 不变，operator 与 Evo@B 提高 | 主要增强了 Improve/Debug，而非初始 Draft |
| operator 提高，Evo@B 不提高 | parent selection、预算分配或 operator mix 可能不匹配 |
| 所有指标不提高 | 优先检查蒸馏格式、数据质量和训练设置 |

## 6. 优化路线

优化顺序遵循“先证明正确，再提高效率”，避免同时改变多个变量。

### Phase O1：提高单位执行的数据质量

- 只保留可执行验证且方向明确的 improvement；
- 设置最小 score delta，过滤 evaluator noise；
- 每个 task 限制相似样本数量，避免少数任务主导训练；
- 对代码和 message 做 exact/near-duplicate 去重；
- 分开统计 Draft、Improve、Debug 的样本数、成功率和 token 成本；
- 保留 parent、child、feedback、score 和 lineage，确保每条样本可审计。

### Phase O2：提高 SFT 的学习效率

- 先使用单一 candidate 和离线批量 SFT；
- 每代只训练一个 candidate，不运行 matched-compute control 分支；
- 使用 anchor replay 控制遗忘；
- 根据 held-out operator 指标调整 Draft/Improve/Debug 的采样权重；
- 通过 Promotion Set 选择 checkpoint，不在 Final Test 上挑 checkpoint；
- 若 full-parameter SFT 成本仍过高，再单独比较参数高效微调，不能在主实验中混淆变量。

### Phase O3：提高搜索效率

只有核心链路有效后再尝试：

- 在固定最大预算下优先扩展 best parent；
- 根据 valid/invalid 状态选择 Improve 或 Debug；
- 删除持续无收益的分支；
- 以“合格训练样本数 / program execution”和“最终分数 / execution”作为效率指标；
- 当前第一批固定为 D16，不在同一批任务上继续 D32/D64；
- Crossover 保持论文式配置：第 2 generation 后、存在两个合格 parent 时可用。

优化后的版本仍需与原始 fixed-budget profile 做隔离对照，防止把搜索预算变化误认为
模型权重提升。

## 7. 分阶段交付

### Phase 0：冻结实验协议

交付：

- task-family level 的 Train/Promotion/Test manifests；
- 固定 Evo profile、sampling config、seed 和 execution budget；
- G0 baseline 的 direct、operator 和 Evo@B 结果；
- evaluator 重复性与失败处理规则。

退出条件：同一模型重复运行的波动足够小，可以识别预期提升。

### Phase 1：复用并验证现有 Evo-to-SFT 数据链路

交付：

- 验证 `Evo artifacts -> selected steps -> messages -> final SFT data` 的现有路径；
- 复用 validity、parent-improvement、medal、dedup 和 token-length filters；
- 为每条训练样本补充 generation、parent checkpoint、task split 和 lineage；
- 若公开入口停在 `selected_steps.jsonl`，补充最薄的 messages materialization adapter；
- 数据统计和少量人工抽检报告。

退出条件：随机抽样的训练对在 prompt 语义、代码、反馈和分数方向上均一致。

### Phase 2：训练一代 candidate

交付：

- G1 checkpoint；
- 完整训练配置、数据版本和 parent checkpoint 记录。

退出条件：训练稳定完成，且基础 anchor eval 无明显灾难性退化。

### Phase 3：运行小核心 Eval

按顺序运行：

```text
direct@1
-> fixed-state operator eval
-> fixed-budget Evo@B on Promotion
-> promotion decision
-> one-shot Final Test
```

退出条件：G1 在 Final Test 的 Evo@B 主指标优于 G0。

### Phase 4：消融与效率优化

在核心链路通过后，再比较：

```text
endpoint-only
transition-only
endpoint + transition
with / without anchor replay
different strict-improvement thresholds
```

目标是找到最少搜索执行、最少训练 token 下仍能稳定产生增益的数据配方。

### Phase 5：第二代递归验证

使用全新的 Search-Train shard：

```text
G1 -> Evo experience -> SFT -> G2
```

除了检查 `G2 > G1`，还要比较 G0 和 G1 的经验生产效率：

- fixed budget 下找到的 best program；
- valid program rate；
- successful Improve/Debug rate；
- 每个 execution 产生的合格 SFT 样本数；
- 达到同一质量阈值所需的执行次数。

更强的递归信号不是单纯“又训练了一轮”，而是：

```text
G1 比 G0 更擅长制造高质量经验，
并且这些经验能够训练出 G2 > G1。
```

## 8. 实验记录与可复现性

每一代保存一个 lineage manifest，至少包含：

```text
parent checkpoint
task split and task-family hash
Evo code/config hash
sampling parameters and seeds
raw artifact locations
distillation/filter version
selected sample IDs
SFT config and data mixture
candidate checkpoint
promotion and final-test results
```

因此系统中的因果变量始终清晰：

```text
Gym: fixed
Evo harness: fixed
ERL update path: changed
MA1 checkpoint: G0 -> G1
```

这个 roadmap 的第一里程碑不是宣称完整 RSI，而是以最小成本证明：

> Evo 产生的执行验证经验，可以通过 SFT 被吸收到 MA1 权重中，并在未见任务、
> 相同搜索预算下提高整个系统的性能。

## 9. 2026-08-29/30 核心实验结果

本节记录一次完整的原型执行。它验证了权重更新链路的工程可行性，但没有观察到
候选模型的系统性能提升。所有正负结果均按预先固定的 strict gate 保留；没有通过
放宽样本筛选、缩减 Evo 预算或在 Final Test 上重新挑 checkpoint 制造正向结果。

该已完成实验曾运行 token-matched SFT control。2026-08-30 的后续工程决策已将
control 从默认 critical path 删除；下文仍保留其历史数字以保证审计完整性，但未来
generation 不再自动生成、训练或评测该分支。

### 9.1 固定协议

本轮保持以下边界：

- Gym、Evo harness、sandbox 和 scorer 不改；
- 不新增 RL/GSPO，只进行 continued SFT；
- Promotion 和 Final Test 各 8 个完整任务；
- 每题 Evo4，即 4 次程序生成与执行；
- 最大生成长度 12,288 tokens；
- task、LLM 和 sandbox 并发均为 8；
- 使用一个预先固定的 seed `20260829`，把有限预算优先用于更多任务；
- Final Test 在候选 checkpoint 固定后才打开，且不参与模型选择。

正确的 G0 parent 是 G1 实际继承的 RL 后 checkpoint：

```text
checkpoints/rl-frontis-qwen36-public-full-v033-g0-hf
```

一次较早的 Final Test 误把原始 Qwen3.6 checkpoint 当作 G0。该比较及其结论已
标记为 invalid 并从正式结果中排除。

### 9.2 G1：八条严格 transition 的一代 SFT

G0 在 16 个 Search-Train 任务上运行搜索后，蒸馏得到：

```text
8 strict transitions
  6 Debug: invalid parent -> valid child
  2 Improve: child score strictly better than parent
7 contributing tasks
```

训练数据与 matched-compute control 均为 128 行：

```text
G1 candidate: 8 Evo transitions + 120 shared anchors
G1 control:   8 token-matched replay rows + 120 shared anchors
candidate tokens: 1,422,616
control tokens:   1,422,617
SFT updates: 3
```

使用提交 `20f0eb0` 的 pooled raw-score normalization 重新计算 Promotion 后，
G1 相对 matched G0 parent 的结果为：

| 指标 | G1 - G0 | Win/Tie/Loss | 95% paired bootstrap CI |
| --- | ---: | ---: | ---: |
| direct | -0.108879 | 1/5/2 | [-0.422516, 0.143396] |
| best@4 | -0.204352 | 3/1/4 | [-0.685441, 0.275819] |
| search AUC | -0.147507 | 3/1/4 | [-0.517549, 0.206490] |
| valid rate | -0.125000 | 3/1/4 | [-0.437500, 0.187500] |

相对 token-matched SFT control，G1 的 best@4 平均差为 `+0.092921`
（2/5/1），但置信区间跨零，且它相对真正 parent 的 best 和 valid rate 均退化。
Promotion gate 因此为：

```text
G1 decision = reject
formal champion remains G0
```

### 9.3 G2：一条严格 transition 的实验分支

为观察递进行为，实验性地从被拒绝的 G1 candidate 继续搜索。新的 8 题
Search-Train shard 只产生：

```text
1 strict Improve transition
```

G2 使用 `1 Evo + 127 anchors`、共 128 行和 3 次 SFT update。它不被视为正式
champion 的后代，只是诊断性的递进分支。新版 Promotion 结果为：

| 对照 | best@4 平均差 | Win/Tie/Loss | valid-rate 差 |
| --- | ---: | ---: | ---: |
| G1 experimental parent | -0.093165 | 1/4/3 | -0.093750 |
| G1 token-matched control | -0.000243 | 1/5/2 | -0.093750 |

因此：

```text
G2 decision = reject
```

### 9.4 G3：strict gate 触发停止

G2 在三个独立的 8 题 Evo4 shard 上继续采集：

```text
24 tasks
96 executions
0 strict Improve/Debug transitions
```

三批耗时分别为 405、367 和 647 秒。由于没有合格监督信号，系统执行：

```text
stop_without_weight_update
G3 SFT not run
G3 Promotion not run
```

strict gate 没有被放宽，也没有把失败 endpoint 当作训练数据。

### 9.5 正确 G0 对 G1 的 Final Test

正确 G0 和已固定的 G1 checkpoint 在同一个 8 题 Final Test 上分别运行
32 次程序执行：

```text
G0 valid executions: 7/32 = 21.875%
G1 valid executions: 2/32 =  6.250%
```

基于 parent、candidate 合并 raw scores 的逐任务归一化结果：

| 指标 | G0 | G1 | G1 - G0 | Win/Tie/Loss | 95% paired bootstrap CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct | 0.000000 | 0.000000 | 0.000000 | 0/8/0 | [0.000000, 0.000000] |
| best@4 | 0.327075 | 0.125000 | -0.202075 | 1/4/3 | [-0.625000, 0.250000] |
| search AUC | 0.176382 | 0.031250 | -0.145132 | 1/4/3 | [-0.344933, 0.031250] |
| valid rate | 0.218750 | 0.062500 | -0.156250 | 1/4/3 | [-0.406250, 0.093750] |

Final Test 不参与 checkpoint 选择，也不能推翻此前的 Promotion 决策；它作为独立
验证同样未显示 G1 提升。

### 9.6 效率 smoke

在不减少任务数、Evo 深度或 token 上限的前提下，采用了以下优化：

| 环节 | 原设置 | 采用设置 | 观察 |
| --- | --- | --- | --- |
| Eval | TP8，并发 4 | TP8，并发 8 | 307s -> 234s，wall time -23.8% |
| Eval topology | DP2 x TP4 | 单 TP8 replica | DP2 x TP4 明显更慢，弃用 |
| SFT optimizer | CPU offload | GPU optimizer | first/warm-cache step 约 722s -> 284-297s |
| SFT recompute | 尝试关闭 | full uniform recompute | 关闭后更慢且显存更高 |
| SFT packing | 较小 packing | 16K tokens/GPU | steady step 约 102-126s |

正式 G1 candidate 的三步耗时为 289.1、104.9 和 118.0 秒。采用共享编译缓存，
并且不保存本原型不需要的 optimizer/RNG checkpoint payload。

### 9.7 解释与下一步

本轮能支持的结论是：

1. `Evo -> strict transition distillation -> continued SFT -> eval/promotion`
   工程闭环已经端到端跑通；
2. 在本轮小样本配方下，该链路没有提高系统性能，formal champion 仍是 G0；
3. 合格 transition 从 G1 的 8 条下降到 G2 的 1 条，再到 G3 的 0 条，说明经验
   生产质量/密度是当前首要瓶颈，而不是继续无条件增加 SFT 代数；
4. 搜索 feedback 清洗曾遗漏 sandbox stderr，使部分 Debug 样本只能看到
   `code_execution_error`，看不到真实 traceback。修复会保留 stderr，但它只用于
   后续实验，不能回写或美化本轮历史结果。

由于 Promotion 和 Final Test 都只有 8 题、单 seed，且 best@4 的 bootstrap
置信区间跨零，本轮不支持对更大任务分布作强统计结论；这里的 `reject` 是按预先
固定 gate 作出的工程决策。负向的 valid-rate 变化和三代 transition 密度下降仍足以
说明当前配方不应继续自动递进。

下一轮若继续，优先验证 stderr feedback 修复是否提高每次 execution 产生的 strict
Debug/Improve 样本数；只有经验密度恢复后才值得再次执行权重更新。

节点上的完整实验审计目录为：

```text
/data2/openrsi/experiments/ma1_recursive_sft_20260829
```

其中保留原始 rollout、逐次评分、导出 JSONL、Promotion 报告、checkpoint、效率
smoke 和错误 baseline 的 invalid 标记。

## 10. 下一轮：8 节点自生成数据计划

### 10.1 为什么不再使用 Evo4 作为正式采集深度

Evo4 已经完成它的职责：验证

```text
MA1 -> Evo -> sandbox/scorer -> lineage -> messages -> SFT artifact
```

链路可以正常工作。但它只够覆盖 Draft 和最早的一次 Debug，难以稳定产生 Improve，
更难积累两个有效 parent 触发 Crossover，因此不能作为正式 Trace Bank 的默认深度。

论文公开配置对 evolutionary collection 使用每题最多 64 次 operator execution。
对公开的 9,014 条 evolutionary SFT rows 进行统计，所选 operator 在搜索树中的
全局执行位置为：

| 统计量 | 1-based operator position |
| --- | ---: |
| mean | 17.3 |
| median | 12 |
| 75th percentile | 25 |
| 90th percentile | 44 |

累计覆盖率为：

| 搜索预算 | 已覆盖的公开 selected-row 位置 |
| --- | ---: |
| Evo4 | 22.7% |
| Evo8 | 40.2% |
| Evo16 | 61.8% |
| Evo32 | 82.5% |
| Evo48 | 92.7% |
| Evo64 | 99.7% |

这里的 position 是任务内的全局 operator execution 序号，并不等于某个 child 的
祖先链长度。下一轮将统一使用“execution budget”描述深度，避免把二者混淆。

### 10.2 第一批 Trace Bank 的规模

冻结公开 `FrontisAI/Frontis-MA1-35B` 作为 G0，选择 64 个全新的 Search-Train
任务。任务与固定 Promotion、Final Test 以及本轮 pilot 按
competition/dataset family 隔离。

8 个节点各运行一个 TP8 SGLang replica，并按任务分片：

```text
D16: 64 tasks，每个任务最多 16 executions
```

最大执行量：

```text
64 x 16 = 1,024 operator executions
```

8 个节点分别承担 8 个 D16 任务。只使用一个固定采样 seed，把计算优先用于任务
覆盖和有效谱系数量，而不是 seed 重复。

如果第一批没有达到训练数据 yield gate，再使用第二批 64 个完全不重合的
Search-Train 任务重复该协议；不在当前任务上继续加深。

### 10.3 搜索配置

Evo4 smoke 使用的 `crossover_prob=0`、单 candidate、单层 Debug 配置不再用于正式
采集。正式 profile 以论文公开 evolutionary profile 为基础：

```text
operators: Draft / Improve / Debug / Crossover
individuals_per_generation: 5
crossover_prob: 0.5
num_generations_till_crossover: 2
max_debug_depth: 10
execution budget: 16
```

模型始终冻结。Evo 只负责选择 operator 和 parent、维护搜索树并调度执行；每次
operator 的输出仍由同一个 G0 生成。

### 10.4 固定 D16 边界

本轮不做任务级深度晋级，也不根据中间结果把部分任务加深到 D32/D64。这样 64 个
任务使用同一个最大 execution budget，避免把自适应预算策略引入首个数据配方实验。
基础设施错误允许重试并明确标记；模型自身的错误、stderr、timeout 和 scoring
failure 必须原样保留。

### 10.5 Raw Trace Bank 与 SFT 视图

Raw Trace Bank 无条件保存四类 operator 的成功与失败：

```text
Draft
Debug
Improve
Crossover
```

每条记录保留 task/shard、G0 revision、operator、parent/child、完整 messages、
代码、execution feedback、score/方向、validity、token、runtime 和 hash。Raw bank
验收后冻结；不同训练配方只做离线派生，不重新调用模型。

从同一 Raw bank 生成四个可审计视图：

| View | 规则 |
| --- | --- |
| `strict_only` | invalid-to-valid Debug；严格优于 parent 的 Improve；严格优于更强 parent 的 Crossover |
| `verified_endpoint` | 每题最佳的已复验 valid endpoint，改写为 Draft |
| `causal_segment` | 达到 endpoint gate 的局部 segment 中，对最终有效方案有继承贡献的步骤 |
| `main_g1` | 当前 Trace Bank 的 Draft endpoint 与高质量 non-Draft 的去重、限额并集 |

每条 selected row 显式标记 `strict`、`causal` 或 `endpoint`。非 strict 样本可以进入
候选池，但绝不标成 strict。第一版不恢复旧的 historical anchor/replay；当前
Trace Bank 内的高质量 Draft 自身承担基础能力样本的作用。

### 10.6 数据量与质量门槛

第一批最多 1,024 次 raw executions 不预设必须达到的 SFT 行数。完成后首先报告：

```text
实际 execution 数及有效/失败数
Draft / Debug / Improve / Crossover 数量与比例
strict transition 和 verified endpoint 数
contributing task 数
full-message / assistant token 长度分布
```

这些统计决定后续 recipe 和是否需要第二批不重叠任务；不把 non-strict 记录改写成
strict，也不为了凑数纳入不可执行或 parent 不一致的样本。

最终数据还必须通过：

- selected endpoint 重新执行与重新评分；
- parent/child lineage 一致性；
- private-label、外网与主机路径泄漏检查；
- 完整 messages exact dedup 和代码去重；
- near-duplicate 报告与 per-task cap；
- G0 chat template 精确 tokenization；
- 32,768-token 上限；
- JSONL/Parquet 内容一致及文件 hash 固化。

### 10.7 与论文规模的关系

这仍然是 core experiment，而不是论文复现。论文最终使用 26,259 条 SFT 样本，
覆盖 4,891 个任务，其中 9,014 条来自最多 64 executions 的 evolutionary path。
本计划只采集第一批最多 1,024 次 execution，用来得到本模型自生成且可验证的
operator 数据。

因此本阶段只回答：

> 增加真实搜索深度后，MA1 能否制造足够多、覆盖四类 operator 的健康 SFT 数据，
> 并使一次 G1 continued SFT 在独立 eval 上超过 G0？

它不声称复制论文的数据规模，也不以 Trace Bank 内部的训练分数作为性能提升证据。

### 10.8 持续唯一任务采集实现

固定 64-task D16 批次之后，采集器切换为 slot 级动态分发，而不是提前给每个节点
发完一个静态 task 列表：

```text
slot 完成当前 task
-> 写入 terminal manifest 和不可变 archive
-> 原子领取一个从未运行过的新 task
-> 仍使用 sample_index=0
```

冻结后的可用 Search-Train inventory 共 928 个唯一 task，按运行环境拆成互不重叠的
825-task H20 pool 和 103-task L20 pool。固定 64 题、Promotion、Final Test 及同源
family 均已排除。共享 `mkdir` claim 是唯一任务所有权边界，因此节点速度不同时也不会
重复运行同一 task。

实现与审计材料位于：

```text
exp/g0_continuous_tracebank_20260831/
```

其中 validator 要求每个成功 trace 具备 16 个连续 execution、完整 parent lineage、
逐步 score/status/token/runtime、搜索事件与状态文件，以及 archive SHA-256 一致。
2026 年 8 月 31 日的首个 25-task L20 检查点已全部通过，共 400 次 execution；
operator 分布为 Draft 111、Debug 220、Improve 63、Crossover 6。完成 task 的中位
端到端耗时为 13.6 分钟，且 8 个 slot 均已自动补入新 task。
