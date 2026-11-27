# Adaptive-RAG 实验报告 · 2026-11-26

**在线补测更新。** DeepSeek 已返回有效 completion 与 token 用量。下文“没有在线成绩”等表述描述首次离线实验的执行条件。其后已补测 28 道意图题及一套独立虚构政策问答与裁判题。[DeepSeek 在线实验及原始回执](23-deepseek-live.md)

本轮最明确的工程收益来自订单状态更新。四个独立进程同时给一笔订单申请全额退款，原实现写出四条申请，修复后在 PostgreSQL 和 SQLite 各写一条。检索方面，真实中文 BGE 已跑通。固定候选数后，重复三轮检索在离线生成环境没有增加证据覆盖。完整 Cross-Encoder 与在线生成效果仍待复现。

## 一、65 题检索对照

39 篇本地政策，同一套 65 个问题。P@1 按首篇相关文档计，R@3 按问题统计前三篇去重文档中相关文档占金标文档的比例，再取均值。MRR 取首个相关文档排名的倒数。每题的文档排序保存在原始 JSON。

| 系统 | 首篇命中 | P@1 | R@3 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 52/65 | 80.00% | 90.00% | 0.8556 |
| 真实 BGE-small 中文向量 | 53/65 | 81.54% | 94.62% | 0.8821 |
| BM25 + BGE + RRF，各取 5 个候选 | 54/65 | 83.08% | 94.62% | 0.8923 |
| 同一融合，各取 8 个候选，保留前 5 | 54/65 | 83.08% | 93.08% | 0.8890 |
| 相同的 8 个融合候选，词面重排取前 5 | 56/65 | 86.15% | 93.08% | 0.9044 |
| 哈希向量 | 2/65 | 3.08% | 6.92% | 0.0518 |
| BM25 + 哈希向量 + RRF，各取 5 个候选 | 46/65 | 70.77% | 82.31% | 0.7872 |

同候选对照中，词面重排修对 3 题、排错 1 题，净增 2 题，双侧精确配对检验 p=0.625。混合召回相对 BM25 修对 4 题、排错 2 题，p=0.688。可以陈述观测值，样本尚不足以声称稳定优势。哈希向量在这批中文题上反而拖累 BM25，后续选择降级策略时应单独评估词面召回。

本次使用缓存的 `BAAI/bge-small-zh-v1.5`，版本 `7999e1d3359715c523056ef9478215996d62a620`。本机项目基础 Python 环境的 sentence-transformers 5.7.0 超出仓库 rerank 可选依赖声明的 3.x 范围，使用这一环境的真实权重完成推理。指定精排模型 `BAAI/bge-reranker-base` 的权重不在本轮可用缓存中。8 月的 89.2% 与 97.7% 保留为历史口径。

[真实 BGE 逐题排名](../eval/results/nov26/retrieval-local.json) · [哈希对照](../eval/results/nov26/retrieval-hash.json)

## 二、自适应路由、检索轮数与回答覆盖

编写 24 道开发题和 12 道留出题，每组各半是简单题和复合题。留出题更换政策主题和连接方式，例如用两个独立问句同时询问包装费与发票。系统看不到金标。三种策略用同一 LangGraph、政策库、BGE、词面重排与离线摘录生成。固定策略仅替换路由决策，固定三轮另外关闭提前停止。关闭跨请求检索缓存，按固定随机种子轮换策略顺序，每题每种策略运行一次。

**证据覆盖**要求列出的参考政策前六个检索片段中出现全部金标短语。**回答覆盖**要求用户最终看到的文本包含全部短语，忽略兼容字符和空白差异。该指标会漏掉同义改写，也无法判断否定、条件和错误附加说明。下表按题计数，延迟是本地工作流中位数，排除建索引及数据库会话初始化。独立人工正确率尚待双人盲评。

| 开发集 24 题 | 证据覆盖 | 回答覆盖 | 总检索轮数 | 中位耗时 |
|---|---:|---:|---:|---:|
| 改动前固定单轮，默认取 3 个候选 | 17 | 10 | 24 | 20.13 ms |
| 改动前固定三轮，每轮取 8 个候选 | 24 | 10 | 72 | 20.01 ms |
| 改动前自适应，按复杂度取 3 或 8 个候选 | 23 | 10 | 24 | 19.67 ms |
| 控制候选数均为 8，改动前固定单轮 / 固定三轮 / 自适应 | 24 / 24 / 24 | 10 / 10 / 10 | 24 / 72 / 24 | 20.79 / 19.76 / 21.55 ms |
| 改动后自适应 | 23 | 16 | 24 | 22.34 ms |

离线改写器原样返回查询，充分性判断在离线模式下对非空结果提前停止。强制重复相同问题三次没有增加新信息。默认候选数不同造成了表面上的“多轮收益”。毫秒级单次测时波动较大，无法据此推断在线 LLM 的响应速度、Token 或成本。

**改动与代价。** 原离线回答只摘录首个片段。复合路由现在可选取最多三个覆盖额外问题词的片段，逐条显示出处。开发集简单题维持 9/12，复合题从 1/12 升到 7/12，共修对 6 题、回退 0 题。平均回答从 90.7 字变为 128.2 字。开发集精确配对 p=0.03125，此集参与过诊断，检验仅作探索性描述。

**留出结果。** 自适应回答仍为 5/12，逐题零变化。6 道复合题都被分到单轮，摘录改动没有触发。相同留出集强制三轮时，改动前后从 5/12 到 8/12。下一轮应在新收集的自然问法上分别评估复合问题识别、证据齐全度、回答完整度和额外无关条款，继续保留这 12 题作为回归集。不要把开发集增幅写成用户回答准确率。

[开发题](../eval/dataset/strategy_policy.jsonl) · [留出题](../eval/dataset/strategy_holdout.jsonl) · [分组统计与配对检验](../eval/results/nov26/summary.json) · [留出题逐题答案](../eval/results/nov26/holdout-after/strategy.json)

## 三、意图分类与忠实度

| 意图方法 | 原有 100 题 | 新编 28 个问法 |
|---|---:|---:|
| 基础关键词规则 | 50 | 5 |
| 仅纠偏词表，未覆盖计错 | 100 | 2 |
| 规则 + 纠偏，当前离线路径 | 100 | 6 |
| BGE 最近邻分类 | 42 | 21 |
| BGE 最近邻 + 纠偏 | 100 | 22 |

新题覆盖 14 类、每类两题。最近邻实验只索引原 100 道有标签示例，使用归一化余弦最近邻，不调阈值。原集回测排除相同问题。新题组合方案相对规则方案 16 胜、0 负、12 平。仍有 6 道错题，例如把“包裹揽收以后一直没动”判成投诉，把古诗请求判成推荐。BGE 最近邻目前是实验分支，未接管应用的意图决策。真实 DeepSeek v3 单独分类与加纠偏后的对照入口已编写，两种分数复用同一份模型原始分类；本轮没有在线成绩。

忠实度反例包含 10 对回答。每对共享关键词，但调整适用品类、否定、时间起算点或收费承担方。10 条正确、10 条错误回答均通过原关键词指标，在这个专门构造的挑战集上误放 10/10。这个数字衡量关键词指标的盲点。真实模型裁判与人评的误放率都仍未知。通用退货政策本身也有“15 日”条款，判断一条答复能否成立，需要核对适用条款和举证要求，仅凭出现“15 日”无法判错。

已导出随机顺序的匿名回答与参考政策。两名评审应独立填写品类适用、必要条件、额外错误三栏；完成后按编号对齐，报告各项通过率、分歧数与一致性，再讨论分歧。当前表中评分栏为空。

[意图逐题结果](../eval/results/nov26/intent.json) · [最近邻及错题](../eval/results/nov26/intent-bge.json) · [忠实度成对测例](../eval/dataset/faithfulness_pairs.jsonl) · [盲评表](../eval/results/nov26/holdout-after/blind-review.csv)

## 四、订单并发、确认重试与会话

每个数据库实验使用全新的合成订单。四个独立 Python 进程各持一个数据库会话，通过屏障同时开始执行仓储调用；核对提交结果、退款表和取消审计表。PostgreSQL 使用本地 18.0 实例中的独立临时数据库，结束后删除该实验库。

| 四个并发请求针对同一订单 | SQLite 改前 → 改后 | PostgreSQL 改前 → 改后 |
|---|---:|---:|
| 实际写入的全额退款申请 | 4 → 1 | 4 → 1 |
| 实际写入的取消审计记录 | 4 → 1 | 1 → 1 |

原先先读状态、再更新，允许多个请求同时通过检查。现在在数据库的一条条件更新中检查允许的状态并转入退款中或已取消，再在同一事务写业务记录。修复后两种库均只允许一个请求提交。PostgreSQL 取消实验在这一次基线中本来就只写了一条，不能声称该项也有四倍改进。

应用层另测了从用户消息生成退款确认卡、同时确认、查询落库结果。原退货技能目录缺少退款工具，新增退款申请及进度查询流程。相同确认 ID 的并发重试现在可在状态冲突后核对幂等键和参数指纹，复用已完成结果。SQLite 单实例修复前对四次同时确认只有一条回复报受理；修复后连续三轮各 4/4 报受理，各库只有一条退款申请。不同服务实例接收另一实例签发的确认仍返回无效，待确认记录当前保存在进程内。实验没有接触支付渠道。

会话另测了同一会话“那物流呢？”能接回 ORD10001 并回答运输中。新会话未继承该订单，缺单号的写请求未生成确认卡。服务端恢复同属主会话的摘要，向生成节点传入有界历史，压缩时把将移出窗口的消息交给摘要器。超长自然对话的摘要正确率与全请求 Token 上限仍未取得实测。

[PostgreSQL 改前](../eval/results/nov26/pg-before.json) · [改后](../eval/results/nov26/pg-after.json) · [并发重试改前](../eval/results/nov26/runtime-before-replay.json) · [应用层实测](../eval/results/nov26/runtime.json)

## 五、工程验证与阻塞项

| 项目 | 本轮结果或具体阻塞 |
|---|---|
| 限流与故障处理 | 单用户 32 个并发请求、内存限额 8，放行 8、拒绝 24。回归测试覆盖暂时性故障后重试成功、鉴权失败不重试。没有持续压力下的容量数据。 |
| 端到端客服 | 16 个真实工作流请求，使用 BGE 和离线生成。关键词全命中 10/11，预期拒答 3/3，工具路径启发式检查 5/5。P50 为 49.2 ms，P95 为 59.1 ms。数据库副作用另见订单实验。 |
| UI 和 SSE | 前端构建成功，3 个 Vitest 用例通过，后端 HTTP/SSE 样例通过。Chrome 出现 crashpad 命名管道错误，Edge 的页面调试连接被重置。本轮未取得从输入框到刷新后显示的完整浏览器通过结果。 |
| 数据库和向量服务 | 真实 PostgreSQL 仓储并发完成。Qdrant 嵌入式客户端重复 upsert 后仍为 2 个点，查询首位正确。独立 Qdrant HTTP 和 Redis 均无监听。Docker Desktop 引擎启动超时，三份 Compose 配置静态校验通过。 |
| 文档入库 | 使用真实 Unstructured 解析合成 DOCX 与英文 PNG，各得到 1 个片段，两者均含预期“15 days”。仅为两例冒烟测试。 |
| Skill / MCP | 本地工具返回营业时间。通过回环 HTTP 服务验证工具参数传递、成功响应和上游 503 错误传递。当前接口使用项目自定义调用格式，通用 MCP 协商与动态发现没有实现。 |
| 真实生成与模型重排 | 当前可用网络不包含 DeepSeek 或模型仓库，缺指定 Cross-Encoder 权重。在线查询改写、生成、裁判、v3 意图消融以及真实模型 Token/延迟均没有本轮成绩。 |
| 构建与回归 | 后端 171 项测试通过，1 条 LangGraph 弃用提示；Ruff、mypy 通过。 |

[端到端逐题数据](../eval/results/nov26/e2e-offline.json) · [HTTP 工具与 SSE 事件实录](../eval/results/nov26/transports.json) · [解析结果](../eval/results/nov26/ingestion.json) · [后端测试记录](../eval/results/nov26/pytest.xml)

## 六、复核与复跑

所有测例均为合成数据。原始版本起点是 `2a12a62` 加当时工作区已有改动。实验结果存放在 `eval/results/nov26/`，其中 `policy`、`holdout-before` 保存修复前的逐题回答，`policy-after`、`holdout-after` 保存修复后回答。完整配对统计从已保存的原文重新计分，无需模型。复跑时使用新输出目录可保留原始记录。

```powershell
python -m eval.retrieval_study --out scratch/recheck/retrieval.json
python -m eval.strategy_study --out scratch/recheck/policy
python -m eval.strategy_study --equal-k --out scratch/recheck/equal-k
python -m eval.strategy_study --cases eval/dataset/strategy_holdout.jsonl --out scratch/recheck/holdout
python -m eval.intent_study --out scratch/recheck/intent.json
python -m eval.intent_embedding_study scratch/recheck/intent-bge.json
python -m eval.faithfulness_study --out scratch/recheck/faithfulness.json
python -m eval.write_study --backend sqlite --out scratch/recheck/sqlite.json
python -m eval.write_study --backend postgres --out scratch/recheck/postgres.json
python -m eval.runtime_study --out scratch/recheck/runtime.json
python -m eval.transport_study scratch/recheck/transports.json
python -m eval.ingestion_study scratch/recheck/ingestion.json
python -m eval.browser_study scratch/recheck/browser.json
python -m eval.study_summary
```

BGE 复跑前需使项目 Python 同时可导入仓库依赖和 sentence-transformers，并能读取相同权重；本机通过 `site.addsitedir(sys.base_prefix + '/Lib/site-packages')` 使用已有的基础环境，设置 `HF_HUB_OFFLINE=1`、`OMP_NUM_THREADS=2`。OCR 另需 Office 解析依赖和 Tesseract。浏览器实验需构建前端并提供可运行的 Chrome 或设置 `CHROME_BIN`。批准使用的在线模型可用时，意图、策略与忠实度实验均可加 `--live` 复跑。测试标签和评审参考文档始终只供评分阶段使用；最近邻意图实验的原 100 条明确作为有标签索引。

[简历更新计划](22-resume-update-plan.md)

