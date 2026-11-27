# 功能真实性与面试边界审计（2026-11-20）

## 1. 结论与验证范围

当前代码是一个可以运行的 **LangGraph 电商客服工作流**，已经实现五策略路由、检索、结构化工具计划、受控写操作、生成后核验、离线回退、HTTP/SSE、前端和评测基础设施。不能把它描述为已经完成生产闭环的自主多智能体系统。

最重要的结论：

1. **完整多轮记忆未接通。** 历史存储正常，但历史没有进入意图、路由和生成提示词；已保存摘要未恢复。真实 DeepSeek 对话已复现第二轮遗忘订单号。
2. **当前环境没有运行 BGE / Cross-Encoder。** 缺少 `torch`、`sentence_transformers`，实际为 hash 向量和 lexical 精排。实现了模型适配器，不等于本次运行了真实模型。
3. **写操作基本链路与顺序幂等成立，但存在业务和并发边界。** 取消能落库、重复同一确认能回放；确认记录未消费、未持久化，多实例不能共享。退款缺少完整状态机和业务去重。
4. **忠实度确实由 LLM 判断，但会漏判适用范围。** 实测把电子产品的 15 日条款用于定制马克杯，检索充分性评估和忠实度裁判均放行。
5. **执行结果与用户回答可能不一致。** 取消已经落库，模型仍说无法确认完成；当前提示词和工具风险元数据存在冲突。
6. **SSE 是核验后的分块输出。** 模型生成过程没有实时 token 透传给用户。
7. **MCP 是本地固定函数与自定义 HTTP JSON 代理。** 未实现通用 MCP 初始化、能力协商与工具发现。
8. **身份认证与会话属主检查不完整。** 订单工具有归属检查，但请求体身份可伪造；复用他人的 session_id 能在内部读取其历史。

本报告逐项审阅 README 和相关技术手册中的主要功能声明，并执行已有测试、独立边界探针、真实模型场景和离线检索评测。“已有实现”“真实调用通过”“有限实现”“环境未启用”“未验证”分别说明；**有限测试无法证明所有输入和所有并发时序绝无问题**。

### 版本和对照依据

- 仓库：`https://github.com/JEen-wang/Adaptive-rag`。
- 本地 HEAD：`2a12a62382a8c3ec033fa9a5b19b9b23b8780ab8`。
- `git ls-remote origin HEAD` 返回同一提交。因而本地 `git show HEAD:README.md` 可作为远程该提交 README 的对照。
- 直接网页读取失败；没有将无法读取的网页当作证据。
- 本次审计对象包含已有未提交改动。开始时已有变更涉及 `.env.example`、README、意图覆盖、配置、DeepSeek、ChatService 和入门/配置文档，以及已有路由观察脚本、DeepSeek 测试。它们未被本次审计覆盖或回滚。
- 新增内容是审计脚本、本报告及 `output/` 下的证据；没有修复业务代码，以免混淆“原代码现状”和“修复后结果”。

## 2. 实际执行结果

| 验证 | 结果 | 解释 |
|---|---|---|
| 原有 pytest 全量运行 | **165 passed**，1 条依赖弃用警告 | Python 3.12.7；不是 165 次真实模型调用 |
| 原有测试覆盖率 | **71.94%** | `app` + `eval` 的语句覆盖率，不是功能正确率 |
| Ruff lint / format | 通过 | 新审计脚本也完成格式与 lint 检查 |
| mypy | 通过，123 个源文件 | 不证明运行时业务规则完备 |
| 新增边界探针 | **62 条观察记录** | 包含正常行为和缺陷复现，不应称“62 项全部功能通过” |
| 真实主链路 | 11 次对话请求 + 3 次直接裁判检查 | 44 次模型调用，24,841 tokens；另有 43-token 连通性探测 |
| 真实自纠错与摘要组件 | 4 次调用，631 tokens，通过 | 将错误的“支持”改为“不支持”，摘要正确保留订单号；与多轮集成情况分开 |
| 检索评测 | 65 条，4 组对照均完成 | 本次实际使用 hash / lexical，不能当作 BGE 结果 |
| 意图规则评测 | 100 条完成：关键词规则50%，覆盖层100% | 覆盖层覆盖全部100条；不是本次纯 LLM 100 条准确率 |
| 前端测试 | **2 个文件、3 个测试通过** | 覆盖 SSE 解码和消息渲染等有限范围 |
| 前端构建 | TypeScript + Vite 通过 | 已安装锁文件中的依赖；没有完整浏览器交互回归 |
| 三份 Compose 配置 | `config --quiet` 全通过 | 只验证配置，不证明容器已部署 |
| Docker 实际运行 | **未验证** | 本机 Docker daemon 不可用 |
| PostgreSQL / Redis / Qdrant 真服务 | **未验证** | 本次主要使用隔离 SQLite 与内存后端；适配代码/部分替身测试不等于真实多服务集成 |
| Python 3.11 与远程 CI 运行 | **未在本次复跑** | 仓库有双版本 CI 配置，本机实际测试是 Python 3.12 |

### DeepSeek 连通性与当前环境

配置了真实密钥，但原样使用当前 `deepseek_trust_env=false` 时直连失败，报 `DeepSeek transport error`。启用继承代理时，最初因缺少 `socksio` 而构造客户端失败。审计仅向本地虚拟环境安装 `socksio==1.0.0`，并在验证进程的 Settings 副本中设置 `deepseek_trust_env=true`，随后真实调用成功。**没有修改 `.env` 或业务配置文件。** 以后按原配置启动，仍可能遇到直连失败；依赖代理的环境应自行保持一致配置。

真实运行实际后端：

| 组件 | 实际后端 |
|---|---|
| LLM | DeepSeek `deepseek-chat`，真实调用 |
| 数据库 | 审计专用 SQLite；没有写入用户原有业务库 |
| 缓存 / 限流 | 进程内存 |
| 向量存储 | `InMemoryVectorStore` |
| Embedding | **HashEmbeddingProvider** |
| 精排 | **LexicalRerankProvider** |
| Unstructured | 当前未安装；Markdown 走后备解析器 |

## 3. 功能矩阵：入口、路由与 Agent

证据类型：“实测”指本次真实模型或本地运行；“探针”指受控输入/替身验证；“测试”指原有自动化测试；“审阅”指代码能确认的行为，未冒充外部实测。

| 功能 | 实际情况和边界 | 证据入口 |
|---|---|---|
| FastAPI JSON 聊天 | 已实现；正常请求、空消息 422、trace 返回均可用 | `app/api/routes/chat.py`、`tests/e2e/test_chat_api.py`、HTTP 探针 |
| SSE 聊天 | 已实现 started/node/token/answer/done；token 在图结束、记忆节点之后才发送 | `ChatService.stream`、`sse_event_order` |
| 真正生成中 token 透传 | Provider 层有 stream 实现，主聊天工作流未调用它；不能据此声称用户看到了实时模型输出 | `providers/deepseek.py`、`nodes/generate.py` |
| CLI 连续会话 | 有 HTTP / local 模式、命令和渲染，原有测试通过；连续保留 session_id 不等于模型有记忆 | `app/cli/`、`tests/unit/test_cli.py` |
| LangGraph 状态与分支 | 已实现共享 AgentState、条件边；五条策略真实可进入 | `app/agents/graph.py`、真实对话记录 |
| 多 Agent 团队自主协作 | 没有独立角色 Agent 间自主通信或协商；主体为固定图节点 | `build_graph` 审阅 |
| ReAct 观察后再规划循环 | 没有 executor→planner 回边；一次计划后执行，不会按每个执行结果重新规划 | `graph.py` |
| Checkpoint / 中断恢复 | `graph.compile()` 未配置 checkpointer；人工确认是新请求重新构造计划，不是恢复暂停的图 | `graph.py`、`planner_node` |
| 意图识别 | LLM JSON 分类，出错回规则；之后还会应用覆盖层 | `nodes/intent.py`、真实调用 |
| 意图覆盖层 | 有；命中规则且标签不同时直接覆盖，即便模型置信度高；存在否定句误判 | `intent_overlay.py`、overlay 探针 |
| 复杂度分类 | LLM 使用当前问题和意图；特定订单/政策规则可覆盖，失败走规则 | `retrieval/adaptive.py` |
| 复杂度和策略一致性 | 枚举可校验，但未联合校验映射；`simple + multi` 可被接收 | `route_inconsistent_pair_accepted` |
| 五策略 | DIRECT/SINGLE/MULTI/AGENT/REFUSE 均有路径；DIRECT 仍可能调用意图、路由、生成三次 LLM | 真实 direct 场景为 3 次调用 |
| Planner | LLM 返回自定义计划 JSON，Pydantic 解析；移除未知/不允许工具，最多保留六步 | `nodes/planner.py` |
| 原生模型 Function Calling | Registry 可导出 OpenAI tools schema，但主规划请求没有向 API 发送 `tools`，而是文本工具目录 + 自定义 JSON | `Tool.openai_schema`、`Planner.create_plan`、`DeepSeekProvider.generate` |
| 参数校验 | Tool.run 前用 input_model 校验；Planner 产生确认卡片前未完整校验参数 | `tools/base.py`、占位订单号实测/探针 |
| 缺参数追问 | 只检查部分写操作参数是否为空；非空占位符会绕过该检查 | `planner_placeholder_id_generates_confirmation` |
| 多步结果依赖 | 顺序执行静态参数；没有通用结果引用解析、动态参数绑定机制 | `executor_node` 审阅 |
| 工具调用预算 | `max_tool_calls` 限制有效；`max_agent_steps` 在循环内未累计本轮调用 | 设置 steps=2、calls=6 时实际执行 6 次的探针 |
| Planner Fallback | LLM 失败时按 Skill 生成计划或降为知识检索；可能无法保留完整用户目标 | `Planner.create_plan` |
| Executor Fallback | 映射中的只读工具失败可尝试其他读工具；不是无限重试/动态重新规划 | `READ_FALLBACKS`、原有测试 |
| 跨权限失败降级 | 订单归属错误可能变为一般政策检索；不能把后备检索成功理解为订单查询成功 | `get_visible_order`、`execute_with_fallback` |
| Skill 流程模板 | 有四个 JSON 模板，按触发词/意图匹配一个，并限制 Planner 工具集合 | `app/skills/`、原有测试 |
| Skill “按需加载” | 更准确是按需放入提示词；匹配函数仍遍历读取全部 JSON，不是文件层面的惰性加载 | `load_skills` / `match_skill` |
| 跨 Skill 复合任务 | 一次只匹配一个 Skill；查物流与库存等跨域任务可能被工具白名单限制 | `match_skill` / `_allowed_tools`；本次未把任意复合任务成功作为能力承诺 |

## 4. 功能矩阵：知识库与检索

| 功能 | 实际情况和边界 | 证据入口 |
|---|---|---|
| Markdown 入库 | 已实现，当前环境实际可用 | 39 篇知识文档、ingest 和检索实测 |
| 目录扫描 | 只扫描知识目录直接子文件，不递归扫描子目录 | `list_knowledge_files` |
| PDF/Office/HTML/图片 | 有 Unstructured 适配路径和替身测试；本机缺依赖，非 Markdown 会返回空块并记录日志；本次未验证真实 OCR | `unstructured_parser.py`、环境记录 |
| OCR/复杂表格 | 依赖 Unstructured、外部模型及系统包；代码保留部分表格 HTML，不保证识别正确或结构完全保留 | `elements_to_chunks` |
| 文本清洗 | NFKC、零宽/控制符、空白、中文硬换行等处理已实现；不把中文数字统一成阿拉伯数字 | `cleaning.py`、原有测试 |
| 精确去重 | 按内容 SHA1 去除相同文本块；跨文档相同文本只保留一个来源 | `dedupe_chunks` |
| 近重复 | 有 SimHash 距离检测，但仅日志，不删除、不进行 embedding 聚类 | `dedupe_chunks` |
| PII 脱敏 | 非 Markdown 入库块有脱敏；政策 Markdown 不走相同脱敏流程 | `_maybe_redact` |
| 切块 | Markdown 标题分段后约 500 字符窗口、60 字符重叠；不是精确 Token 切块 | `chunking.py` |
| 稳定文档 ID | 相对路径生成，跨目录位置稳定；chunk ID 按序号，编辑前文造成后续分块变更时不能保证旧引用仍指向同一事实 | `document_id_from_path` / `stable_chunk_id` |
| BM25 | jieba 等分词后排序；没有最低相关度阈值，完全不相关的问题也可能拿到 top-k | `bm25.py` |
| 稠密召回 | BGE/API 适配器已实现；当前 hash 仅占位，不应称语义检索质量已经验证 | `providers/embeddings.py`、本次评测 |
| BGE 型号 | 配置默认 `bge-small-zh-v1.5`；类名 `BgeM3EmbeddingProvider` 不代表实际用了 bge-m3 | `build_embedding_provider` |
| 混合召回与 RRF | 已实现 BM25 + 向量排名融合；两路调用当前顺序执行，不是并发召回 | `hybrid.py` |
| hash 降级的融合风险 | 当前仍把 hash 与 BM25 融合；本次 P@1 从纯 BM25 80.0% 降为融合 70.77% | 65 条检索实测 |
| 单路故障隔离 | 任一路有结果可继续；两路都为空也会报 RetrievalError，不区分“正常空结果”和“双路故障” | `HybridRetriever.retrieve` |
| Cross-Encoder | 本地适配器与推理路径存在，有注入模型测试；本次实际未加载，词面重排不是 Cross-Encoder | `providers/reranker.py` |
| Cohere | 未接入 Cohere API；README 简历原稿有该词，但其后已有未接入的说明 | README、providers 审阅 |
| 查询改写 | LLM 返回最多三个改写；实际 SINGLE/MULTI 每轮最多保留原问题及一个额外改写做检索 | `query_rewrite.py`、`nodes/retrieval.py`、`self_rag.py` |
| SINGLE | 一轮流程也可能执行两个检索 query，再去重和精排；不是严格一次底层 search | `retrieval_node` |
| MULTI / Self-RAG | 只有 MULTI 进入循环；相关/充分/缺失方面由 LLM 判断，默认最多三轮 | `self_rag.py`、真实 multi 场景、原有测试 |
| Self-RAG 停止 | 实际仅根据 `sufficient` 停止；`relevant` 字段没有单独控制流程；达到上限仍返回累计块，不向下游强制传递“不充分”标志 | `SelfRAGLoop.retrieve` |
| Self-RAG 故障 | 判定 LLM 不可用时有块即视为充分；保证可继续不等于保证证据充分 | `selfrag_judge_outage_nonempty` |
| Self-RAG 论文一致性 | 应用层 LLM JSON 反思循环；没有训练论文中的反思 token 模型 | 实现审阅 |
| Agent 内知识检索 | search_knowledge 直接混合召回 top4，内容每块最多 400 字；不自动使用同一 Self-RAG/精排链 | `tools/knowledge_tools.py` |
| 引用绑定 | 生成 payload 中的 chunk ID 只允许映射到已检索块；不能保证句子与引用逐一对应，也未强制每句事实都有引用 | `generate_node` |
| Agent 工具政策引用 | 工具返回块位于 tool_results，不自动加入 retrieved_chunks；可能正文出现 ID 但结构化 citations 为空 | 真实 memory_followup、`generate_node` |
| 政策缓存 | query+top_k、TTL；订单样式 query 跳过。缓存 key 无知识版本；更新后可能短期读取旧内容 | `retrieval/cache.py`、原有测试 |
| 知识更新 | 每次启动全量 ingest；无增量更新/删除 API；Qdrant upsert 不自动删除已经消失的旧块 | `ingest.py` / `vector.py` |
| 政策版本 | payload 有 policy_version，但召回未按版本过滤，也没有多索引原子切换 | `QdrantVectorStore.search` |

## 5. 全部注册工具的实际能力

本次启动初始共 **17 个工具：15 个业务工具 + 2 个本地 MCP 代理**。审计使用隔离数据，逐个调用了全部 17 个工具。下面的“正常执行”只证明示例输入可用，不能替代上层权限、参数和业务状态审核。

| 工具 | 当前实际行为 | 边界 |
|---|---|---|
| get_order | 读取数据库订单与基本状态 | 归属匹配请求中的 user_id；身份本身未认证 |
| get_order_items | 读取订单商品及 customized 等字段 | 不自动依据政策判断售后资格 |
| track_shipment | 根据本地订单 paid/shipped/delivered 时间拼装物流事件 | 没有真实承运商轨迹、实时物流 API |
| get_refund_status | 查询本地 RefundRow 列表 | 不查询支付渠道到账状态 |
| create_refund_request | 新建退款申请并把订单改为 refunding | 无支付退款；缺完整状态检查和订单级业务去重 |
| create_return_request | 新建退货申请 | 不自动退款、无真实寄回物流；未完整校验品类/时效/重复申请 |
| cancel_order | 仅允许 created/paid 状态取消，并写审计 | 执行时会重新读状态；未使用行锁/条件更新保证并发原子性 |
| search_products | 对名称、品类、标签做 SQL ILIKE，最多约五条 | 长句、预算和多条件理解不在此工具内 |
| get_product_detail | 根据本地 SKU 查详情 | 不是外部商品平台查询 |
| recommend_products | 用 category/query 做同类 SQL 搜索，取三条 | 无个性化模型，也没有真正按预算排序/过滤 |
| get_coupon | 返回固定示例券规则 | user_id 未用于个性化；不查询账户券、不执行领券 |
| check_inventory | 读取本地库存字段 | 无库存锁定、预占或真实仓储同步 |
| get_warranty | 对任意通过格式校验的 SKU 返回固定 12 个月及免责说明 | 探针使用不存在 SKU 仍返回 12 个月；不能称为真实商品保修查询 |
| search_knowledge | 返回检索文本块 | 不是订单事实、不是完整 Self-RAG；结构化引用未自动接通 |
| escalate_to_human | 拼装 ticket_id 并返回 human 队列字段 | 没有持久化工单或外部人工队列；同一会话 ID 前缀可能返回同一票号 |
| mcp_store_hours | 返回内置营业时间字典 | 不是外部实时营业信息 |
| mcp_warehouse_cutoff | 返回内置截单时间字典 | 不能保证实际仓库当日发货 |

### Skill 与 MCP 的准确描述

- Skill 是业务模板、触发词和工具白名单，不是可任意执行代码的技能平台。
- 本地 MCP 代理确实能执行并返回结果，不是空壳。
- 远程代理只是固定 endpoint 上 POST `{name, arguments}`，没有 MCP 协议初始化、工具发现、JSON-RPC 会话或通用认证协商；没有用真实远端 MCP 服务验收。
- MCP 通用参数只有 `payload` 字典，无法自动继承远端工具的精确输入 schema。
- EXTERNAL 风险不在 `WRITE_RISKS` 内。转人工和 MCP 代理默认不会像 WRITE/DESTRUCTIVE 一样要求确认；探针已证实转人工计划不产生 pending_action。

## 6. 写操作、确认、幂等与业务边界

### 已验证成立的部分

1. 用户明确请求取消未发货订单，系统生成 pending_action；未确认时不会执行写计划。
2. 确认 ID 与会话、用户和 15 分钟有效期绑定；错误/过期确认会拒绝。
3. 确认后使用已保存的工具名与参数，不重新让模型自由决定该写操作。
4. 工具调用前参数校验，订单读取检查归属。
5. 同一确认 ID 顺序重复请求时，数据库中的幂等记录能够回放同一结果。
6. 真实取消 ORD10003 后，数据库状态为 cancelled，只有一条对应取消审计；重复确认未再次写审计。
7. 取消时会重新查询订单状态，已发货状态不满足取消条件。

### 已复现或已确定的限制

| 问题 | 证据 / 实际后果 |
|---|---|
| 确认记录不消费 | 同一 ID 在有效期内可反复 resolve；不等于一次性 token |
| 确认记录不持久化 | 新 worker 的字典为空；重启或请求落到另一副本后无法确认 |
| 幂等不等于业务去重 | 新 action_id / 新 key 可以再次对同一订单申请退款或退货 |
| 并发幂等不完备 | get→execute→put 不是预占锁；有唯一键不等于外部副作用 exactly-once。本次没有跑真实多进程并发压测 |
| 失败结果被记为成功 | 自定义写工具返回 success=false 后仍被写入幂等表；回放被构造成 success=true。受控探针已复现；当前内置写工具常以异常表达失败，因此这是扩展接口路径的缺陷 |
| 退款缺状态机 | 已取消订单仍能新建退款申请；继续再次申请得到不同 refund_id，探针已复现 |
| 售后资格没有硬约束 | 政策/模型建议不是仓储约束；没有统一检查时效、定制品资格、可退金额、重复申请等 |
| 缺订单号追问可被绕过 | 真实模型填入 `<用户提供的订单号>`，非空检查通过，生成了无效确认卡片；执行时格式校验仍会失败，不能说已经误取消了真实订单 |
| 确认前并不保证订单可执行 | pending_action 时整个 executor 直接跳过，原计划中的读步骤也不执行；确认卡片不是业务可执行性审核结论 |
| 直接调用 Tool.run | 权限确认主要位于 executor；调用底层工具时不能假设它独立完成 HITL。工具内订单归属检查仍在 |
| 输出风险元数据不一致 | 部分写工具返回的 ToolResult 未赋 risk，默认变成 READ；真实回答据此质疑取消完成 |

真实取消后的异常回答不是“数据库未执行”。`live-db-evidence.json` 已证实数据库执行成功。生成提示词仍要求“对取消/退款只说明将发起确认，不要声称完成”，与已经确认且执行成功的场景冲突。

## 7. 忠实度、自纠错与安全

| 能力 | 当前事实 | 边界 |
|---|---|---|
| 生成后忠实度 | LLM 同时阅读问题、回答、最多八块文档、工具结果；返回 supported 和批评 | 是证据支持判断，不是关键词/向量相似度 |
| 简单否定识别 | 真实裁判正确区分“支持”与证据中的“不支持” | 仅证明该例成功，不证明所有否定/复杂条件正确 |
| 工具结果核验 | 真实裁判能拒绝把 requested 说成“退款到账” | 不检查数据库与外部系统是否本身正确，也不能回滚副作用 |
| 适用范围核验 | **实测漏判**：电子产品 15 日条款被用于定制马克杯，裁判仍 supported=true | 检索相关性、事实出处和条款适用范围是不同问题 |
| Self-RAG 的充分性 | 同一案例先被 sufficient=true 放行 | 多次同模型判断可能共享同一错误 |
| 自纠错 | 有 LLM 重写一次→复判→抽取回退→再判，最后可拒答 | 不会无限纠错；自纠错后的引用没有完整刷新 |
| 引用更新 | 探针改写回答成功，但返回 dict 没有 citations，旧引用会保留 | 不能承诺改写后引用与新事实严格对应 |
| 核验器故障 | supported=false/checker_unavailable 后用受控离线回答 | 不会一律返回 HTTP 503；有时仍返回 200 的原文节选 |
| DIRECT / 离线 | 跳过 LLM 忠实度；等待确认、缺订单号和拒答也跳过 | “所有答案都经裁判检查”不成立 |
| 空检索 | 在线 SINGLE/MULTI 在无块时拒答 | 有块但不相关不等于已经证明证据充分 |
| 注入检测 | 用户 query 正则拦截已知表达；真实攻击测试零模型调用、直接拒答 | 不保证隐蔽注入、编码混淆和文档攻击都能防住 |
| 文档隔离 | 提示词将证据/工具标为不可信数据 | 是软约束，不是可证明的指令隔离机制 |
| 业务范围 | 意图 OUT_OF_SCOPE、覆盖层、路由实现部分边界 | `assert_in_scope` 函数没有接到主图；不能把该独立函数的测试当作主路径全覆盖 |
| 身份 | 无 JWT/登录认证，接受请求体 user_id | 订单归属校验正确也无法抵御伪造同一 user_id |
| 会话属主 | get_or_create 发现已有 session 不比较 user_id；recent_messages 仅按 session_id | 探针换用户读取到已有历史；当前历史未送 LLM 不等于隔离问题不存在 |
| 管理接口 | production 无 token 关闭、配置 token 则常量时间比较 | development/testing 无 token 时开放是代码当前行为 |
| 日志脱敏 | 有 SecretStr、部分 PII 正则、mapping 脱敏函数 | 日志 Filter 主要处理字符串 msg，不等于递归处理 args/extras 和任意密钥文本；不要承诺日志绝无敏感内容 |

忠实度和 correctness 不等价：回答可能逐字来自证据，但证据选错品类、过期或缺必要条件。当前裁判没有独立可信政策引擎。

## 8. 多轮记忆专项验证

### 真实对话

同一 session_id=`live_memory`：

1. 用户：“请记住，我要查询的订单号是 ORD10002。”
2. 系统调用工具查到该订单，并返回已签收信息。
3. 用户：“这个订单现在是什么状态？”
4. 系统回答：“本次未提供有效订单号，请补充订单号”，随后混入了一般政策说明。

数据库中两轮的四条消息都存在。因此这是 **历史没有进入决策和生成上下文**，不是历史未保存。确认操作可以依赖 action_id 恢复具体参数，但这是一条专门的确认协议，不代表一般多轮记忆已实现。

### 逐环节检查

| 环节 | 当前情况 | 验证 |
|---|---|---|
| 用户/助手消息持久化 | 正常；每轮 append 两条原始消息 | 真实数据库与探针 |
| 历史读取 | recent_messages 默认读最近 20 条消息，不是 20 轮 | 代码和状态检查 |
| 摘要持久化 | 非空摘要有 save_summary 路径 | 仓储代码/探针 |
| 摘要恢复 | **未接通**，initial_state 始终 summary="" | 预先写摘要后再初始化，仍为空 |
| 意图使用历史 | **没有**，只传本轮 query | 提示词捕获与代码 |
| 复杂度路由使用历史 | **没有**，仅本轮 query + intent | 提示词捕获 |
| Planner 使用历史 | 不接 messages；模板有 summary 字段，但它通常因初始化为空而无信息 | 提示词捕获 |
| 回答生成使用历史 | **没有**，仅本轮 query、意图、策略、证据、工具结果 | 提示词捕获、真实失忆案例 |
| Token 估算 | 字符数/1.5 的启发式；100 汉字与100英文字符都估66 | 探针；不是模型官方 tokenizer |
| 预算覆盖范围 | 只计算 messages+summary；未把当前证据、系统提示词和工具 payload 统一计入 | 大工具 payload 探针仍判 none |
| soft 50% | 主要截短工具结果到400字符 | 组件实现和测试 |
| hard 70% | 工具160字符 + 最近八条消息滑窗 | 不会将全部旧对话自动改为高质量要点 |
| auto 85% | 有 LLM 摘要调用，但先裁剪历史后再摘要 | 探针原有20条，摘要器只收到最近8条，早期标记已丢 |
| 压缩时机 | 在本轮生成与核验之后 | 不能保护已经发出的本轮模型请求免于超限 |
| 压缩结果持久化 | `_persist` 仍追加原始本轮消息；压缩后的完整 messages 不替换数据库历史 | 状态压缩不等于持久化历史被压缩 |
| 跨轮连续摘要 | 已保存摘要没有恢复，也没有完整逐轮摘要累积闭环 | 代码审阅与探针 |
| 超长单条消息 | 滑动窗口不限制单条消息长度；不保证压缩后一定低于预算 | 无最终强制预算断言 |
| 长期用户记忆 | 没有独立用户事实库、跨会话事实检索或记忆冲突处理 | 结构审阅 |

正确的面试表述：**“实现了会话存储和预算/压缩组件，但当前多轮上下文接入尚未闭环；我通过同会话追问和提示词捕获定位到了历史与摘要的断点。”**

不能表述为“128k 上下文完整支持、历史自动压缩且不会丢失重要事实”。128000 只是当前默认预算配置，并非所有请求的实际预算保证。

## 9. 工程、降级、部署与前端

| 功能 | 当前情况 / 边界 |
|---|---|
| 依赖装配 | DI 容器、request-scoped AsyncSession 与 contextvars 已实现；避免把单个会话绑定到长期 Graph |
| DeepSeek 超时重试 | generate 有拆分超时与指数退避/full jitter；固定不可重试状态会停止。stream 不做同样的中途重试 |
| 故障并非统一503 | 多个节点捕获 LLMProviderError 后返回规则/节选/离线结果；文档“运行中模型故障503”过度概括 |
| Redis 启动故障 | 连接失败选内存后端；不是保有多副本全局语义 |
| Redis 运行期故障 | 限流 fail-open，缓存 miss/忽略 set；不自动更换为完整内存限流实例 |
| 限流 | 按请求体 user_id；伪造用户可绕过“每用户”限制；本机内存限流不跨副本 |
| Qdrant 启动故障 | ingest 失败回退内存；运行中搜索失败通常由 Hybrid 使用 BM25 继续，不等于动态重建内存向量库 |
| 远程 Embedding 故障 | 配置远程 provider 后 embed 请求错误会抛出；启动 ingest 不保证自动换本地/hash |
| 已加载模型推理故障 | Cross-Encoder 的“加载失败回 lexical”不等于任何 predict 异常都自动回退 |
| capabilities | 有接口；retriever/llm 一部分基于配置推导，无实际 dense_backend 字段；配置本地模型却 hash 降级时仍可报 hybrid |
| readiness | 只把 DB 当硬依赖；LLM ping 主要检查密钥是否存在，离线 provider ping 也可为true，并标 configured；不证明 API 能请求成功 |
| 健康状态码 | /health 存活；/ready 返回对象中的 ready 字段。不能仅凭HTTP200判断所有依赖已就绪 |
| 日志与 trace | JSON日志、trace_id/request_id、请求关联已实现；不是 OpenTelemetry 全链路 span 平台 |
| Prometheus | 请求、工具、策略、阶段延迟、token和幻觉计数已实现；进程内指标不自动聚合所有副本 |
| Token 计量 | state 主要累加生成节点 usage；意图、路由、改写、规划、裁判、自纠错没有完整累计。真实确认提案用了3次LLM但响应tokens=0 |
| 延迟计量 | 有部分阶段计时；规划/核验未完整独立计时，rerank_ms经常为0而耗时归在retrieval_ms，不是全节点成本账本 |
| SQLite/PostgreSQL | SQLAlchemy 适配、连接池配置存在；本次实际仅 SQLite |
| 数据迁移 | 启动 create_all；有 Alembic 文件但非默认启动迁移路径，也未本次验证升级/回滚 |
| 原子业务写入 | 仓储采用请求内事务，可覆盖本地多表写；不等于分布式事务、并发锁或外部支付一致性 |
| Docker 三档 | 文件与语法验证成立；未起 daemon，不能宣称本次容器端到端通过 |
| 多实例确认 | 两个API副本可配置，但进程内pending_actions使确认语义不共享；“可部署多实例”不等于业务链路跨实例正确 |
| CI | 配置有锁文件检查、Python3.11/3.12、Ruff/mypy/pytest/70%覆盖率；当前工作流没有前端测试或Docker构建步骤 |
| 锁文件一致性 | CI中有 `uv lock --check`，本机未执行该项，不把静态配置当作通过结果 |
| 演示数据 | 默认启动不写入；seed=True显式创建，已有数据不覆盖；初始示例状态不保证与反复演示后的状态一致 |
| 前端聊天/确认卡片 | 实现并构建通过；3个单测不足以证明全部浏览器交互、断线和点击竞态正确 |
| 前端会话 | localStorage持久化、数量限制、恢复；不是服务端会话列表同步，不支持跨设备恢复 |
| 前端删除/取消 | 删除会话、关闭确认卡片主要是本地UI动作；不等于删除服务器历史或撤销服务器pending action |
| 前端停止生成 | AbortController取消请求；不保证此前已经执行的写操作被撤销 |
| 前端状态展示 | App主要调用/health显示连接状态；虽有getCapabilities函数，主界面未完整展示实际模型降级 |
| 前端节点进度 | 服务端推node事件，但useChat主要处理token/answer；不宜声称前端已完整呈现每节点进度 |

## 10. 仓库声明与实际实现的差异

需要公平区分：README 的“简历写法”包含较强表述，但其“仓库实现说明”和“路线图”也已经承认部分缺口。下面不是把已公开声明的路线图缺口说成刻意隐瞒。

| 仓库相关表述 | 审计后的准确结论 |
|---|---|
| 多智能体电商客服 | 固定 LangGraph 多节点工作流 + LLM决策节点；无自主多Agent团队 |
| Function Calling统一工具 | 能导出schema并自行调度；主LLM请求走文本目录+JSON计划，不是原生tools调用 |
| 128k与三级压缩 | 组件有，端到端记忆未接通；预算计算也不完整，先裁再摘要会丢信息 |
| Skill按需加载 | 提示词选择单Skill和工具子集；加载文件仍全量遍历 |
| MCP可插拔 | 自定义本地/HTTP代理可执行；非完整通用MCP协议客户端 |
| Cohere精排 | 未接入；README自身已有澄清，不应写成个人已完成事项 |
| 实际Embedding/精排后端可见 | capabilities只部分准确，缺真实dense_backend；当前hash退化没有被精确标明 |
| 写操作有权限/确认/幂等/审计 | 机制存在；身份、持久化、一次性消费、并发语义和售后状态机仍不足 |
| 流式输出 | SSE成立；生成后分块，不是实时逐token生成 |
| 模型故障运行中503 | 节点大多会自行降级，常返回200；不能泛化 |
| 幻觉0%/<4% | 是历史特定评估口径；本次已经复现错误品类条款被裁判放行 |
| 多实例部署 | Compose存在；确认记录在内存，跨实例会丢失待确认状态 |
| 前端/工程化完成 | 构建可通过，但前端节点进度、降级展示、服务端会话同步等未完整接通 |

## 11. 数字、实验口径与本次检索结果

65条检索评测，本次实际结果：

| 方法 | P@1 | R@3 | MRR |
|---|---:|---:|---:|
| BM25 | 80.00% | 90.00% | 0.8556 |
| hash-vector | 3.08% | 6.92% | 0.0518 |
| BM25 + hash + RRF | 70.77% | 82.31% | 0.7872 |
| Hybrid + lexical fallback | 80.00% | 90.00% | 0.8590 |

不能把本次最后一行标为 Cross-Encoder，也不能拿来证明 README 中 BGE 历史成绩已经在本机复现。当前结果反而证明：**不可靠向量与BM25强行融合会拖累召回排序。**

历史指标注意：

- README“简历原稿”的 Precision 58→85、Recall63→91、Faithfulness54→88 等，不是本次实测数字；README后文已有口径说明。
- `eval/MEASURED.md` 的历史 BGE/Cross-Encoder 与同模型裁判成绩，缺随仓库分发的原始JSON，本次未重现同一模型配置；不能当作自己的复现实验。
- `docs/05-agents-and-tools.md` 中 RAG幻觉1.5%与MEASURED中的3.1%存在口径/历史版本不一致，面试应明确引用具体报告，不能拼接使用。
- 关键词命中采用 all(word in answer)，不能处理否定和条件关系，不等于答案准确率。
- 端到端 tool_path_hit 主要按回答含ORD/库存或存在pending_action判断，不核对全部真实工具调用和数据库效果；不能把该数值称为工具执行正确率。
- 意图分类器会捕获LLM故障并回规则。评测报告的 backend=deepseek 和 errors=0，不足以证明每一条样本都成功使用了模型；需要调用层日志/统计。
- 生成实验的无检索回答也被拿去与检索出来的证据比较，测的是对这组证据的支持关系，不是一般世界知识正确率。
- 本次没有重新跑65条真实生成、300条多版本意图模型对比或生产压测；已执行的真实场景和本地评测分别保留证据，不外推为全量准确率。

## 12. 面试可以怎样说

### 可以有证据地陈述

> 我基于LangGraph学习并验证了一个电商客服工作流，包含五策略路由、混合检索、Self-RAG风格的补检索、工具规划与执行、人工确认和顺序幂等。我使用真实DeepSeek验证了主要路径，并结合提示词捕获、数据库证据与受控故障测试检查了边界。
>
> 当前项目不是完整生产系统。多轮历史与摘要尚未进入决策和生成；确认记录还在进程内，售后状态机不完整；忠实度裁判能识别简单矛盾，但会漏掉条款适用范围问题。我能明确指出这些问题的触发条件和修复方向。

根据真实参与情况，把“学习、验证、修改、独立实现”分开表述。仓库已有能力不应改写成自己的独立贡献。

### 不应承诺

- 完整长期记忆、128k全上下文预算保证、无损压缩。
- 自主多智能体协作、完整ReAct闭环、checkpoint恢复。
- 原生Function Calling主链路、通用MCP协议已经完整对接。
- 当前本机已运行BGE/Cross-Encoder，或已经接入Cohere。
- 退款真实到账、真实承运商物流、真实人工客服工单。
- 完整鉴权、多租户隔离、多副本HITL可靠、并发exactly-once。
- 忠实度100%即真实零幻觉、关键词命中即业务成功。
- 历史README数字就是自己本次跑出的成绩。

### 修复优先级

1. **P0 身份与会话隔离**：可信身份绑定，已有session属主校验。
2. **P0 业务执行真实性**：退款状态/金额/重复申请约束，修复确认后生成提示词与risk元数据。
3. **P1 多轮记忆闭环**：恢复摘要，生成/路由/规划接入预算化历史；先摘要后裁剪，统一计算全部模型输入。
4. **P1 确认与并发幂等**：持久化pending，一次性消费与状态机，原子预占/冲突处理，失败结果不能回放成成功。
5. **P1 检索与裁判质量**：真实语义模型、退化hash不参与融合或降低权重、政策品类/版本过滤、针对性反例评测。
6. **P2 可观测性**：全节点Token/耗时计量、真实后端标识、自纠错引用更新、前端降级和进度展示。
7. **P2 交付验证**：真实外部服务、Docker、多实例、浏览器端到端、迁移升级回滚与断线恢复。

## 13. 证据与复跑

所有路径相对于仓库根目录：

| 文件 | 内容 |
|---|---|
| `scripts/audit_features.py` | 独立审计脚本，默认只跑隔离边界；--live调用真实API，--correction验证真实自纠错/摘要组件 |
| `output/feature-audit-20261120/boundary-results.json` | 62条观察，包括17个工具、记忆、确认、幂等、路由、HTTP/SSE |
| `output/feature-audit-20261120/live-connectivity.json` | 不含密钥的连通性对照 |
| `output/feature-audit-20261120/live-results.json` | 11次真实对话与3次裁判检查 |
| `output/feature-audit-20261120/live-prompts.json` | 实际请求提示词、返回与usage，使用审计合成数据 |
| `output/feature-audit-20261120/live-usage.json` | 44次主链路调用计量 |
| `output/feature-audit-20261120/live-backends.json` | capabilities与实际后端对照 |
| `output/feature-audit-20261120/live-db-evidence.json` | 取消只写一次、历史存储等数据库证据 |
| `output/feature-audit-20261120/live-self-correction.json` | 真实自纠错与摘要组件补充结果 |
| `output/feature-audit-20261120/retrieval-results.json` | 本次65条检索四组对照 |
| `output/feature-audit-20261120/intent-results.json` | 本次100条规则与意图覆盖层评测 |
| `output/feature-audit-20261120/compose-validation.json` | 三份Compose配置的静态验证 |
| `output/feature-audit-20261120/environment.json` | 版本、依赖存在性和审计时工作区状态，无密钥 |
| `output/feature-audit-coverage.json` | 原有测试的详细语句覆盖率 |

在仓库根目录复跑：

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=app --cov=eval --cov-report=term:skip-covered --cov-fail-under=70
.\.venv\Scripts\python.exe -m scripts.audit_features
.\.venv\Scripts\python.exe -m scripts.audit_features --live
.\.venv\Scripts\python.exe -m scripts.audit_features --correction
.\.venv\Scripts\python.exe -m eval.runner --task retrieval
.\.venv\Scripts\python.exe -m eval.runner --task intent
```

--live和--correction会消耗API额度；模型输出可能变化，不要求复跑得到逐字相同回答。审计live数据库仅用于合成订单；脚本在原live.db已存在时新建带时间后缀的审计数据库，避免前次已取消状态影响场景。没有为了得到更好结果而修改业务代码或覆盖历史评测数字。
