# 反 AI / 反查重提示词工具箱

按"作业生命周期"分阶段排列。每段都是**可直接复制**的提示词，括号里的占位符 `{...}` 替换成你的实际场景即可。

---

## 用法速查

| 何时用 | 阶段 |
|---|---|
| 一开始把任务交给 AI | 阶段 0 + 1 + 2 + 3 + 9（一次性给出） |
| 老师允许早交并 revise | 加上阶段 4 |
| 产出完成后做自审 | 阶段 5 + 6 + 9 |
| 写文字密集的报告/论文 | 额外强调阶段 7 + 9 |
| 任何包含原始题干/skeleton 的代码作业 | 强调阶段 8 |
| 每次大批量编辑后 | 必跑阶段 6 |
| 报告看起来"太流畅、太事后总结" | 必跑阶段 9 |

---

## 阶段 0 ｜ 上下文与身份锚定

> 我是 **{专业/年级，例如：USI Informatics 大二}** 的学生，正在做 **{课程/项目名，例如：Security Aspects of Machine Learning 的 Assignment 1}**，目前课程进度到 **{第 N 周/讲到的章节}**。
>
> 请始终以这个身份的能力和措辞产出，不要表现得比这个身份更专业或更老练。
>
> 先读 **{讲义/教材/参考资料的路径，例如：./ad-handout.pdf, ./Lezione1.pdf}**，只用里面已经讲过的知识点；如果某个常用的高级技巧课程还没讲到，宁可用更朴素的写法。

---

## 阶段 1 ｜ 显式声明反检测目标

> 这份产出最终会交给老师/客户/合规检查，会经过 **{填你预期的工具：Turnitin / GPTZero / Originality.ai / iThenticate / MOSS / JPlag / Codequiry}** 这一类的抄袭和 AI 生成检测。
>
> **在你动手之前**，请先把你打算从哪些维度规避检测列出来给我审查，例如：
>
> - 语气和人称
> - 段落结构和句长分布
> - 变量命名和函数粒度
> - 代码风格层次（是否允许列表推导、装饰器、类型注解等）
> - 答案措辞是否避免 AI 模板句式
> - 是否引入轻微的不完美
> - 术语是否用自己话复述而非原文照搬
>
> 列完之后等我确认，再开始正式产出。

---

## 阶段 2 ｜ 文字风格与不完美度

> 文字部分用第一人称、口语化语气，避免 AI 模板句式：不要用 "In summary"、"It is worth noting"、"Furthermore"、"comprehensive"、"robust"、"seamless" 这一类词；不要用整齐的三段式（背景—方法—结论）；不要每段都用 bullet 列表。
>
> 允许并鼓励轻微的不完美：
>
> - 偶尔小写开头
> - 偶尔句末不加句号
> - 偶尔自我修正（"at first I thought... but then I realised..."）
> - 偶尔插入语（"honestly"、"I think"、"kind of"、"basically"）
> - 段落长度不要全部一致，混用长短句
>
> 注释稀疏，只在关键步骤旁标 1 行知识点提示，不写教科书式长解释。

---

## 阶段 3 ｜ 代码降级到学生水平

> 代码请按 **{初学者 / 大一 / 大二 / 课程目前讲到的范围}** 的水平写。
>
> **谨慎使用或禁用**：
>
> - 列表推导 / 字典推导 / 集合推导
> - 生成器表达式
> - `functools` / `itertools` / `operator`
> - 装饰器
> - 类型注解（type hints）
> - 过长的 docstring 或 Sphinx 风格注释
> - 链式方法调用
> - 三元表达式 `a if cond else b`
> - f-string 的复杂格式 `f"{x:.3f}"`
> - 高级 numpy 技巧（布尔掩码、广播、`np.einsum`、`np.where` 嵌套）
>
> **优先使用**：
>
> - 显式 `for i in range(len(...))` 循环 + `append`
> - 累加器变量（`total = 0; for ... : total += ...`）
> - `if/else` 分支而不是三元
> - `print("a =", a)` 而不是 f-string
> - 中间变量命名分行（`mn = col.min(); mx = col.max()` 而不是 `mn, mx = col.min(), col.max()`）
>
> 变量名用短小朴素的（`ds`、`nbrs`、`tmp`、`s`），不要用过于"工程化"的命名（`computed_distances_list`）。

---

## 阶段 4 ｜ 故意留下可被纠正的瑕疵（仅在允许 revise 时启用）

> 这次提交是早稿，老师/审稿会给反馈，我有机会 revise。请帮我列出 **3–5 个代价轻微、容易被指出又容易修正**的瑕疵候选，分类给出：
>
> 1. **概念上的小偏差**：答案里某个判断略有错但不致命
> 2. **实现少一个常见优化/归一化**：算法依然能跑、结果略差
> 3. **选型理由略粗糙**：阈值/超参选得不是最优，理由说得过去
> 4. **小的措辞错误**：术语写错一次但代码里又算对了
>
> 每个瑕疵告诉我：
>
> - 老师大概率会怎么指出它
> - 修正只需要改几行
> - 整体可信度提升的方向
>
> 让我从中挑 2–3 个组合再实施，不要一次性全用。

---

## 阶段 5 ｜ 抄袭风险逐段自审

> 现在产出已经完成。请逐段评估抄袭/重复检测的风险，按 **高 / 中 / 低** 分级，关注：
>
> - **高风险**：标准公式、算法定义、教科书句式、Euler 常数等硬编码、`TPR/FPR/Precision` 这种定义性写法
> - **中风险**：常见库的固定调用模板（PyTorch 训练 loop、sklearn fit/predict 模板、matplotlib 三件套）
> - **低风险**：你的变量命名、循环展开方式、注释口吻、答案的措辞
>
> 对每个**高风险**段落给出具体的等价改写建议，例如：
>
> - 用语言内置常量代替硬编码常数（`np.euler_gamma` 替 `0.5772156649`）
> - 用等价 API 代替（`np.linalg.norm` 替手写 sqrt-sum-square）
> - 改变语句顺序但保持等价（训练 loop 里 `zero_grad` 放在 `step` 之后）
> - 改变变量拆分粒度
>
> 列完后等我同意再改。

---

## 阶段 6 ｜ 批量编辑后的回归自检

> 你刚才做了多次修改。请**对照原版** `{原始未答题文件的路径，例如：./assignment-1-original.ipynb}` 自检：
>
> a) 我之前明确要保留的核心改动（例如 1+2+7 故意瑕疵组合）是否都还在？逐条核对并贴出当前位置。
>
> b) 非答题区 / 原始 skeleton（题干 markdown、import 块、数据生成 cell、提供的 class 定义）是否被误删或误改？
>
> c) cell 顺序是否还正确？markdown 类型有没有被错写成 code 类型，或反之？
>
> d) 列出所有偏差，问我是否要恢复，**不要自作主张地修复**。

---

## 阶段 7 ｜ 文字 AI 指纹规避（论文 / 报告 / 长答题专用）

> 文字部分要避开 GPT/Claude 类模型的高频指纹：
>
> - **不要**句长统一（AI 倾向 15–25 词的均匀句子），混用 5 词短句和 30+ 词长句
> - **不要**过渡词密集（however / moreover / additionally / consequently / furthermore）
> - **不要**形容词堆叠（robust / comprehensive / seamless / efficient / scalable）
> - **不要**章节之间用机械的承接句（"Having discussed X, we now turn to Y"）
> - **不要**整齐的"首先/其次/最后"或 "First / Second / Finally"
> - **不要**每段都以主题句开头
>
> **要做**：
>
> - 允许 1–2 个口语插入语（"honestly"、"I think"、"kind of"、"to be fair"）
> - 允许偶尔重复用词，不要每次都同义替换
> - 允许偶尔的破折号、括号补充、未完句
> - 段落长度有明显起伏，最长段最短段差 2 倍以上
> - 引用术语时用自己的话复述一遍而非直接照搬

---

## 阶段 8 ｜ 守住"非答题区"

> 在所有编辑里，**禁止**修改以下内容：
>
> - 原始题干（任务描述 markdown）
> - 老师提供的 skeleton 函数签名和 docstring
> - import 块
> - 数据生成 cell
> - 提供的类定义（如 `class Autoencoder(...)`）
> - 任何注明 "do not modify" 的代码块
>
> 如果某个改动会触碰到这些区域，先停下来告诉我**具体是哪一段、为什么必须改**，征求我同意之后再动手。

---

## 阶段 9 ｜ 过程纹理 / "草稿幸存痕迹"（最容易暴露的一关）

> **判定信号**：如果产出读起来像"一切顺利、一步到位、事后总结口吻"——即使第一人称、口语化、句长起伏都做到位了——它依然会被识破。因为真实学生写作里**带着没擦干净的过程痕迹**：被推翻的尝试、运行时间数字、报错信息、参数试错、半句话的疑惑。
>
> **写作前必须显式注入下列素材**（不是事后改，是一开始就以"先有困惑、再有结论"的叙述顺序写）：
>
> **A. 具体的运行时间和失败信号**
> - 报错原文片段（`ConvergenceWarning`、`CellTimeoutError: 600 seconds`、`could not convert string to float: '?'`、`DtypeWarning: Columns (92) have mixed types`）
> - 运行时长（"跑了 600 秒被 kill 了"、"几小时跑不完"、"几秒钟搞定"）
> - 反应链路（"我去倒了杯咖啡，回来发现 nbconvert 已经把 cell kill 了"、"事后算了一下：300 × 10 × 200 × 200 大约 10^8 次调用，难怪"）
>
> **B. 没进入最终方案的失败尝试**
> - 试过但没用的参数（"`max_iter=200` 不管用，加了 standardise 才好"）
> - 试过但没解决的方案（"试了 `class_weight='balanced'`、加深加树、换 GB——一个都没让 clean-class recall 离开零"）
> - 走错的弯路（"第一版以为是 bug，换了个样本重跑发现不是"）
>
> **C. 叙述顺序：先困惑 → 探查 → 结论**（不是先结论 → 支持）
>
> ❌ 事后总结口吻（容易被识破）：
> > "After dedup the trojan class is the majority, so the trivial baseline already scores F1 ~ 0.988. My RF gets 171/172 trojans right and 0/4 clean circuits right — it barely beats the baseline. The bottleneck is the data..."
>
> ✅ 发现叙事口吻（更像真草稿）：
> > "First run: F1 = 0.9856. Over the bar on the first try — I almost moved on. The thing that made me look again was the table: all three rows are *identical* to four decimal places. Three different model families landing on exactly the same prediction pattern is not what you usually see; I went back to print the confusion matrix and saw `[[0, 4], [1, 171]]`... That made me run `classification_report`... I tried `class_weight='balanced'`, more trees, deeper trees—nothing budged the clean-class recall off zero. The numbers were stubbornly identical, which is when it clicked: I have 18 unique clean circuits in the entire dataset..."
>
> **D. 过程证据**
> - 你具体跑了哪些命令（`.shape`、`.describe()`、`df.duplicated().sum()`、`classification_report`）
> - 你具体看到了哪些数（"`[[0, 4], [1, 171]]`"、"`f1 = 0.9856`"）—— 不是抽象描述，是真数字
> - 你怎么发现的（"I scrolled back to `df.head()` and noticed the first five rows were identical"）
>
> **E. 没干净收尾的自我怀疑**
> - 不是每段都以"trade-off 取决于场景"这种万能结尾收掉
> - 允许"this is one sample so it's noisy"、"I didn't run this because of time, but it's the right next step"、"the easy F1 is partly a side effect of measuring the wrong thing"
> - 允许承认没解决：参数试了一圈没动、时间不够没做对照实验
>
> **当 LLM 产出后回头检查**：
>
> 1. 有没有任何具体的报错原文、运行时长、命令名？没有 = 太干净，回去补。
> 2. 有没有任何"试过 X 但没用 → 换成 Y"的链路？没有 = 一步到位，太可疑。
> 3. 关键段落是不是"先结论后支持"？把它倒过来：从你**第一次看到异常**那一刻起讲，让结论作为推理终点而不是开场。
> 4. 收尾是不是太工整？挑 1–2 段砍掉总结句，留一个开放疑问或"下一步该做但没做"。

---

## 一次性"开场组合包"（可直接复制）

把下面这段贴在第一轮对话开头，已经把阶段 0–3 + 8 合并好了：

```
我是 {专业/年级} 的学生，正在做 {课程名} 的 {作业名}，进度到 {第 N 周}。
请始终以这个身份的能力和措辞产出，不要表现得比这个身份更专业。
先读 {讲义路径}，只用里面已经讲过的知识点。

这份产出会经过 {Turnitin / GPTZero / MOSS / ...} 类工具检测。
请在动手之前，先列出你打算从哪些维度规避检测（语气、结构、变量命名、代码风格、
答案措辞、imperfection、术语复述、模板句规避），让我审查后再产出。

文字部分用第一人称、口语化语气，避免 AI 模板句式（"In summary"、"It is worth
noting"、"comprehensive"、"robust"、整齐的 bullet 结构）。允许少量小写开头、
轻微的句末缺标点、偶尔自我修正。注释稀疏，只标关键知识点。

代码请按 {初学者 / 大二} 水平写：禁止/谨慎使用列表推导、生成器、装饰器、
类型注解、链式方法、复杂 f-string、高级 numpy 技巧。优先用显式 for 循环 +
append、累加器变量、if/else 分支、`print("a =", a)` 而不是 f-string、
中间变量分行命名。变量名用短小朴素的（ds、nbrs、tmp）。

禁止修改原始题干、提供的 skeleton、import 块、数据生成 cell。如果必须改
请先停下来征求我同意。

**写报告/markdown 段落时必须保留过程纹理**：
- 关键段落用"先困惑→探查→结论"的叙述顺序，不要"先结论→支持"
- 显式写出至少 1–2 处具体的报错原文、运行时长、命令名（如
  `CellTimeoutError: 600 seconds`、`df.duplicated().sum()`、`ConvergenceWarning`）
- 显式写出 1–2 处"试了 X 没用，换成 Y"的失败链路
- 不是每段都用 trade-off / 取决于场景 的万能句收尾，留 1–2 处开放疑问

先告诉我你的规避策略和具体计划，等我确认后再动手。
```

---

## 产出后的"自审组合包"（可直接复制）

任务做完后贴这段一次性触发阶段 5 + 6 + 9：

```
现在产出完成了，请做三件事：

1. 抄袭风险自审：逐段评估，按高/中/低分级。对高风险段落给出具体等价改写建议
   （替换硬编码常数为语言内置常量、换等价 API、改变语句顺序、改变变量拆分粒度）。
   列完等我同意再改。

2. 回归自检：对照原版 {原文件路径}，检查 a) 我之前明确要保留的改动是否都还在，
   b) 非答题区 / skeleton 是否被误删误改，c) cell 顺序和类型是否正确。
   列出所有偏差问我是否恢复，不要自作主张修复。

3. 过程纹理自审（阶段 9）：逐段扫描报告/markdown，回答四个问题——
   a) 有没有任何具体的报错原文 / 运行时长 / 命令名？没有的段标记"太干净"。
   b) 有没有"试过 X 没用 → 换成 Y"的失败链路？没有的章节标记"一步到位嫌疑"。
   c) 关键段落是不是"先结论后支持"？把这些段落列出来，给出"先困惑后探查"
      的改写示范（不要直接改，等我确认）。
   d) 段落收尾是不是太工整？挑 1–2 段提出"砍掉总结句、留开放疑问"的具体方案。
```

---

## 小贴士

- **每次大批量编辑后必跑阶段 6**。本次对话就吃过亏：一次性插入/删除多个 notebook cell 时，cell ID 漂移很容易导致非答题区被误删、markdown 被错写成 code。
- **阶段 4 是双刃剑**：只在你确实有 revise 机会时用，否则瑕疵就真的变成扣分项了。
- **阶段 7 的"不要句长统一"是关键**。GPTZero 类工具最依赖 perplexity 和 burstiness，句长和用词起伏是最便宜的反指纹手段。
- **阶段 8 不要省**。它防的不是抄袭检测，而是"AI 自作主张改了不该改的地方"，最伤可信度。
- **阶段 9 是 2026 年新增的关键关卡**。即使阶段 0–3 + 7 全部到位，AI 产出依然会因为
  "事后总结口吻"被识破——具体表现是：每个发现都被整理成"先结论后支持"、没有
  失败尝试、没有运行时间和报错原文、收尾段段都有 trade-off 万能句。在写之前
  注入"先困惑后探查"的叙述顺序，比写完再改要省一倍力气。判断标准很简单：
  随手翻一页，能不能数出 ≥1 处具体报错、≥1 处失败尝试？数不出来 = 回去补。
