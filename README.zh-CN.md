<div align="center">

# attention-residual-routing

**能否用注意力残差信号学习按输入决定的深度路由？**
**Can attention-residual signal learn per-input depth routing?**

![Status](https://img.shields.io/badge/status-dormant-lightgrey)
![Language](https://img.shields.io/badge/language-Python-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)
![Closure](https://img.shields.io/badge/closure-2026--03-blue)

[한국어](./README.md) · [English](./README.md#english) · **中文**

</div>

> 🧊 **这是一个处于休眠状态的研究探索性项目。**

## 这项研究想看的是什么

语言模型内部的各层（layer）并不是对所有输入都做同样多的工作。有些句子要走到很深的层才能解决，有些句子在很浅的层就够了。那么，**能不能读取模型内部的某个信号，判断"这个输入是否需要深层"，在不需要的时候跳过部分层呢** —— 这就是出发问题。

核心假设有三个：

- 模型内部的某个信号（注意力残差类信号）能透露出每个输入"需要多少深度"。
- 可以根据该信号为每个输入做出不同的跳层决定。
- 由此既能保持质量，又能减少计算量和实际推理时延。

我们从小规模开始，逐步扩展到更大的模型和更多样的数据集（WikiText、TinyStories、OpenWebText、FineWeb-Edu、CC News 等）来验证。

## 我们发现了什么

- **信号本身确实存在。** 按深度测量利用度时，能看到它会随着输入而发生有意义的变化。
- **但最初设想的"在任何数据集任何设定下都通用的跳层策略"没能站住脚。** 一旦把信号换成实际的路由策略，往往打不过"总是跳同样几层"的静态基线。
- **不过有一个窄窄的结果坚持到了最后。** 在某一个特定数据集（`cc_news`）上，跨随机种子可复现的小幅质量优势始终留存了下来。即使在新的数据切分（lockbox，"上锁箱"）上，这个窄优势也依旧成立。
- **但实际推理速度并没有变快。** 质量稍好，但因为动态跳层本身有开销，wall-clock（端到端时延）反而比静态策略更慢。

更详细的结果请见：

- 🇰🇷 [`reports/PROJECT_FINAL_REPORT_KO.md`](reports/PROJECT_FINAL_REPORT_KO.md)
- 🇬🇧 [`reports/PROJECT_FINAL_REPORT_EN.md`](reports/PROJECT_FINAL_REPORT_EN.md)

## 为什么先放一放

信号确实存在，并且在一个窄范围内留下了可复现的质量优势。但我们没能把它变成一个**真正更快的系统**，更广的泛化性（向其他语料的迁移）也没拿到。等下一次有新的刺激（不同的数据、更轻的 selector、不同的路由几何结构）再唤醒它，会比继续沿当前路径推进更自然。

## 重新打开时，先看哪里

- 📖 [`GLOSSARY.md`](GLOSSARY.md) —— 解码源代码与最终报告里出现的内部术语（数据集别名、子层掩码、路由分数模式、配置轮次、lockbox 等）。
- 🇨🇳 这份 README（中文）
- 🇰🇷 [`reports/PROJECT_FINAL_REPORT_KO.md`](reports/PROJECT_FINAL_REPORT_KO.md) —— 韩文版最终报告
- 🇬🇧 [`reports/PROJECT_FINAL_REPORT_EN.md`](reports/PROJECT_FINAL_REPORT_EN.md) —— 英文版最终报告
- [`src/attnres_routing/`](src/attnres_routing/) —— 模型、路由、数据、分析的核心库
- 后期 config 文件夹（`configs/scale_heterogeneity_v7..v9/`）里的 YAML 文件，最能体现最终阶段的实验形态

## 代码地图

| 文件 | 它在做什么 |
|---|---|
| [`src/attnres_routing/model.py`](src/attnres_routing/model.py) | 解码器模型 + 注意力残差栈（标准 / `block_attnres` 两种残差模式） |
| [`src/attnres_routing/routing.py`](src/attnres_routing/routing.py) | 决定跳哪几层的路由逻辑、解码计时、KV-cache 过滤 |
| [`src/attnres_routing/analysis.py`](src/attnres_routing/analysis.py) | 把记录下来的深度权重整理成 per-source utility 和排序统计量 |
| [`src/attnres_routing/data.py`](src/attnres_routing/data.py) | 数据集别名、加载、tokenize、按 `seq_len` 打包 |
| [`src/attnres_routing/sequence_manifest.py`](src/attnres_routing/sequence_manifest.py) | 训练 / 验证 / lockbox 切分的 manifest 构建 |
| [`src/attnres_routing/sublayer_masks.py`](src/attnres_routing/sublayer_masks.py) | `SublayerMask`、`action_types`、`to_id()` 编码、候选枚举、FLOP 估算 |
| [`src/attnres_routing/normalizers.py`](src/attnres_routing/normalizers.py) | 深度轴归一化器（`softmax` / `sparsemax` / `entmax15` / `topk_softmax`） |
| [`src/attnres_routing/train.py`](src/attnres_routing/train.py) | 语言模型预训练循环（DDP、AMP、cosine LR、STP 正则化等） |
| [`src/attnres_routing/utils.py`](src/attnres_routing/utils.py) | 种子、YAML I/O、目录创建、HF token 解析、cosine LR、参数计数 |
| [`scripts/train_lm.py`](scripts/train_lm.py) | 基础训练入口 |
| [`scripts/evaluate_functional_oracles.py`](scripts/evaluate_functional_oracles.py) | 以"理想策略"为 oracle 进行评估 |
| [`scripts/evaluate_prompt_routing.py`](scripts/evaluate_prompt_routing.py) | 早期的 per-input 路由评估 |
| [`scripts/train_candidate_conditioned_ranker_v7.py`](scripts/train_candidate_conditioned_ranker_v7.py) | 后期成为主线的"候选条件 ranker"训练 |
| [`scripts/evaluate_deployment_measurement_v7.py`](scripts/evaluate_deployment_measurement_v7.py) | 同时测量质量和实际延迟 |
| [`scripts/build_lockbox_manifests_v9.py`](scripts/build_lockbox_manifests_v9.py) | 生成从未见过的验证切分（lockbox） |

完整的术语（数据集别名、子层掩码、分数模式、`_v7` 后缀的含义等）请参见
📖 [`GLOSSARY.md`](GLOSSARY.md)。

## 文件夹地图

```
.
├── src/attnres_routing/   模型 / 路由 / 数据 / 分析 / 训练 / manifest
├── scripts/               训练 / 评估 / 汇总 / 流水线入口
├── configs/               按轮次组织的实验配置（v5 → v9）
├── reports/               最终报告（韩文 / 英文）
├── GLOSSARY.md            内部术语解码
└── requirements.txt
```

体积较大的产物（数据、结果、日志、外部依赖）没有放进这份归档。

## 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=...   # 仅在需要时
```

## 状态

🧊 **休眠中** —— 一个窄范围内可复现的质量优势留存了下来，但实际速度上的优势没能达到。

## 许可证

按 [CC BY-NC 4.0](./LICENSE) 发布。
