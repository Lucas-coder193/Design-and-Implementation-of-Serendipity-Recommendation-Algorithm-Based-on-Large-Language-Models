# 虚构演示数据

本目录中的电影名、简介、历史和五因子评分均为手工编写的虚构示例，用于说明数据格式与排序流程，不含原始用户记录，也不代表模型预测或实验结果。

- `synthetic_candidates.json`：五条虚构历史、二十个候选及其五维评分和排序权重。
- `analyst_predictions.jsonl`：Analyst 格式示例，仅包含相关性和有用性。
- `explorer_predictions.jsonl`：Explorer 格式示例，仅包含新颖性、意外性和多样性。
- `rerank_demo.py`：无需第三方依赖即可运行的 Top-3 排序示例。

从仓库根目录运行：

```bash
python examples/rerank_demo.py
```

Streamlit 页面在本地完整预测文件不存在时会自动读取本目录的两个 JSONL 文件。
