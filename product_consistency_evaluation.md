# 电商海报商品一致性评估模块 - 技术方案

## 一、模块概述

### 1.1 目标

基于 VLM（Vision Language Model）对电商海报中的**商品主体**进行一致性评估，判断生成海报中的商品是否与输入素材保持一致。采用**二分类**判定：0分（不可接受）/ 非0分（可接受）。

### 1.2 评估范围

| 评估对象 | 说明 | 示例 |
|---------|------|------|
| 主体商品 | 海报中核心商品与输入商品图的一致性 | 洗面奶瓶身、手机正面 |

### 1.3 输入/输出总览

```
输入:
  ├── poster_image        # 生成的海报图片
  └── product_image       # 主体商品参考图

输出:
  ├── result.json         # 结构化评估结果（含统计）
  └── result.csv          # 表格化评估结果
```

---

## 二、评估维度与判定规则（二分类）

### 2.1 分类标准

采用**二分类**判定，仅区分「0分」与「非0分」：

| 分类 | 含义 | 说明 |
|------|------|------|
| **0分** | 不可接受 | 商品一致性存在严重问题，触发任一0分条件 |
| **非0分** | 可接受 | 商品与参考图一致，无严重问题 |

### 2.2 0分判定条件 — 商品不一致 (PRODUCT_MISMATCH)

仅评估一个维度，包含 6 个子类型，命中任一即为0分：

| 子类型编码 | 名称 | 判定标准 |
|------------|------|----------|
| `A_主体差异` | 商品主体根本性差异 | 商品品类、外观、颜色发生根本性变化，无法识别为同一商品 |
| `B_形状改变` | 商品外观形状改变 | 商品瓶身/盒身/外形发生可感知的形变（拉伸、压缩、扭曲、比例失真） |
| `C_数量位置改变` | 商品数量与位置改变 | 商品数量增减，或多个商品的相对位置/排列方式与参考图不一致 |
| `D_文案遮挡` | 文案遮挡商品主体 | 海报文字/标签遮挡商品主体的大部分区域，导致无法清晰辨识商品完整外观 |
| `E_包装不一致` | 商品包装不一致 | 包装设计（花纹、配色、标签布局、文字内容）与参考图有可感知的差异 |
| `F_Logo不一致` | Logo不一致 | 商品上的Logo形状/颜色/位置/文字与参考图不一致，或Logo变形/缺失 |

### 2.3 非0分（可接受的情况）

以下情况不应判为0分：
- 商品完全一致，仅存在轻微光照/色温差异（非颜色根本变化）
- 商品角度/视角略有不同，但外观形状、包装、Logo均保持一致
- 商品被少量文字遮挡（遮挡面积小，不影响商品整体辨识）
- 背景/环境变化但商品本身未发生改变

---

## 三、VLM Prompt 设计

### 3.1 评估 Prompt

```text
你是一位专业的电商海报质量审核员，专注于评估海报中商品主体与原始参考图的一致性。

━━━━━━━━━━━━━━━━━━━━━━━━
📷 输入的两张图
━━━━━━━━━━━━━━━━━━━━━━━━
- 图1「主体图」：原始商品参考图（基准）
- 图2「生成图」：AI 生成的营销海报（检查对象）

━━━━━━━━━━━━━━━━━━━━━━━━
🎯 评估任务
━━━━━━━━━━━━━━━━━━━━━━━━
判断图2（生成海报）中的商品主体是否与图1（参考图）保持一致。
输出二分类结果：is_zero = true（0分，不可接受）或 is_zero = false（非0分，可接受）。

━━━━━━━━━━━━━━━━━━━━━━━━
🚨 0分判定条件（命中任一项 → is_zero = true）
━━━━━━━━━━━━━━━━━━━━━━━━
1. 商品不一致 (PRODUCT_MISMATCH)
2. 海报化 (POSTERIZED)
3. 无包装 (NO_PACKAGING)
4. 商品缺失 (PRODUCT_MISSING)

━━━━━━━━━━━━━━━━━━━━━━━━
✅ 非0分（可接受的情况）
━━━━━━━━━━━━━━━━━━━━━━━━
- 商品可辨识为同一商品，但存在轻微色差、光照差异
- 商品角度/视角略有不同，但仍能确认是同一商品
- 商品细节略有模糊但整体一致
- 商品被部分遮挡但可见部分与参考图一致

━━━━━━━━━━━━━━━━━━━━━━━━
输出格式（严格 JSON，不要 markdown 包裹）
━━━━━━━━━━━━━━━━━━━━━━━━
{
  "product_location": "商品在海报中的位置描述",
  "is_zero": true或false,
  "zero_reason": "PRODUCT_MISMATCH|POSTERIZED|NO_PACKAGING|PRODUCT_MISSING|null",
  "reason": "判定理由"
}
```

---

## 四、输出格式定义

### 4.1 JSON 输出格式

单条评估结果：

```json
{
  "row": 2,
  "is_zero": true,
  "zero_reason": "POSTERIZED",
  "product_location": "海报中央偏左区域",
  "reason": "商品被处理为卡通插画风格，完全丧失了真实商品的质感和形态",
  "model": "qwen3.6-vl-plus",
  "elapsed_seconds": 3.5,
  "label_is_zero": true,
  "label_zero_reason": "商品海报化"
}
```

批量评估结果（完整 JSON 文件）：

```json
{
  "meta": {
    "total_count": 294,
    "valid_count": 292,
    "zero_count": 68,
    "non_zero_count": 224,
    "zero_rate": 0.2329,
    "accuracy_vs_label": 0.9178,
    "model": "qwen3.6-vl-plus",
    "dimension": "product_consistency",
    "classification": "binary (0分/非0分)",
    "total_seconds": 876.5,
    "avg_seconds": 3.0
  },
  "results": [
    { "row": 2, "is_zero": false, "..." : "..." },
    { "row": 3, "is_zero": true, "..." : "..." }
  ]
}
```

### 4.2 CSV 输出格式

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `row` | int | 行号 |
| `is_zero` | bool | 是否判定为0分 |
| `zero_reason` | string | 0分原因编码 |
| `product_location` | string | 商品在海报中的位置 |
| `reason` | string | 判定理由 |
| `label_is_zero` | bool | 人工标注（用于对比） |
| `label_zero_reason` | string | 人工标注的原因 |
| `match` | string | 与人工标注是否一致（✓/✗） |
| `elapsed_seconds` | float | 单条耗时 |

CSV 示例：

```csv
row,is_zero,zero_reason,product_location,reason,label_is_zero,label_zero_reason,match,elapsed_seconds
2,False,,海报中央,商品与参考图一致,False,,✓,3.2
3,True,POSTERIZED,海报左侧,商品被卡通化处理,True,商品海报化,✓,4.1
4,True,NO_PACKAGING,海报中部,商品丢失外包装,True,无包装,✓,3.8
```

---

## 五、模块架构

### 5.1 整体流程

```
┌─────────────────────────────────────────────────────┐
│  1. 数据加载层                                       │
│  dataset_loader: 从 xlsx 提取嵌入图片(base64)        │
│  ├── 主体图 (必须)                                    │
│  └── 生成图 (必须)                                    │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  2. VLM 推理层                                       │
│  vlm_call:                                           │
│  ├── 传入 2 张图（主体图 + 生成图）                    │
│  ├── 商品一致性二分类 Prompt                         │
│  ├── VLM API 调用 (指数退避重试)                      │
│  └── JSON 解析                                        │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  3. 结果统计层                                       │
│  batch_evaluate:                                     │
│  ├── 二分类统计（0分数/非0分数/0分率）                │
│  ├── 0分原因分布统计                                  │
│  ├── 与人工标注对比（一致率）                         │
│  └── 输出 JSON + CSV                                  │
└─────────────────────────────────────────────────────┘
```

### 5.2 核心数据结构

```python
from enum import Enum


class ZeroReason(str, Enum):
    PRODUCT_MISMATCH = "PRODUCT_MISMATCH"
    POSTERIZED = "POSTERIZED"
    NO_PACKAGING = "NO_PACKAGING"
    PRODUCT_MISSING = "PRODUCT_MISSING"
```

### 5.3 关键函数说明

```python
async def vlm_call(image_urls, prompt, model, max_retries=3):
    """底层 VLM API 调用，支持指数退避重试"""
    ...

async def evaluate_product_consistency(poster_url, product_url, vlm_model):
    """单条评估：传入主体图+生成图 → 输出二分类结果"""
    ...

async def batch_evaluate_product_consistency(xlsx_path, vlm_model, limit, concurrency):
    """批量评估：并发控制 + 统计汇总 + 输出 JSON/CSV"""
    ...
```

---

## 六、输入数据格式

输入数据为 xlsx 文件，图片嵌入在单元格中：

| 列 | 字段名 | 说明 | 必填 |
|----|--------|------|------|
| A | 主体图 | 商品参考图（嵌入图片） | ✅ |
| B | logo | 品牌Logo图（嵌入图片） | ❌ |
| C | 生成图 | AI生成的海报（嵌入图片） | ✅ |
| D | 是否0分 | 人工标注（“是”/“否”） | - |
| E | 0分原因 | 人工标注的原因 | - |

数据加载通过 `dataset_loader.py` 实现，自动将嵌入图片转为 base64 data URL。

---

## 七、VLM 选型与配置

### 7.1 当前使用模型

| 模型 | 说明 |
|------|------|
| **qwen3.6-plus** | 当前主用模型，中文理解强，商品识别准确，性价比高 |

### 7.2 模型配置参数

```python
VLM_MODEL = "qwen3.6-plus"
BASE = "https://aiboost.zacz.cn/api/v1"

# API 调用参数
temperature = 0.0          # 低温度保证稳定性
timeout = 180.0            # 超时 180s
max_retries = 3            # 最大重试次数（指数退避 2s/4s/8s）
```

---

## 八、异常处理与兜底策略

### 8.1 VLM 输出异常

| 异常类型 | 处理方式 |
|---------|---------|
| JSON 解析失败 | 重试最多3次，仍失败则标记为 `EVAL_FAILED`，不计入统计 |
| 字段缺失 | 填充默认值（score=null, is_zero=null），标记 `PARTIAL_EVAL` |
| API 超时 | 指数退避重试(2s/4s/8s)，超限后降级到备用模型 |
| API 限流 | 自动降低并发数，等待后重试 |

### 8.2 输入异常

| 异常类型 | 处理方式 |
|---------|---------|
| 图片文件不存在 | 跳过该条，记录到 error log |
| 图片无法解码 | 跳过该条，记录到 error log |
| 图片尺寸过大 (>20MB) | 自动压缩至长边 2048px |

---

## 九、评估质量保障

### 9.1 校准方法

1. **构建标注集**: 人工标注 100+ 海报样本（覆盖所有 0 分原因类型），作为 ground truth
2. **计算一致率**: VLM 0分判定与人工标注的一致率（目标 > 90%）
3. **混淆矩阵分析**: 分析各 0 分原因的 precision / recall

```
                    VLM判定0分   VLM判定非0分
人工判定0分           TP            FN        ← 漏检率=FN/(TP+FN)
人工判定非0分         FP            TN        ← 误判率=FP/(FP+TN)
```

### 9.2 持续优化

- **BadCase 收集**: 每轮评估后抽检不一致样本，分析 prompt 弱点
- **Prompt 迭代**: 针对高频错误类型补充 few-shot 示例
- **阈值调优**: 对可量化指标（如 CLIP 相似度）校准 0 分阈值

### 9.3 辅助校验（可选增强）

在 VLM 评估之外，可引入量化指标作为辅助交叉验证：

| 辅助方法 | 用途 | 阈值参考 |
|---------|------|---------|
| CLIP 余弦相似度 | 商品区域 vs 参考图 | < 0.75 触发复核 |
| DINOv2 特征距离 | 商品细节一致性 | 补充 VLM 无法量化的局部差异 |
| Grounding-DINO 定位 | 商品/Logo 区域裁切 | 提升 VLM 多图对比精度 |

---

## 十、使用方式

### 10.1 命令行调用

```bash
# 运行全量评估
python product_consistency_evaluate.py

# 输出文件
# results_product_consistency_qwen3_6-vl-plus.json
# results_product_consistency_qwen3_6-vl-plus.csv
```

### 10.2 Python SDK 调用

```python
from product_consistency_evaluate import (
    evaluate_product_consistency,
    batch_evaluate_product_consistency,
)
import asyncio

# 单条评估
result = asyncio.run(evaluate_product_consistency(
    poster_url="data:image/jpeg;base64,...",
    product_url="data:image/jpeg;base64,...",
    vlm_model="qwen3.6-plus",
))
print(result["is_zero"])       # True/False
print(result["zero_reason"])   # 'POSTERIZED' | None

# 批量评估
results = asyncio.run(batch_evaluate_product_consistency(
    xlsx_path="data/dataset_without_gift_label.xlsx",
    vlm_model="qwen3.6-plus",
    limit=10,         # 限制条数（测试用）
    concurrency=5,
))
```

---

## 十一、性能预估

| 指标 | qwen3.6-plus |
|------|------------------|
| 单条耗时 | 3-5s |
| 输入图片数 | 2张（主体图 + 生成图） |
| 294条批量(concurrency=5) | ~5min |
| 294条批量(concurrency=10) | ~3min |

---

## 十二、后续扩展

| 方向 | 说明 |
|------|------|
| 多维度扩展 | 在同一框架下增加 Logo 一致性、文字正确性、排版合理性等评估维度 |
| Few-shot 增强 | 针对高频 badcase 在 prompt 中加入典型示例图 |
| 人机协同 | VLM 判定不确定的边界 case 自动路由到人工复核队列 |
| 评估报告 | 自动生成可视化评估报告（混淆矩阵、典型 badcase 展示） |
