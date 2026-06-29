# 商品一致性评估准确率提升方案 — 多模型 Ensemble & 商品区域裁切

> **当前基线**: Qwen 3.6 VL Plus 单模型 / 全图输入 → Accuracy 76.04%, F1 72.29%
> **目标**: 在不大幅增加成本的前提下突破 80%+ 准确率

---

## 方案一：多模型 Ensemble（跨模型投票）

### 1.1 核心思路

不同 VLM 对同一张海报的错误分布不重叠。当前 Qwen 3.6 VL Plus 的 28 条 FP 集中在「外观与包装不一致」（20条）和「数量改变」（8条），说明该模型在这两个子类型上阈值过严。换一个判定风格不同的模型，大概率不会在同样的 case 上犯错。

**多数投票（Majority Voting）** 是最简单有效的 ensemble 策略：3 个模型分别独立判定，取 ≥2 个模型同意的结果。

### 1.2 模型选择建议

| 模型 | 特点 | 角色 |
|:--|:--|:--|
| **Qwen 3.6 VL Plus** | 当前主力，偏严（FP 高） | 严格审核员 |
| **GPT-4o** | 视觉理解能力强，判定相对均衡 | 均衡裁判 |
| **Claude Sonnet 4.6** / **Gemini 2.5 Pro** | 推理能力强，对细节敏感 | 第三视角 |

> 选模型的原则：**差异化优先于性能**。两个性能相似但犯错模式不同的模型，比两个都很强但犯错高度重叠的模型更有 ensemble 价值。

### 1.3 投票策略

#### 策略 A：简单多数投票（推荐先试）

```
3 个模型独立判定 → 取 ≥2 个模型同意的结果
```

- 优点：实现最简单，对 FP 压制效果最好
- 适合：你当前 FP（28条）>> FN（12条）的不对称分布

#### 策略 B：加权投票

```
每个模型有权重 w_i，按历史准确率分配
final_score = sum(w_i * vote_i) / sum(w_i)
final_score > 0.5 → is_zero = true
```

- 适合：当模型之间准确率差异大时，避免弱模型拖累强模型

#### 策略 C：分维度路由

```
if VLM判定子类型 == "A_外观与包装不一致":
    → 走 3 模型投票（当前 FP 重灾区）
elif VLM判定子类型 == "B_数量改变":
    → 走 3 模型投票
else:
    → 单模型即可（当前准确率已足够）
```

- 适合：精细化控制成本，只在高误判维度上投入多模型

### 1.4 实现方案

#### 步骤 1：扩展 VLM 调用支持多模型

在现有 `vlm_call()` 基础上，添加对不同模型 API 的适配。当前代码已支持 `model` 参数，只需要处理不同 API 的认证和格式差异。

```python
# 模型配置
ENSEMBLE_MODELS = [
    {
        "name": "qwen3.6-plus",
        "base_url": "https://aiboost.zacz.cn/api/v1",
        "key": "...",
        "weight": 1.0,  # 可后续根据历史表现调整
    },
    {
        "name": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "key": "...",
        "weight": 1.0,
    },
    {
        "name": "claude-sonnet-4-6-20250514",
        "base_url": "https://api.anthropic.com/v1",
        "key": "...",
        "weight": 1.0,
    },
]
```

#### 步骤 2：并行调用 + 投票

```python
async def ensemble_evaluate(
    poster_url: str,
    product_url: str,
    logo_url: str | None = None,
    models: list[dict] = ENSEMBLE_MODELS,
    vote_threshold: int = 2,  # ≥2 票判定为 0 分
) -> dict:
    """多模型并行评估 + 多数投票"""

    # 并行调用所有模型
    tasks = [
        evaluate_product_consistency(
            poster_url=poster_url,
            product_url=product_url,
            logo_url=logo_url,
            vlm_model=m["name"],
        )
        for m in models
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 收集有效投票
    votes = []
    model_details = []
    for model_cfg, result in zip(models, results):
        if isinstance(result, Exception):
            continue
        if result.get("is_zero") is not None:
            votes.append({
                "model": model_cfg["name"],
                "is_zero": result["is_zero"],
                "weight": model_cfg["weight"],
                "reason": result.get("reason", ""),
                "zero_sub_type": result.get("zero_sub_type"),
            })
            model_details.append(result)

    # 多数投票
    zero_votes = sum(1 for v in votes if v["is_zero"])
    final_is_zero = zero_votes >= vote_threshold

    # 加权投票（备选）
    # weighted_score = sum(v["weight"] * int(v["is_zero"]) for v in votes)
    # total_weight = sum(v["weight"] for v in votes)
    # final_is_zero = (weighted_score / total_weight) > 0.5

    return {
        "is_zero": final_is_zero,
        "vote_count": f"{zero_votes}/{len(votes)}",
        "votes": votes,
        "reason": f"Ensemble({len(votes)}模型): {zero_votes}票判0分, "
                  f"{'≥' if final_is_zero else '<'}{vote_threshold}票阈值 → "
                  f"{'0分' if final_is_zero else '非0分'}",
    }
```

#### 步骤 3：集成到批量评估

将 `batch_evaluate_product_consistency()` 中的 `evaluate_product_consistency()` 调用替换为 `ensemble_evaluate()`。注意降低 concurrency（从 30 降到 10），因为每条样本会同时调 3 个 API。

### 1.5 成本与性能分析

| 项目 | 单模型（当前） | 3模型 Ensemble |
|:--|:--|:--|
| API 调用数 | N | 3N |
| 单条耗时 | ~3-5s | ~5-8s（并行，取最慢） |
| 总成本 | 1x | ~3x |
| 预期 Accuracy | 76% | 82-85%+ |

### 1.6 分阶段落地建议

1. **第一步 — 离线 A/B 测试**：用现有 192 条标注数据，对 2-3 个候选模型分别跑一遍，观察各模型的 FP/FN 分布是否互补
2. **第二步 — 选择互补性最强的 2 个模型**，和 Qwen 组成 3 模型 ensemble
3. **第三步 — 在全量数据上验证投票效果**，调整 vote_threshold 和权重
4. **第四步 — 上线后按需优化**：如果成本敏感，可以用「单模型初筛 + 低置信走 ensemble」的分级策略

---

## 方案二：商品区域裁切（Grounding 预处理）

### 2.1 核心思路

当前流程是把**完整海报**和**参考图**一起送入 VLM 比较。但海报中包含大量非商品元素（文字、价格标签、装饰背景、促销图标等），这些元素会对 VLM 的判定产生干扰。

从 FP 分析来看，28 条误判中有 20 条是「外观与包装不一致」——VLM 很可能把海报中的设计元素误认为商品变化。

**解决思路**：在调 VLM 之前，先检测并裁切出海报中的商品主体区域，让 VLM 只比较「裁切后的商品区域」和「参考图」，消除背景干扰。

```
当前流程:  参考图 + 完整海报 → VLM → 判定
改进流程:  完整海报 → 检测/裁切商品 → 裁切区域 + 参考图 → VLM → 判定
```

### 2.2 技术选型

#### 方案 A：Grounding DINO + SAM（推荐）

| 组件 | 作用 | 说明 |
|:--|:--|:--|
| **Grounding DINO** | 开放词汇目标检测 | 输入文本 prompt（如 "product, bottle, box, cosmetics"）+ 图片 → 输出商品 bounding box |
| **SAM (Segment Anything)** | 精细分割 | 将 bounding box 精细化为像素级 mask，可选 |

- 开源，可本地部署，无 API 费用
- 检测精度高，支持零样本（不需要针对你的商品类型做训练）

#### 方案 B：VLM 自带的 Grounding 能力

部分 VLM（如 Qwen-VL 系列、GPT-4o）支持在 prompt 中要求模型返回商品的 bounding box 坐标，然后代码侧做裁切。

- 优点：不需要额外部署模型
- 缺点：增加一次 VLM 调用（但可以合并到第一次调用中）

#### 方案 C：YOLO 系列检测

如果商品类型固定，可以用 YOLOv8 训练一个轻量检测器。

- 优点：推理速度极快（<50ms）
- 缺点：需要标注训练数据，泛化性不如 Grounding DINO

**推荐方案 A（Grounding DINO）**，原因：零样本即可工作，对电商场景的各类商品（瓶装、盒装、袋装、散装等）泛化性好。

### 2.3 实现方案

#### 步骤 1：环境准备

```bash
# 安装 GroundingDINO
pip install groundingdino-py

# 或者使用 Hugging Face transformers（更简单）
pip install transformers torch torchvision

# 下载模型权重（~700MB）
# Grounding DINO: IDEA-Research/grounding-dino-base
# SAM（可选）: facebook/sam-vit-base
```

#### 步骤 2：商品检测模块

```python
"""
product_grounding.py — 从海报中检测并裁切商品主体区域
"""
import io
import base64
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch


class ProductGrounder:
    """检测海报中的商品主体并裁切"""

    def __init__(self, model_id="IDEA-Research/grounding-dino-base", device="cpu"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.device = device

    def detect(
        self,
        image: Image.Image,
        text_prompt: str = "product . bottle . box . package . cosmetics . item . goods",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> list[dict]:
        """
        检测图片中的商品区域。
        返回: [{"box": [x1, y1, x2, y2], "score": float, "label": str}, ...]
        """
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # (H, W)
        )[0]

        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            detections.append({
                "box": box.tolist(),   # [x1, y1, x2, y2] 像素坐标
                "score": float(score),
                "label": label,
            })

        # 按面积从大到小排序（主商品通常最大）
        detections.sort(key=lambda d: (d["box"][2]-d["box"][0]) * (d["box"][3]-d["box"][1]), reverse=True)
        return detections

    def crop_product(
        self,
        image: Image.Image,
        detections: list[dict],
        padding_ratio: float = 0.05,
        merge_all: bool = True,
    ) -> Image.Image | None:
        """
        根据检测结果裁切商品区域。
        - merge_all=True: 将所有检测到的商品合并为一个大 bbox 后裁切（保留所有商品）
        - merge_all=False: 只裁切最大的商品
        - padding_ratio: 在 bbox 外扩展一定比例的 padding，避免裁太紧
        """
        if not detections:
            return None

        w, h = image.size

        if merge_all:
            x1 = min(d["box"][0] for d in detections)
            y1 = min(d["box"][1] for d in detections)
            x2 = max(d["box"][2] for d in detections)
            y2 = max(d["box"][3] for d in detections)
        else:
            x1, y1, x2, y2 = detections[0]["box"]

        # 加 padding
        pad_x = (x2 - x1) * padding_ratio
        pad_y = (y2 - y1) * padding_ratio
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        return image.crop((x1, y1, x2, y2))


def data_url_to_image(data_url: str) -> Image.Image:
    """base64 data URL → PIL Image"""
    header, b64_data = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(img_bytes))


def image_to_data_url(image: Image.Image, fmt: str = "jpeg") -> str:
    """PIL Image → base64 data URL"""
    buf = io.BytesIO()
    image.save(buf, format=fmt.upper(), quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt};base64,{b64}"
```

#### 步骤 3：集成到评估流程

在 `evaluate_product_consistency()` 调用 VLM 之前，增加裁切预处理：

```python
# 在 product_consistency_evaluate.py 中集成

from product_grounding import ProductGrounder, data_url_to_image, image_to_data_url

# 全局初始化（只加载一次）
grounder = ProductGrounder(device="cuda" if torch.cuda.is_available() else "cpu")


async def evaluate_product_consistency(
    poster_url: str,
    product_url: str,
    logo_url: str | None = None,
    vlm_model: str = VLM_MODEL,
    use_grounding: bool = True,   # 新增开关
) -> dict:
    result = { ... }  # 同原有初始化
    t0 = time.perf_counter()

    try:
        actual_poster_url = poster_url

        # ── 新增：商品区域裁切 ──
        if use_grounding:
            poster_img = data_url_to_image(poster_url)
            detections = grounder.detect(poster_img)

            if detections:
                cropped = grounder.crop_product(poster_img, detections)
                if cropped:
                    actual_poster_url = image_to_data_url(cropped)
                    result["grounding_detections"] = len(detections)
                    result["grounding_applied"] = True
            else:
                # 检测不到商品 → fallback 用原图
                result["grounding_applied"] = False

        # ── 原有 VLM 调用逻辑不变 ──
        has_logo = bool(logo_url and logo_url.startswith("data:image/"))
        if has_logo:
            image_urls = [product_url, logo_url, actual_poster_url]
        else:
            image_urls = [product_url, actual_poster_url]

        prompt = build_prompt(has_logo)
        check, usage = await vlm_call(image_urls, prompt, vlm_model)
        # ... 后续解析逻辑不变 ...
    except Exception as e:
        ...
```

#### 步骤 4：Prompt 微调（适配裁切后的输入）

裁切后送入 VLM 的是纯商品区域而非完整海报，需要调整 `IMAGE_DESC` 让 VLM 知道输入图已经是裁切过的：

```python
IMAGE_DESC_2_CROPPED = """- 图1「主体图」：原始商品参考图（基准）—— 以此为标准判断商品是否一致
- 图2「商品裁切图」：从 AI 生成海报中裁切出的商品主体区域（检查对象）
  注意：此图已经是从海报中提取的商品区域，背景和文字已去除，请直接对比商品本体"""
```

### 2.4 关键设计决策

#### Q1：检测不到商品怎么办？

```
检测结果为空 → fallback 使用原始完整海报（降级到当前流程）
```

这是安全策略，保证不会因为检测模块失败而漏评。

#### Q2：检测到多个商品怎么办？

```
策略 1（推荐）: merge_all=True，合并所有检测框为一个大区域
  → 保留所有商品，适合「数量改变」维度的判定

策略 2: 只取最大的商品区域
  → 聚焦主商品，减少干扰，但可能丢失数量信息
```

建议使用策略 1，因为你的评估维度包含「B. 数量改变」，需要看到所有商品。

#### Q3：裁切后图片太小/分辨率不够？

```
设置最小尺寸阈值：裁切后的图片宽或高 < 100px → fallback 用原图
```

#### Q4：Grounding DINO 部署在哪？

| 方式 | 适合场景 |
|:--|:--|
| 本地 CPU | 开发调试，<50 张/批 |
| 本地 GPU | 批量评估，推理 ~100ms/张 |
| 云端 API | 大规模部署，如 Hugging Face Inference API |

你当前是批量离线评估（200 条），本地 GPU 完全够用。

### 2.5 效果预估

| 指标 | 当前（全图） | 预期（裁切后） | 分析 |
|:--|:--|:--|:--|
| FP (误判) | 28 条 | 预计降到 10-15 条 | 去掉背景干扰后，VLM 不会再把设计元素误认为商品差异 |
| FN (漏判) | 12 条 | 预计持平或小幅改善 | 裁切后商品更聚焦，有助于发现细节差异 |
| Accuracy | 76.04% | 预计 82-86% | 主要靠 FP 大幅下降 |

### 2.6 实施注意事项

1. **先在 28 条 FP 上验证**：把这 28 条 FP case 的海报跑一遍 Grounding DINO，检查裁切结果是否合理。如果裁切准确，FP 大概率能修正
2. **保留原图作为 fallback**：永远不要完全依赖裁切结果
3. **记录裁切信息**：在结果 JSON 中保留 `grounding_applied`、`detections` 信息，方便后续分析
4. **Prompt 中的「海报包装化」维度**：裁切后 VLM 看不到海报全貌，无法判断 C 类问题。可以走双通道：裁切图判 A/B 类，全图判 C 类

---

## 方案一 + 方案二 组合

两个方案解决的是不同问题，可以组合使用：

```
                    ┌──────────────────┐
                    │   输入: 海报+参考图  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Grounding DINO   │
                    │  检测 & 裁切商品    │  ← 方案二：减少背景干扰
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌───────────┐ ┌───────────┐ ┌───────────┐
        │ Qwen VL   │ │ GPT-4o    │ │ Claude    │  ← 方案一：多模型互补
        │ Plus      │ │           │ │ Sonnet    │
        └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼─────────┐
                    │   多数投票 ≥2/3    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   最终判定结果      │
                    └──────────────────┘
```

**预期组合效果**:

| 指标 | 当前 | 单用方案一 | 单用方案二 | 组合 |
|:--|:--|:--|:--|:--|
| FP | 28 | ~18 | ~12 | ~6-8 |
| FN | 12 | ~8 | ~11 | ~6-7 |
| Accuracy | 76% | ~82% | ~84% | ~88-90% |

---

## 落地优先级与步骤

| 阶段 | 动作 | 预计耗时 |
|:--|:--|:--|
| **Phase 1** | 在 28 条 FP 上测试 Grounding DINO 裁切效果 | 1 天 |
| **Phase 2** | 对 192 条标注集跑 2-3 个候选 VLM 的 A/B 测试 | 1-2 天 |
| **Phase 3** | 实现方案二（裁切预处理）并全量评估 | 2 天 |
| **Phase 4** | 实现方案一（Ensemble 投票）并全量评估 | 1 天 |
| **Phase 5** | 组合两个方案，调优参数 | 1 天 |

---

## 参考文献

1. **Grounding DINO** — Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection", ECCV 2024, arXiv:2303.05499
2. **SAM (Segment Anything)** — Kirillov et al., "Segment Anything", ICCV 2023, arXiv:2304.02643
3. **LLM-Blender (Ensemble)** — Jiang et al., "LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion", ACL 2023, arXiv:2306.02561
4. **DINOv2** — Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", TMLR 2024, arXiv:2304.07193
5. **CLIP** — Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021, arXiv:2103.00020
