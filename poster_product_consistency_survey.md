# 营销海报商品一致性评估方案调研

## 一、问题定义

评估 AI 生成的营销海报中，商品主体、赠品、Logo 是否与原始输入图保持一致。

核心维度：

| 维度 | 说明 |
|------|------|
| 主体商品 | 外观、形状、颜色、包装、完整性 |
| Logo | 形状、颜色、位置、文字拼写 |
| 赠品 | 是否存在、品类是否正确 |
| 数量 | 商品数量是否增减 |
| 整体 | 是否"包装化"、是否合理构图 |

---

## 二、三类主流评估方案

### 方案一：CLIP / DINOv2 特征相似度

**原理**

使用预训练视觉模型提取输入图和生成图的特征向量，计算 cosine similarity 作为一致性分数。

| 模型 | 擅长 | 局限 |
|------|------|------|
| CLIP (OpenAI) | 语义级一致性（"是同一类商品"） | 对细节（纹理、Logo文字）不敏感 |
| DINOv2 (Meta) | 纹理/形状/结构等视觉细节 | 不理解语义，对布局变化敏感 |

**典型指标**

- **CLIP-I**：CLIP image encoder 提取两图特征后计算 cosine similarity
- **DINO score**：DINOv2 CLS token 的 cosine similarity

**适用维度**

- 主体商品：DINOv2（细节还原度要求高）
- 赠品：CLIP（语义级存在即可）
- Logo：DINOv2 + OCR 辅助验证

**优势**

- 开源成熟，代码现成
- 计算快，可批量运行
- Subject-driven generation 领域的事实标准（IP-Adapter、DreamBooth 均使用）

**局限**

- 全局特征，无法区分"哪个维度不一致"
- 阈值需要人工标定
- 对遮挡、裁切等场景需要先做检测/分割

**参考资源**

- CLIP 论文：https://arxiv.org/abs/2103.00020
- CLIP 代码：https://github.com/openai/CLIP
- DINOv2 论文：https://arxiv.org/abs/2304.07193
- DINOv2 代码：https://github.com/facebookresearch/dinov2
- DreamBooth（DINO score 出处）：https://arxiv.org/abs/2208.12242
- IP-Adapter（CLIP-I + DINO 评估）：https://arxiv.org/abs/2308.06721

---

### 方案二：检测 + 分区域特征对比 Pipeline

**原理**

先用目标检测模型定位生成图中的商品/Logo/赠品区域，再对每个区域与对应输入图做特征对比。

**Pipeline 架构**

```
输入图（主体图、赠品图、Logo）
        ↓
检测模块（GroundingDINO / YOLOv8）
  → 定位生成图中的商品区域、Logo区域、赠品区域
        ↓
分区域特征提取（DINOv2 / CLIP）
  → 逐区域与输入图计算相似度
        ↓
辅助校验
  → Logo: OCR 文字验证（PaddleOCR / EasyOCR）
  → 数量: 检测框计数
  → 颜色: 直方图对比
        ↓
规则引擎（阈值 + 多维度加权）
  → 输出结构化评分
```

**适用维度**

全维度覆盖，每个维度独立打分。

**优势**

- 可解释性强：每个维度有独立分数
- 精细可控：可以针对不同维度设置不同阈值
- 业界验证：电商头部公司（阿里妈妈、京东等）内部系统的常见架构

**局限**

- 工程复杂度高，需要组合多个模型
- 检测模型准确率直接影响下游
- 需要维护阈值和规则

**参考资源**

- GroundingDINO（开放词汇检测）：https://github.com/IDEA-Research/GroundingDINO
- YOLOv8：https://github.com/ultralytics/ultralytics
- PaddleOCR：https://github.com/PaddlePaddle/PaddleOCR
- AnyDoor（阿里，电商场景评估）：https://arxiv.org/abs/2307.09481

---

### 方案三：多模态大模型 (VLM) 评判

**原理**

将输入图和生成图同时传给 VLM（GPT-4o / Qwen-VL / Claude），通过 prompt 让模型从多个维度打分，输出结构化 JSON。

**当前项目已采用此方案**，使用 qwen3.6-plus 模型做二分类评估。

**适用维度**

全维度覆盖，通过 prompt 灵活定义评估规则。

**优势**

- 无需训练，无需标注数据
- 理解语义，可以判断"包装化"等高级概念
- 输出可解释（自然语言理由）
- 灵活适配业务规则变更（改 prompt 即可）
- 多维度评估一次调用完成

**局限**

- 成本高（大模型推理费用）
- 速度慢（单次 ~3-10s）
- 一致性不稳定（同一输入多次调用可能结果不同）
- 对像素级细节（Logo 微小变形）判断不够精准
- 受幻觉影响，可能"看到"不存在的问题

**参考资源**

- T2I-CompBench（VLM 评估生成图）：https://arxiv.org/abs/2310.01852
- ImageReward：https://arxiv.org/abs/2304.05977
- SEED-Bench（多模态评估基准）：https://github.com/AILab-CVC/SEED-Bench

---

## 三、是否有专用模型？

**目前没有"营销海报商品一致性"的专用模型。** 原因：

1. 任务过于垂直，公开数据集缺乏
2. 业界头部公司的方案均为自建 pipeline，不对外开源
3. 最接近的学术方向是 subject-driven generation 的评估，使用 CLIP-I + DINO score 组合

最相关的方向：

| 方向 | 评估方式 | 参考 |
|------|---------|------|
| IP-Adapter | CLIP-I + DINO | https://arxiv.org/abs/2308.06721 |
| AnyDoor (阿里) | DINO + CLIP-I + 人工标注 | https://arxiv.org/abs/2307.09481 |
| ImageReward | 训练过的人类偏好打分模型 | https://arxiv.org/abs/2304.05977 |
| HPS v2 | 人类偏好打分 | https://arxiv.org/abs/2306.09341 |

---

## 四、推荐方案

结合当前项目现状（已有 VLM 方案），推荐**方案二 + 方案三组合**：

| 维度 | 方法 | 原因 |
|------|------|------|
| 主体商品 | DINOv2 similarity | 细节还原度要求高，VLM 对像素级变形不敏感 |
| Logo | DINOv2 similarity + OCR | 需要像素级忠实 + 文字准确性 |
| 赠品 | CLIP similarity | 语义级"存在且品类正确"即可 |
| 数量 | 检测框计数 | 精确计数比 VLM 可靠 |
| 整体合理性 | VLM 打分（当前方案） | 需要语义理解，如"包装化"判断 |

**分阶段落地建议：**

1. **短期**（当前）：继续使用 VLM 方案，优化 prompt，建立标注基准集
2. **中期**：引入 DINOv2 / CLIP 做定量指标，与 VLM 结果交叉验证，提升稳定性
3. **长期**：在标注数据积累后，训练专用的 learned metric 模型
