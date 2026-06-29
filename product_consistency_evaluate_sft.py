"""
商品一致性评估 — 基于 SFT 测试集 (sft_200_test_from_train.json)

使用与 product_consistency_evaluate.py 完全相同的 Prompt，
对 SFT 测试集进行评估，并与 ground truth 标签对比计算准确率。

用法：
  python product_consistency_evaluate_sft.py
"""
import asyncio
import json
import logging
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import httpx

# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────
KEY = "RFI5VHdQRG1KUGVoRlRsQWg4c09zM3VjTjN0ckk5bVJyVVJKM1g4MnNSQzo0Q0VrUXVtT21sRXFSdmhvNXR1OXpmMGVkbnZDUVRmbw=="
BASE = "https://aiboost.zacz.cn/api/v1"
VLM_MODEL = "qwen3.7-plus"

TEST_DATASET = "data/sft_200_test_from_train.json"

# ──────────────────────────────────────────────────────────────
# Prompt：与 product_consistency_evaluate.py 完全一致
# ──────────────────────────────────────────────────────────────

PROMPT_HEADER = """你是一位专业的电商海报质量审核员，专注于评估海报中商品主体与原始参考图的一致性。

━━━━━━━━━━━━━━━━━━━━━━━━
📷 输入图片
━━━━━━━━━━━━━━━━━━━━━━━━
{image_desc}

━━━━━━━━━━━━━━━━━━━━━━━━
🎯 评估任务
━━━━━━━━━━━━━━━━━━━━━━━━
判断生成海报中的商品主体是否与参考图保持一致。
输出二分类结果：is_zero = true（0分，不可接受）或 is_zero = false（非0分，可接受）。"""

PROMPT_BODY = """
━━━━━━━━━━━━━━━━━━━━━━━━
⚖️ 判定原则（核心）
━━━━━━━━━━━━━━━━━━━━━━━━
当存在疑虑、拿不准是否符合0分条件时，倾向于判定为0分（宁可严判，不可漏判）。
商品包装盒/包装袋/瓶身等外观载体是商品主体不可分割的一部分。
若海报缺失或根本性替换了参考图中的包装实体，即使露出了内容物，也判定为商品不一致。

━━━━━━━━━━━━━━━━━━━━━━━━
🚨 0分判定条件 — 商品不一致 (PRODUCT_MISMATCH)
━━━━━━━━━━━━━━━━━━━━━━━━
以下任一情况命中即判定为0分（is_zero = true）：

**A. 商品外观与包装不一致**（逐项检查以下维度，任一项不通过即判0分）

   【品类】商品品类发生变化（如参考图是洗面奶，海报中变成了口红）

   【外观/形状】商品外观、形状发生商家用户可感知的变化：
     - 包括拉伸、压缩、扭曲等形变
     - 瓶身/盒身/管身比例失真（变胖、变窄、变矮、变高）
     - 商品部件位置发生改变（如标签移位等）

   【颜色】商品主体颜色发生根本性变化（如白色变成黑色、红色变成蓝色）

   【完整性/轮廓】商品不完整或被改变：
     - 商品被裁切、缺失部分主体
     - 商品轮廓与参考图明显不一致
     - 参考图中完整的包装实体（盒/袋/瓶）在海报中被替换为内容物展示或无包装状态

   【包装】商品包装设计不一致：
     - 包装印花/花纹、配色方案、标签布局与参考图不一致
     - 参考图为带包装盒/袋的商品，海报中包装实体缺失，仅展示内容物
     - 包装样式发生了根本性改变（如从立体盒装变为平面展示、从袋装变为散装）

   【Logo】商品上的品牌Logo不一致：
     - Logo形状、颜色、位置与参考图不同
     - Logo发生变形、模糊到无法辨认、或完全缺失
     - Logo文字拼写与参考图不同

   【遮挡】海报中文字/文案/标签遮挡商品主体大部分区域，导致无法清晰辨识完整外观

**B. 商品数量改变**
   - 参考图中有N个商品，海报中商品数量发生增减（包括主体商品和附属品），如果是零售商品，只要不影响消费者对总价/单价的判断，轻微数量变化（如1个变2个）可以宽容；但如果数量变化过大（如3个变10+个）或套装组合中品类件数发生明显增减，则判定为不一致。
   - 重点关注：数量变化是否会影响消费者对商品总价或单价的判断,如果不涉及单价，则数量变化较大时也可以宽容；但如果涉及单价，数量变化过大则判定为不一致。
   - 注意：海报为美观可能调整商品相对位置和排列方式，单纯的位置/排列变化不作为0分判定依据

**C. 海报包装化**
   - 生成的海报整体看起来像商品包装/标签，而非营销海报
   - 海报失去了营销设计感，变成了产品外包装的平面展开图
   - 海报直接复制/模仿了商品包装的版式布局，缺乏海报应有的创意构图


━━━━━━━━━━━━━━━━━━━━━━━━
✅ 非0分（可接受的情况）
━━━━━━━━━━━━━━━━━━━━━━━━
以下情况不应判为0分：
- 商品完全一致，仅存在轻微光照/色温差异（非颜色根本变化）
- 商品角度/视角略有不同，但外观形状、包装实体、Logo均完整保留且保持一致
- 商品被少量文字遮挡（遮挡面积小，不影响商品整体辨识，不覆盖关键特征）
- 背景/环境变化但商品本身未发生改变
- 商品之间的相对位置/排列方式变化（海报为美观可调整布局）
- 包装上的文字内容差异（如产品名、说明文字等，本次暂不纳入评估）
- 海报具有正常的营销设计构图（即使借鉴了部分包装元素作为装饰）
- 注意：包装实体（盒/袋/瓶身）是商品的组成部分，若海报去掉了包装只展示内容物，仍判为0分

━━━━━━━━━━━━━━━━━━━━━━━━
📋 评估步骤
━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: 在生成海报中定位商品主体的位置
Step 2: 检查商品品类是否与参考图一致
Step 3: 对比商品外观形状（轮廓、比例、是否有形变、部件位置是否正常）
Step 4: 检查商品颜色是否发生根本变化
Step 5: 检查商品是否完整（有无裁切、缺失、包装实体是否被替换为内容物）
Step 6: 对比商品包装设计（包装盒/袋/瓶身的印花、配色、标签布局）
Step 7: 对比商品Logo（形状、颜色、位置、是否变形或缺失）
Step 8: 检查商品是否被文案大面积遮挡
Step 9: 对比商品数量（重点关注影响总价/单价的数量变化）
Step 10: 判断海报是否整体呈现为包装化
Step 11: 给出最终判定（存疑时倾向于判0分）

━━━━━━━━━━━━━━━━━━━━━━━━
输出格式（严格 JSON，不要 markdown 包裹）
━━━━━━━━━━━━━━━━━━━━━━━━
{
  "product_location": "商品在海报中的位置描述（如找不到则填'未找到'）",
  "is_zero": true或false,
  "zero_reason": "商品不一致|null",
  "zero_sub_type": "具体命中的子类型：A_外观与包装不一致|B_数量改变|C_海报包装化|null",
  "reason": "判定理由，简要说明为什么判为0分或非0分"
}

⚠️ 只输出 JSON，不要输出任何其他内容。
⚠️ zero_reason 只在 is_zero=true 时填写"商品不一致"，is_zero=false 时填 null。
⚠️ zero_sub_type 填写最主要的一个子类型编码，is_zero=false 时填 null。"""


IMAGE_DESC_2 = """- 图1「主体图」：原始商品参考图（基准）—— 以此为标准判断商品是否一致
- 图2「生成图」：AI 生成的营销海报（检查对象）—— 只检查这张图中的商品"""

IMAGE_DESC_3 = """- 图1「主体图」：原始商品参考图（基准）—— 以此为标准判断商品是否一致
- 图2「Logo图」：品牌Logo参考图（基准）—— 以此为标准判断Logo是否一致
- 图3「生成图」：AI 生成的营销海报（检查对象）—— 只检查这张图中的商品"""


def build_prompt(has_logo: bool) -> str:
    """根据是否有Logo图动态拼装Prompt"""
    image_desc = IMAGE_DESC_3 if has_logo else IMAGE_DESC_2
    return PROMPT_HEADER.format(image_desc=image_desc) + PROMPT_BODY


# ──────────────────────────────────────────────────────────────
# 解析测试集 ground truth
# ──────────────────────────────────────────────────────────────
def parse_ground_truth(assistant_content: str) -> Dict[str, str]:
    """
    解析 assistant 回复，提取 ground truth 标签。
    例如：
      商品是否一致：是
      品牌Logo是否一致：否
      赠品是否一致：是
    """
    labels = {}
    for line in assistant_content.strip().split("\n"):
        line = line.strip()
        if "商品是否一致" in line:
            labels["product"] = "是" if "是" in line.split("：")[-1] else "否"
        elif "品牌Logo是否一致" in line:
            labels["logo"] = "是" if "是" in line.split("：")[-1] else "否"
        elif re.search(r"赠品\d*是否一致", line):
            key_match = re.search(r"(赠品\d*)", line)
            key = key_match.group(1) if key_match else "赠品"
            labels[key] = "是" if "是" in line.split("：")[-1] else "否"
    return labels


def detect_sample_type(user_content: str) -> Dict[str, bool]:
    """
    根据 user prompt 检测样本类型（是否有Logo、赠品等）。
    """
    info = {
        "has_logo": "品牌Logo" in user_content or "Logo" in user_content,
        "has_gift": "赠品" in user_content,
    }
    # 统计赠品数量
    gift_matches = re.findall(r"赠品(\d+)", user_content)
    if gift_matches:
        info["gift_count"] = max(int(m) for m in gift_matches)
    else:
        info["gift_count"] = 1 if info["has_gift"] else 0
    return info


# ──────────────────────────────────────────────────────────────
# VLM 调用（与 product_consistency_evaluate.py 一致）
# ──────────────────────────────────────────────────────────────
async def vlm_call(
    image_urls: List[str],
    prompt: str,
    model: str = VLM_MODEL,
    max_retries: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    调用 VLM API，传入多张图片 + prompt。
    返回 (parsed_json, usage_info)。
    """
    content_parts = []
    for url in image_urls:
        content_parts.append({"type": "image_url", "image_url": {"url": url}})
    content_parts.append({"type": "text", "text": prompt})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content_parts}],
        "temperature": 0.0,
        "extra": {"businessSceneCode": "retailadvqa"},
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                r = await client.post(
                    f"{BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                r.raise_for_status()
                d = r.json()

            raw = d["choices"][0]["message"]["content"].strip()
            usage = d.get("usage", {})

            # 清理 markdown 代码块
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            try:
                return json.loads(raw), usage
            except json.JSONDecodeError:
                return {"raw": raw, "parse_error": True}, usage

        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
            last_err = e
            wait = 2 ** attempt
            logging.warning(
                f"VLM 重试 {attempt+1}/{max_retries} ({type(e).__name__}): {e}，等待 {wait}s"
            )
            await asyncio.sleep(wait)
        except Exception as e:
            last_err = e
            break

    raise RuntimeError(f"VLM 调用多次失败: {type(last_err).__name__}: {last_err}")


# ──────────────────────────────────────────────────────────────
# 单条评估
# ──────────────────────────────────────────────────────────────
async def evaluate_single(
    product_url: str,
    poster_url: str,
    logo_url: Optional[str] = None,
    vlm_model: str = VLM_MODEL,
) -> Dict[str, Any]:
    """
    商品一致性评估（二分类），使用与 product_consistency_evaluate.py 相同的 Prompt。
    """
    result: Dict[str, Any] = {
        "is_zero": None,
        "zero_reason": None,
        "zero_sub_type": None,
        "product_location": "",
        "reason": "",
        "usage": {},
        "elapsed_seconds": 0.0,
    }
    t0 = time.perf_counter()
    try:
        has_logo = bool(logo_url)
        if has_logo:
            image_urls = [product_url, logo_url, poster_url]
        else:
            image_urls = [product_url, poster_url]
        prompt = build_prompt(has_logo)
        check, usage = await vlm_call(image_urls, prompt, vlm_model)
        result["usage"] = usage

        if check.get("parse_error"):
            result["error"] = "VLM 返回非 JSON"
            result["raw"] = check.get("raw", "")
        else:
            result["is_zero"] = bool(check.get("is_zero"))
            result["zero_reason"] = check.get("zero_reason")
            result["zero_sub_type"] = check.get("zero_sub_type")
            result["product_location"] = check.get("product_location", "")
            result["reason"] = check.get("reason", "")
    except Exception as e:
        logging.error(f"VLM 调用失败: {traceback.format_exc()}")
        result["error"] = f"VLM failed: {e}"

    result["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    return result


# ──────────────────────────────────────────────────────────────
# 生成 MD 报告
# ──────────────────────────────────────────────────────────────
def generate_report(
    report_path: str,
    dataset_path: str,
    vlm_model: str,
    total_count: int,
    valid_count: int,
    error_count: int,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    tp: int, fn: int, fp: int, tn: int,
    fp_list: List[Dict],
    fn_list: List[Dict],
    fp_sub_dist,
    fn_sub_dist,
    total_time: float,
    avg_time: float,
):
    """生成 FP/FN 分析报告 Markdown"""
    from datetime import date

    lines = []
    L = lines.append

    L("# 商品一致性判定 — SFT 测试集误判与漏判分析报告\n")
    L(f"> **数据集**: {dataset_path}")
    L(f"> **模型**: {vlm_model}")
    L(f"> **日期**: {date.today()}")
    L(f"> **任务**: 商品一致性 0分/非0分 二分类\n")
    L("---\n")

    # ── 一、总体指标 ──
    L("## 一、总体指标\n")
    L("| 指标 | 值 |")
    L("|------|:--:|")
    L(f"| 有效比对样本数 | {valid_count} |")
    L(f"| 跳过（无结果/空值） | {error_count} |")
    L(f"| **准确率 (Accuracy)** | **{accuracy*100:.2f}%** |")
    L(f"| **精准率 (Precision)** | **{precision*100:.2f}%** |")
    L(f"| **召回率 (Recall)** | **{recall*100:.2f}%** |")
    L(f"| **F1 Score** | **{f1*100:.2f}%** |\n")

    L("### 混淆矩阵\n")
    L("| | VLM预测=0分 | VLM预测=非0分 |")
    L("|--|:--:|:--:|")
    L(f"| **人工标注=0分** | TP = {tp} | FN = {fn} |")
    L(f"| **人工标注=非0分** | FP = {fp} | TN = {tn} |\n")
    L("---\n")

    # ── 二、误判分析 (FP) ──
    L("## 二、误判分析 (FP) — VLM判0分但人工判非0分".replace("0分但", "0分但"))
    L(f"共 {len(fp_list)} 条\n")
    L(f"### 子类型分布\n")
    if fp_sub_dist:
        L("| VLM判定子类型 | 数量 | 占比 |")
        L("|:--|:--:|:--:|")
        for sub, cnt in fp_sub_dist.most_common():
            pct = cnt / len(fp_list) * 100 if fp_list else 0
            L(f"| {sub} | {cnt} | {pct:.1f}% |")
        L("")
    else:
        L("无\n")
    L("> ⚠️ 核心问题：VLM在判定时过于严格，将大量正常的海报设计表达误判为0分。\n")
    L("### 逐条详情\n")
    if fp_list:
        L("| test_index | VLM子类型 | VLM判定理由 |")
        L("|:--|:--|:--|")
        for r in fp_list:
            sub = r.get("zero_sub_type") or "-"
            reason = (r.get("reason") or "").replace("|", "/").replace("\n", " ")[:120]
            L(f"| {r['test_index']} | {sub} | {reason} |")
        L("")
    L("---\n")

    # ── 三、漏判分析 (FN) ──
    L(f"## 三、漏判分析 (FN) — VLM判非0分但人工判0分")
    L(f"共 {len(fn_list)} 条\n")
    L("### 逐条详情\n")
    if fn_list:
        L("| test_index | VLM判定理由 |")
        L("|:--|:--|")
        for r in fn_list:
            reason = (r.get("reason") or "").replace("|", "/").replace("\n", " ")[:120]
            L(f"| {r['test_index']} | {reason} |")
        L("")
    L("---\n")

    # ── 四、总结与建议 ──
    L("## 四、总结与建议\n")
    L("| 问题 | 数量 | 建议 |")
    L("|:--|:--:|:--|")
    if fp_list:
        fp_a = fp_sub_dist.get("A_外观与包装不一致", 0)
        fp_b = fp_sub_dist.get("B_数量改变", 0)
        L(f"| **FP 误判** | {len(fp_list)} | 放宽判定阈值，区分'设计性差异'和'根本性差异'（A类{fp_a}条，B类{fp_b}条） |")
    if fn_list:
        L(f"| **FN 漏判** | {len(fn_list)} | 加强商品缺失、无主体、包装差异等场景的识别 |")
    L("")
    L(f"> ⏱️ 总耗时: {total_time:.1f}s | 平均: {avg_time:.1f}s/条")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ──────────────────────────────────────────────────────────────
# 批量评估
# ──────────────────────────────────────────────────────────────
async def batch_evaluate(
    dataset_path: str,
    vlm_model: str = VLM_MODEL,
    limit: Optional[int] = None,
    concurrency: int = 5,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    基于 SFT 测试集的批量评估，并与 ground truth 对比。
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if limit:
        test_data = test_data[:limit]

    if not output_path:
        safe_name = vlm_model.replace("/", "_").replace(".", "_")
        output_path = f"results_sft_test_{safe_name}.md"

    print(f"📂 测试集: {dataset_path} | 条数: {len(test_data)} | 并发: {concurrency}", flush=True)
    print(f"🤖 VLM: {vlm_model}", flush=True)
    print(f"📏 评估维度: 商品一致性（使用 product_consistency_evaluate.py 的 Prompt）", flush=True)

    sem = asyncio.Semaphore(concurrency)
    results: List[Optional[Dict[str, Any]]] = [None] * len(test_data)

    async def _run(i: int, sample: Dict[str, Any]):
        async with sem:
            test_index = sample["test_index"]
            images = sample["images"]
            messages = sample["messages"]

            user_content = messages[0]["content"]
            assistant_content = messages[1]["content"]

            # 解析 ground truth
            gt = parse_ground_truth(assistant_content)
            # 检测样本类型
            sample_type = detect_sample_type(user_content)

            has_logo = sample_type["has_logo"]

            # 图片映射：测试集格式为 [商品原图, 海报图, (可选)Logo/赠品参考图...]
            product_url = images[0]   # 第1张：商品参考图
            poster_url = images[1]    # 第2张：生成海报
            logo_url = images[2] if has_logo and len(images) > 2 else None

            try:
                res = await evaluate_single(
                    product_url=product_url,
                    poster_url=poster_url,
                    logo_url=logo_url,
                    vlm_model=vlm_model,
                )
            except Exception as e:
                res = {"error": str(e)}

            # 将 is_zero 映射为 是/否 与 ground truth 对比
            if res.get("is_zero") is True:
                pred_product = "否"  # is_zero=true → 商品不一致
            elif res.get("is_zero") is False:
                pred_product = "是"  # is_zero=false → 商品一致
            else:
                pred_product = None

            gt_product = gt.get("product")
            product_correct = (pred_product == gt_product) if (pred_product is not None and gt_product) else None

            item = {
                "test_index": test_index,
                "train_index": sample.get("train_index"),
                "has_logo": has_logo,
                "has_gift": sample_type["has_gift"],
                "num_images": len(images),
                # 模型预测
                "pred_product": pred_product,
                "is_zero": res.get("is_zero"),
                "zero_sub_type": res.get("zero_sub_type"),
                "reason": res.get("reason", ""),
                # ground truth
                "gt_product": gt_product,
                "gt_labels": gt,
                # 对比
                "product_correct": product_correct,
                # 元信息
                "model": vlm_model,
                "error": res.get("error"),
                "usage": res.get("usage", {}),
                "elapsed_seconds": res.get("elapsed_seconds", 0.0),
            }
            results[i] = item

            tag = "✅" if product_correct else ("❌" if product_correct is False else "⚠️")
            print(
                f"  [{i+1}/{len(test_data)}] idx={test_index} {tag}"
                f" pred={pred_product} gt={gt_product}"
                f" 耗时={item['elapsed_seconds']:.1f}s"
                + (f" | API错误: {item['error']}" if item.get("error") else ""),
                flush=True,
            )

    await asyncio.gather(*[_run(i, s) for i, s in enumerate(test_data)])

    # ── 统计（以「0分/商品不一致」为正例，与 fp_fn_analysis_v2.md 对齐）──
    valid = [r for r in results if r and r.get("product_correct") is not None]
    correct_count = sum(1 for r in valid if r["product_correct"])
    total_valid = len(valid)
    accuracy = correct_count / total_valid if total_valid else 0.0

    # 正例 = 0分（商品不一致, is_zero=true, pred="否"）
    tp = sum(1 for r in valid if r["gt_product"] == "否" and r["pred_product"] == "否")   # 都判0分
    fn = sum(1 for r in valid if r["gt_product"] == "否" and r["pred_product"] == "是")  # VLM漏判（判非0分, 实际0分）
    fp = sum(1 for r in valid if r["gt_product"] == "是" and r["pred_product"] == "否")  # VLM误判（判0分, 实际非0分）
    tn = sum(1 for r in valid if r["gt_product"] == "是" and r["pred_product"] == "是")   # 都判非0分

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    elapsed_all = [r.get("elapsed_seconds", 0.0) for r in results if r]
    total_time = sum(elapsed_all)
    avg_time = total_time / len(elapsed_all) if elapsed_all else 0.0

    error_count = sum(1 for r in results if r and r.get("error"))

    print(f"\n{'='*60}", flush=True)
    print(f"📊 SFT 测试集评估结果:", flush=True)
    print(f"   总样本: {len(test_data)} | 有效: {total_valid} | 错误: {error_count}", flush=True)
    print(f"   准确率: {accuracy*100:.1f}% ({correct_count}/{total_valid})", flush=True)
    print(f"   TP={tp} FN={fn} FP={fp} TN={tn}", flush=True)
    print(f"   Precision={precision*100:.1f}% Recall={recall*100:.1f}% F1={f1*100:.1f}%", flush=True)
    print(f"⏱️  总耗时: {total_time:.1f}s | 平均: {avg_time:.1f}s/条", flush=True)
    print(f"{'='*60}", flush=True)

    # ── FP/FN 分类 ──
    from collections import Counter

    fp_list = [r for r in valid if r["gt_product"] == "是" and r["pred_product"] == "否"]  # VLM判0分, GT非0分
    fn_list = [r for r in valid if r["gt_product"] == "否" and r["pred_product"] == "是"]  # VLM判非0分, GT 0分

    fp_sub_dist = Counter(r.get("zero_sub_type") or "未分类" for r in fp_list)
    fn_sub_dist = Counter(r.get("zero_sub_type") or "未分类" for r in fn_list)

    # ── 生成 MD 报告 ──
    report_path = output_path if output_path.endswith(".md") else output_path.replace(".json", ".md")
    generate_report(
        report_path=report_path,
        dataset_path=dataset_path,
        vlm_model=vlm_model,
        total_count=len(test_data),
        valid_count=total_valid,
        error_count=error_count,
        accuracy=accuracy, precision=precision, recall=recall, f1=f1,
        tp=tp, fn=fn, fp=fp, tn=tn,
        fp_list=fp_list, fn_list=fn_list,
        fp_sub_dist=fp_sub_dist, fn_sub_dist=fn_sub_dist,
        total_time=total_time, avg_time=avg_time,
    )
    print(f"\n💾 报告已写入: {report_path}", flush=True)

    return results


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("🧪 商品一致性评估 — SFT 测试集")
    print("   Prompt: product_consistency_evaluate.py (不变)")
    print("=" * 60)

    await batch_evaluate(
        dataset_path=TEST_DATASET,
        vlm_model=VLM_MODEL,
        limit=None,        # 全量 200 条
        concurrency=30,    # 并行 30
        output_path="results_sft_test_qwen3_7-plus_v2.md",
    )


if __name__ == "__main__":
    asyncio.run(main())
