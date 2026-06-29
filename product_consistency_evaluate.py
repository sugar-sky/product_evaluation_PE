"""
商品一致性评估 — 二分类（0分 / 非0分）

评估维度：仅评估「商品主体一致性」（PRODUCT_MISMATCH）
  - 0分子类型：
    A. 商品外观与包装不一致（品类、外观、形状、颜色、比例、完整性、轮廓、包装、Logo、遮挡）
    B. 商品数量改变（重点关注影响总价/单价的商品数量增减）
    C. 海报包装化
  - 非0分：商品与参考图一致

链路：单次 VLM 调用，传入 2 张图
  - 图1「主体图」→ 商品参考基准
  - 图2「生成图」→ 检查对象

模型：qwen3.7-plus

用法：
  python product_consistency_evaluate.py
  → 输出 results_product_consistency_<model>.json + .csv
"""
import asyncio
import csv
import json
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import httpx

from dataset_loader import load_dataset

# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────
KEY = "RFI5VHdQRG1KUGVoRlRsQWg4c09zM3VjTjN0ckk5bVJyVVJKM1g4MnNSQzo0Q0VrUXVtT21sRXFSdmhvNXR1OXpmMGVkbnZDUVRmbw=="
BASE = "https://aiboost.zacz.cn/api/v1"
VLM_MODEL = "qwen3.7-plus"

# ──────────────────────────────────────────────────────────────
# Prompt：商品一致性二分类评估（动态拼装）
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

   【外观/形状】商品外观、形状发生可感知的变化：
     - 包括拉伸、压缩、扭曲等任何可察觉的形变（即使是轻微的）
     - 瓶身/盒身/管身比例失真（变胖、变窄、变矮、变高）
     - 商品部件位置发生改变（如盖子分离悬浮、标签移位等）

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
   - 参考图中有N个商品，海报中商品数量发生增减（包括主体商品和附属品）
   - 重点关注：数量变化是否会影响消费者对商品总价或单价的判断
     （如原本1个变为2个、3个变为10+个、套装组合中增加或减少品类件数）
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
# VLM 调用
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
        if url and url.startswith("data:image/"):
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
            wait = 2**attempt
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
async def evaluate_product_consistency(
    poster_url: str,
    product_url: str,
    logo_url: Optional[str] = None,
    vlm_model: str = VLM_MODEL,
) -> Dict[str, Any]:
    """
    商品一致性评估（二分类）。
    传入：生成图 + 主体商品参考图 + Logo参考图（可选）
    返回：{is_zero, zero_reason, zero_sub_type, product_location, reason, usage, elapsed_seconds}
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
        has_logo = bool(logo_url and logo_url.startswith("data:image/"))
        # 图片顺序：主体图, [Logo图], 生成图
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
# 批量评估
# ──────────────────────────────────────────────────────────────
async def batch_evaluate_product_consistency(
    xlsx_path: str,
    vlm_model: str = VLM_MODEL,
    limit: Optional[int] = None,
    concurrency: int = 5,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    批量商品一致性评估（二分类：0分/非0分）。
    """
    records = load_dataset(xlsx_path)
    if limit:
        records = records[:limit]
    if not output_path:
        safe_name = vlm_model.replace("/", "_").replace(".", "_")
        output_path = f"results_product_consistency_{safe_name}.json"

    print(f"📂 数据集: {xlsx_path} | 条数: {len(records)} | 并发: {concurrency}", flush=True)
    print(f"🤖 VLM: {vlm_model}", flush=True)
    print(f"📏 评估维度: 商品一致性（二分类）", flush=True)

    sem = asyncio.Semaphore(concurrency)
    results: List[Optional[Dict[str, Any]]] = [None] * len(records)

    async def _run(i: int, rec: Dict[str, Any]):
        async with sem:
            row = rec["row"]
            poster_url = rec.get("生成图")
            product_url = rec.get("主体图")

            if not poster_url or not product_url:
                results[i] = {
                    "row": row,
                    "is_zero": None,
                    "zero_reason": None,
                    "reason": "",
                    "error": "缺少主体图或生成图",
                    "elapsed_seconds": 0.0,
                }
                print(f"  [{i+1}/{len(records)}] row={row} ⚠️ 缺少图片，跳过", flush=True)
                return

            logo_url = rec.get("logo")
            # 检查logo是否为有效base64图片
            if not (logo_url and isinstance(logo_url, str) and logo_url.startswith("data:image/")):
                logo_url = None

            try:
                res = await evaluate_product_consistency(
                    poster_url=poster_url,
                    product_url=product_url,
                    logo_url=logo_url,
                    vlm_model=vlm_model,
                )
            except Exception as e:
                res = {"error": str(e)}

            item = {
                "row": row,
                "has_logo": logo_url is not None,
                "is_zero": res.get("is_zero"),
                "zero_reason": res.get("zero_reason"),
                "zero_sub_type": res.get("zero_sub_type"),
                "product_location": res.get("product_location", ""),
                "reason": res.get("reason", ""),
                "model": vlm_model,
                "error": res.get("error"),
                "usage": res.get("usage", {}),
                "elapsed_seconds": res.get("elapsed_seconds", 0.0),
            }
            results[i] = item

            tag = "🔴 0分" if item["is_zero"] else ("🟢 非0" if item["is_zero"] is False else "⚠️")
            sub = item.get('zero_sub_type') or '-'
            print(
                f"  [{i+1}/{len(records)}] row={row} {tag}"
                f" sub_type={sub}"
                f" 耗时={item['elapsed_seconds']:.1f}s"
                + (f" | API错误: {item['error']}" if item.get("error") else ""),
                flush=True,
            )

    await asyncio.gather(*[_run(i, r) for i, r in enumerate(records)])

    # ── 统计 ──
    valid = [r for r in results if r and r.get("is_zero") is not None]
    zero_count = sum(1 for r in valid if r["is_zero"])
    non_zero_count = len(valid) - zero_count

    elapsed_all = [r.get("elapsed_seconds", 0.0) for r in results if r]
    total_time = sum(elapsed_all)
    avg_time = total_time / len(elapsed_all) if elapsed_all else 0.0

    print(f"\n{'='*60}", flush=True)
    print(f"📊 评估结果统计:", flush=True)
    print(f"   有效: {len(valid)}/{len(records)}", flush=True)
    print(f"   0分: {zero_count} | 非0分: {non_zero_count}", flush=True)
    print(f"   0分率: {zero_count/len(valid)*100:.1f}%" if valid else "", flush=True)
    print(f"⏱️  总耗时: {total_time:.1f}s | 平均: {avg_time:.1f}s/张", flush=True)
    print(f"{'='*60}", flush=True)

    # ── 0分原因分布 ──
    from collections import Counter

    reason_dist = Counter(r["zero_sub_type"] for r in valid if r["is_zero"] and r.get("zero_sub_type"))
    if reason_dist:
        print(f"\n📋 0分子类型分布:", flush=True)
        for reason, cnt in reason_dist.most_common():
            print(f"   {reason}: {cnt}", flush=True)

    # ── 写 JSON ──
    output_data = {
        "meta": {
            "total_count": len(records),
            "valid_count": len(valid),
            "zero_count": zero_count,
            "non_zero_count": non_zero_count,
            "zero_rate": round(zero_count / len(valid), 4) if valid else 0,
            "model": vlm_model,
            "dimension": "product_consistency",
            "classification": "binary (0分/非0分)",
            "total_seconds": round(total_time, 1),
            "avg_seconds": round(avg_time, 1),
        },
        "results": [r for r in results if r],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 JSON 已写入: {output_path}", flush=True)

    # ── 写 CSV ──
    csv_path = output_path.replace(".json", ".csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "row", "has_logo", "is_zero", "zero_reason", "zero_sub_type",
            "product_location", "reason", "elapsed_seconds",
        ])
        for r in results:
            if not r:
                continue
            writer.writerow([
                r.get("row"),
                r.get("has_logo", False),
                r.get("is_zero"),
                r.get("zero_reason", ""),
                r.get("zero_sub_type", ""),
                r.get("product_location", ""),
                r.get("reason", ""),
                r.get("elapsed_seconds", 0.0),
            ])
    print(f"💾 CSV 已写入: {csv_path}", flush=True)

    return results


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────
DATASET = "data/dataset_without_gift_label.xlsx"


async def main():
    print("=" * 60)
    print("🧪 商品一致性评估 — 二分类（0分 / 非0分）")
    print("=" * 60)

    await batch_evaluate_product_consistency(
        xlsx_path=DATASET,
        vlm_model=VLM_MODEL,
        limit=None,       # 全量300条
        concurrency=30,   # 并行30
        output_path="results_product_consistency_qwen3_7-plus_v2.json",
    )


if __name__ == "__main__":
    asyncio.run(main())
