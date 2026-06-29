"""
分维度海报评估框架 — 文字正确性（单 VLM 二分类，参考图增强）

链路：单次 VLM 调用，传入 3 张图
  - 图1「主体图」+ 图2「logo」→ 参考基准（品牌名、产品名、包装文字以此为准）
  - 图3「生成图」→ 检查对象
  - VLM 对比参考图，仅判断生成图的文字渲染/生成错误（错别字/乱码/畸变/缺失/不一致）
用法：
  - 直接运行 → results_<model>.json + .csv
"""
import asyncio, httpx, json, logging, traceback, csv, time
from typing import List, Dict, Any, Tuple, Optional

from dataset_loader import load_dataset

KEY = "RFI5VHdQRG1KUGVoRlRsQWg4c09zM3VjTjN0ckk5bVJyVVJKM1g4MnNSQzo0Q0VrUXVtT21sRXFSdmhvNXR1OXpmMGVkbnZDUVRmbw=="
BASE = "https://aiboost.zacz.cn/api/v1"
VLM_MODEL = "qwen3-vl-plus"      # 单 VLM 一步到位

# ──────────────────────────────────────────────────────────────
# Prompt：单 VLM 一步到位（识别 + 判错 + 二分类）
# ──────────────────────────────────────────────────────────────
PROMPT_TEXT_CORRECTNESS = """你是一个 AI 生成海报的文字渲染质量检查员。

🚨🚨🚨 绝对禁令 🚨🚨🚨
你的唯一且仅有的任务是检查**文字渲染/生成正确性**。以下内容绝对禁止检查、评价、或输出：
- ❌ 禁止判断图片美观度、设计水平、配色、排版、字体风格
- ❌ 禁止判断商品是否好看、图片是否清晰、构图是否合理
- ❌ 禁止判断营销内容是否合理、价格是否真实、文案是否有语病
- ❌ 禁止判断商品与文字是否匹配、产品规格是否准确
- ❌ 禁止对非文字元素（图像、图标、背景、人物）做任何评价
✅ 你只能做一件事：逐字检查图3的文字有没有错别字、乱码、重复、畸变、缺失、与参考图不一致。
如果图3所有文字在渲染层面都正确，直接输出 is_correct=true。

你的唯一任务：检查第 3 张图（AI 生成的营销海报）的文字是否存在**渲染/生成层面的错误**。第 1、2 张图仅作参考，不需要检查其中文字。

━━━━━━━━━━━━━━━━━━━━━━━━
📷 输入的三张图
━━━━━━━━━━━━━━━━━━━━━━━━
- 图1「主体图」：原始商品图（参考用）—— 品牌名、产品名、包装文字以此为准
- 图2「logo」：品牌 logo 标准样式（参考用）—— 品牌 logo 的文字/设计以此为准
- 图3「生成图」：AI 生成的营销海报（检查对象）—— 只检查这张图的文字

━━━━━━━━━━━━━━━━━━━━━━━━
🔍 检查目标（仅图3）
━━━━━━━━━━━━━━━━━━━━━━━━
A. **文案文字**：主标题、副标题、卖点描述、价格、优惠信息、按钮、角标、底部说明等
B. **商品包装文字**：图3中商品本身/包装上的文字，包括品牌名、产品名、规格、标签、说明等

━━━━━━━━━━━━━━━━━━━━━━━━
📋 检查项（仅限文字渲染/生成错误，命中任一项 → is_correct=false）
━━━━━━━━━━━━━━━━━━━━━━━━
1. **错别字**：汉字/英文/数字拼写错误、形近字误用、同音字误用
2. **乱码**：不可识别的符号、无意义字符串、语义完全错乱的文字堆叠（AI 生成常见）
3. **重复**：同一单词/短语连续重复出现，如「优惠优惠」「HIMALAYAHIMALAYA」
4. **畸变**（重点）：字形扭曲、笔画错乱粘连、汉字结构崩坏、伪汉字、无法辨认的变形字
5. **缺失**：明显漏字/缺笔画/半个字、文本被截断、规格数字后面没单位
6. **与参考图不一致**：品牌名/产品名/logo文字与第1、2张参考图明显不同（以参考图为准）

━━━━━━━━━━━━━━━━━━━━━━━━
🚫 严格禁止的"过度思考"行为 —— 以下情况一律不算错误
━━━━━━━━━━━━━━━━━━━━━━━━
- ❌ 不要判断年份是否合理（图中写「2025春季新款」就是2025，不因为今年是2026而判错）
- ❌ 不要判断价格是否真实（图中标多少钱就是多少钱，不判断贵了还是便宜了）
- ❌ 不要判断营销文案是否夸大、是否有语病、是否符合品牌调性
- ❌ 不要判断产品规格参数是否真实（比如「净含量500g」不要质疑应该是 480g）
- ❌ 不要判断商品图片与文字是否匹配（比如写着「草莓味」但图里像蓝莓——不管）
- ❌ 不判断设计美学层面的字体选择、字号大小、配色是否合理
- 设计性艺术字体只要能辨认字义就算正确
- 品牌英文 logo 装饰性排版（字母拆解、堆叠、镜像）属于正常设计
- 中英文混排、特殊品牌写法（iPhone、CHANDO HIMALAYA）均属正常
- 划线价、货币符号、emoji、感叹号等营销元素不算错误

━━━━━━━━━━━━━━━━━━━━━━━━
输出（严格 JSON，不要 markdown 包裹）
━━━━━━━━━━━━━━━━━━━━━━━━
{
  "is_correct": true,
  "errors": [
    {
      "source": "文案|包装",
      "type": "错别字|乱码|重复|畸变|缺失|与参考图不一致",
      "wrong": "有问题的原文",
      "correct": "应该是什么（未知则填\"\"）",
      "context": "出现在图3什么位置，如「右上角价格下方」「商品包装正面」"
    }
  ],
  "poster_text": "按视觉顺序逐行输出图3生成图上提取出的所有文字，保留原始字符，包括不确定的乱码/畸变字",
  "reason": "判定理由（一句话）"
}

⚠️ is_correct=true 表示图3两类文字完全没有渲染/生成错误，errors 必须为空数组。
⚠️ 只要有任何一处错误，is_correct=false。
⚠️ 不要输出 markdown 代码块标记，直接输出 JSON。"""

# ──────────────────────────────────────────────────────────────
# 底层调用：VLM 单图 一次调用完成识别 + 判错
# ──────────────────────────────────────────────────────────────
async def vlm_check(image_urls: List[str], model: str = VLM_MODEL, max_retries: int = 3) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """单次 VLM 调用：传入 [主体图, logo, 生成图]，输出 {is_correct, errors, poster_text, reason}
    带重试，遇到网络/限流/超时错误退避重试。"""
    # 构建多图 content：每张图一个 image_url，最后跟 prompt 文本
    content_parts = []
    for url in image_urls:
        if url and url.startswith("data:image/"):
            content_parts.append({"type": "image_url", "image_url": {"url": url}})
    content_parts.append({"type": "text", "text": PROMPT_TEXT_CORRECTNESS})

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
                    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                    json=payload,
                )
                r.raise_for_status()
                d = r.json()
            raw = d["choices"][0]["message"]["content"].strip()
            usage = d.get("usage", {})
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].removesuffix("```").strip()
            try:
                return json.loads(raw), usage
            except json.JSONDecodeError:
                return {"raw": raw, "parse_error": True}, usage
        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
            last_err = e
            wait = 2 ** attempt
            logging.warning(f"VLM 重试 {attempt+1}/{max_retries} ({type(e).__name__}): {e}\u3001等待 {wait}s")
            await asyncio.sleep(wait)
        except Exception as e:
            last_err = e
            break
    raise RuntimeError(f"VLM 调用多次失败: {type(last_err).__name__}: {last_err}")


# ──────────────────────────────────────────────────────────────
# 文字正确性评估（单 VLM 一步）
# ──────────────────────────────────────────────────────────────
async def evaluate_text_correctness(
    poster_url: str,
    vlm_model: str = VLM_MODEL,
    ref_main_url: Optional[str] = None,
    ref_logo_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    文字正确性评估（单 VLM 二分类）
    传入：生成图 + 参考主体图 + 参考 logo，一并交给 VLM 对比检查
    返回：{is_correct, errors, poster_text, reason, usage, elapsed_seconds}
    """
    result: Dict[str, Any] = {
        "poster_text": "", "is_correct": None,
        "errors": [], "reason": "", "usage": {}, "elapsed_seconds": 0.0,
    }
    t0 = time.perf_counter()
    try:
        image_urls = [ref_main_url, ref_logo_url, poster_url]  # 顺序：主体图、logo、生成图
        check, usage = await vlm_check(image_urls, vlm_model)
        result["usage"] = usage
        if check.get("parse_error"):
            result["error"] = "VLM 返回非 JSON"
            result["raw"] = check.get("raw", "")
        else:
            result["is_correct"] = bool(check.get("is_correct"))
            result["errors"] = check.get("errors", [])
            result["poster_text"] = check.get("poster_text", "")
            result["reason"] = check.get("reason", "")
    except Exception as e:
        logging.error(f"VLM 调用失败: {traceback.format_exc()}")
        result["error"] = f"VLM failed: {e}"
    result["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    return result


# ──────────────────────────────────────────────────────────────
# 批量评估：跑 xlsx 数据集（支持指定模型）
# ──────────────────────────────────────────────────────────────
async def batch_evaluate_text_correctness(
    xlsx_path: str,
    vlm_model: str = VLM_MODEL,
    limit: Optional[int] = None,
    concurrency: int = 3,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    批量「文字正确性」评估（二分类）
    - vlm_model: 使用的 VLM（单模型一步到位）
    - output_path: 默认按模型名命名 results_<vlm>.json
    """
    records = load_dataset(xlsx_path)
    if limit:
        records = records[:limit]
    if not output_path:
        safe = lambda s: s.replace("/", "_").replace(".", "_")
        output_path = f"results_{safe(vlm_model)}.json"

    print(f"📂 数据集: {xlsx_path} | 条数: {len(records)} | 并发: {concurrency}", flush=True)
    print(f"🤖 VLM: {vlm_model}", flush=True)

    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = [None] * len(records)

    async def _run(i: int, rec: Dict[str, Any]):
        async with sem:
            row = rec["row"]
            try:
                res = await evaluate_text_correctness(
                    poster_url=rec["生成图"],
                    vlm_model=vlm_model,
                    ref_main_url=rec.get("主体图"),
                    ref_logo_url=rec.get("logo"),
                )
            except Exception as e:
                res = {"error": str(e)}
            item = {
                "row": row,
                "is_correct": res.get("is_correct"),
                "errors": res.get("errors", []),
                "poster_text": res.get("poster_text", ""),
                "reason": res.get("reason", ""),
                "model": vlm_model,
                "error": res.get("error"),
                "usage": res.get("usage", {}),
                "elapsed_seconds": res.get("elapsed_seconds", 0.0),
            }
            results[i] = item
            tag = "✅" if item["is_correct"] else ("❌" if item["is_correct"] is False else "⚠️")
            err_cnt = len(item["errors"]) if isinstance(item["errors"], list) else 0
            print(f"  [{i+1}/{len(records)}] row={row} {tag} errors={err_cnt} 耗时={item['elapsed_seconds']:.1f}s"
                  + (f" | API错误: {item['error']}" if item.get("error") else ""), flush=True)

    await asyncio.gather(*[_run(i, r) for i, r in enumerate(records)])

    valid = [r for r in results if r and r.get("is_correct") is not None]
    pos = sum(1 for r in valid if r["is_correct"])
    neg = len(valid) - pos

    # 计算耗时统计
    elapsed_all = [r.get("elapsed_seconds", 0.0) for r in results if r]
    total_time = sum(elapsed_all)
    avg_time = total_time / len(elapsed_all) if elapsed_all else 0.0
    print(f"\n📊 模型判断: 有效 {len(valid)}/{len(records)} | 正确 {pos} | 不正确 {neg}", flush=True)
    print(f"⏱️  总耗时: {total_time:.1f}s | 平均每张: {avg_time:.1f}s | 图片数: {len(elapsed_all)}", flush=True)

    # ── 写 JSON ──
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON 已写入: {output_path}", flush=True)

    # ── 写 CSV ──
    csv_path = output_path.replace(".json", ".csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "is_correct", "errors_count", "error_summary",
                          "poster_text", "reason", "elapsed_seconds"])
        for r in results:
            if not r:
                continue
            errors_list = r.get("errors") or []
            error_summary = "; ".join(
                f"{e.get('source','?')}-{e.get('type','?')}: {e.get('wrong','')[:40]}→{e.get('correct','')[:40]}"
                for e in errors_list
            ) if errors_list else ""
            writer.writerow([
                r.get("row"),
                r.get("is_correct"),
                len(errors_list),
                error_summary,
                r.get("poster_text", ""),
                r.get("reason", ""),
                r.get("elapsed_seconds", 0.0),
            ])
    print(f"💾 CSV 已写入: {csv_path}", flush=True)

    return results


# ──────────────────────────────────────────────────────────────
# 测试入口
# ──────────────────────────────────────────────────────────────
VLM_MODELS = [
    "qwen3-vl-plus",
]

DATASET = "data/dataset_without_gift_label.xlsx"


async def main():
    print("=" * 60)
    print("🧪 文字正确性二分类评估（参考图增强）")
    print("=" * 60)

    for vlm in VLM_MODELS:
        await batch_evaluate_text_correctness(
            xlsx_path=DATASET,
            vlm_model=vlm,
            limit=None,        # 设为数字可限制测试条数；None 跑全量
            concurrency=5,
        )


if __name__ == "__main__":
    asyncio.run(main())
