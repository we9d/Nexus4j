# ===== W35: ประเมินคำถาม 5 ข้อ คะแนน 0 ถึง 3 และ precision@5 =====
# ต้องรันต่อจากขั้นตอนที่ 18, 19 (W32) และ 20 (W33) ใน runtime เดียวกัน
import re
import datetime
import pandas as pd

EVAL_K = 5

# relevant = chunk_id ที่ถือว่าเป็นหลักฐานที่ถูกต้อง, keywords = คำที่ต้องอยู่ในคำตอบ
# method: vector = ค้นด้วยเวกเตอร์, graph = เวกเตอร์แล้วเดินกราฟ 2 ระดับ, not_found = ต้องตอบว่าไม่พบ
EVAL_QUESTIONS = [
    {"qid": "Q1", "method": "vector",
     "question": "การสกัดน้ำมันปาล์มดิบชุดหลักใช้ผลปาล์มทะลายสดกี่ kg ต่อน้ำมันปาล์มดิบ 1 kg",
     "relevant": ["KC022"], "keywords": ["5.95"]},
    {"qid": "Q2", "method": "vector",
     "question": "การปลูกปาล์มน้ำมันภาคใต้ต้องใช้น้ำชลประทานเท่าไร",
     "relevant": ["KC018"], "keywords": ["183"]},
    {"qid": "Q3", "method": "vector",
     "question": "การสกัดน้ำมันปาล์มดิบชุดเปรียบเทียบใช้ไฟฟ้าจากชีวมวล CHP เท่าไร",
     "relevant": ["KC035"], "keywords": ["84"]},
    {"qid": "Q4", "method": "graph",
     "question": "ผลปาล์มทะลายสดที่ขนส่งไปโรงสกัด มาจากการปลูกที่อ้างอิงงานวิจัยซึ่งตีพิมพ์ในวารสารใด",
     "relevant": ["KC019", "KC020", "KC021"], "keywords": ["Journal of Cleaner Production"]},
    {"qid": "Q5", "method": "not_found",
     "question": "รถไฟฟ้าสายสีม่วงเที่ยวสุดท้ายออกกี่โมง",
     "relevant": [], "keywords": [NOT_FOUND]},
]


def graph_rag_answer(question, k=EVAL_K):
    """เวกเตอร์ + เดินกราฟ 2 ระดับ ใช้ฟังก์ชันจากขั้นตอนที่ 20"""
    hits = vector_search(question, k=k)
    if not hits or hits[0]["score"] < SIM_THRESHOLD:
        return {"hits": hits, "answer": NOT_FOUND, "mode": "below_threshold"}
    paths = walk_to_sources(seeds_from_hits(hits))
    lines = evidence_from_hits(hits) + graph_evidence(paths)
    generated = call_gemini(build_prompt(question, lines))
    if generated:
        return {"hits": hits, "answer": generated, "mode": "llm"}
    journals = sorted({j for j in paths["journal"] if j})
    answer = ("จากการเดินกราฟ พบวารสาร: " + ", ".join(journals)) if journals else NOT_FOUND
    return {"hits": hits, "answer": answer, "mode": "template"}


def score_answer(item, answer, top_ids):
    """เกณฑ์คะแนนตามสมุดของอาจารย์
    3 = ถูก ครบ และอ้างรหัสหลักฐาน | 2 = ถูกบางส่วนและมีหลักฐาน
    1 = มีทิศทางแต่ไม่ครบ | 0 = ไม่ตอบหรือตอบนอกหลักฐาน
    ข้อที่ต้องตอบว่าไม่พบ ได้ 3 เมื่อระบบตอบว่าไม่พบข้อมูลที่เกี่ยวข้อง"""
    if item["method"] == "not_found":
        return (3 if NOT_FOUND in answer else 0), "ต้องปฏิเสธ"
    if NOT_FOUND in answer:
        return 0, "ตอบว่าไม่พบ ทั้งที่มีหลักฐาน"
    has_kw = all(kw.lower() in answer.lower() for kw in item["keywords"])
    cited = set(re.findall(r"KC\d{3}", answer))
    cites_relevant = bool(cited & set(item["relevant"])) or (item["method"] == "graph" and bool(re.search(r"\[\s*G\d+", answer)))
    relevant_retrieved = bool(set(item["relevant"]) & set(top_ids))
    if has_kw and cites_relevant:
        return 3, "คำตอบถูกและอ้างหลักฐาน"
    if has_kw:
        return 2, "คำตอบถูกแต่ไม่อ้างรหัสหลักฐานที่ถูกต้อง"
    if relevant_retrieved:
        return 1, "ค้นเจอหลักฐานแต่คำตอบไม่มีค่าที่ถูกต้อง"
    return 0, "ไม่พบหลักฐานที่ถูกต้องใน top-k"


rows = []
for item in EVAL_QUESTIONS:
    if item["method"] == "graph":
        r = graph_rag_answer(item["question"])
    else:
        r = rag_answer(item["question"], k=EVAL_K, verbose=False)
    top_ids = [h["chunk_id"] for h in r["hits"]]
    score, reason = score_answer(item, r["answer"], top_ids)
    if item["relevant"]:
        hit_n = len(set(item["relevant"]) & set(top_ids))
        p_at_k = round(hit_n / EVAL_K, 2)
    else:
        hit_n, p_at_k = None, None  # ข้อที่ต้องตอบว่าไม่พบ ไม่มีหลักฐานที่ถูกต้อง จึงไม่คิด P@5
    rows.append({
        "qid": item["qid"], "question": item["question"], "method": item["method"],
        "relevant_ids": ", ".join(item["relevant"]) or "-",
        "expected_keyword": ", ".join(item["keywords"]),
        "top_k_ids": ", ".join(top_ids),
        "top1_score": r["hits"][0]["score"] if r["hits"] else None,
        "relevant_in_top_k": hit_n, "precision_at_k": p_at_k,
        "score_0_to_3": score, "score_reason": reason,
        "answer_mode": r["mode"], "answer": r["answer"],
    })

eval_df = pd.DataFrame(rows)
display(eval_df[["qid", "method", "relevant_ids", "top_k_ids", "relevant_in_top_k",
                 "precision_at_k", "score_0_to_3", "answer_mode"]])
for r in rows:
    print(f"{r['qid']} [{r['answer_mode']}] คะแนน {r['score_0_to_3']} ({r['score_reason']}) | คำตอบ: {r['answer']}")

mean_score = eval_df["score_0_to_3"].mean()
mean_p = eval_df["precision_at_k"].dropna().mean()
print("-" * 70)
print(f"คะแนนเฉลี่ย: {mean_score:.2f} จาก 3 | precision@{EVAL_K} เฉลี่ย (Q ที่มีหลักฐาน): {mean_p:.2f}")
print("หมายเหตุ: คะแนนคิดอัตโนมัติจากคำสำคัญและรหัสที่อ้าง QA ควรอ่านคอลัมน์ answer ทวนอีกรอบ")
if (eval_df["answer_mode"] == "template").any():
    print("คำเตือน: มีข้อที่ตอบแบบแม่แบบ ไม่ได้มาจาก Gemini ต้องเขียนให้ตรงในเล่ม หรือรันใหม่เมื่อ Gemini ว่าง")

eval_df.to_csv("cn230_w14_eval_results.csv", index=False, encoding="utf-8-sig")
model_used = GEMINI_MODEL if ANSWER_MODE == "llm" else "template"
md = [
    "# ผลการประเมิน 5 คำถาม (S02-G07)",
    "",
    f"- วันที่รัน: {datetime.datetime.now():%Y-%m-%d %H:%M}",
    f"- โมเดลสร้างคำตอบ: {model_used} | top-k = {EVAL_K} | เกณฑ์ความคล้าย = {SIM_THRESHOLD}",
    f"- คะแนนเฉลี่ย: {mean_score:.2f} จาก 3 | precision@{EVAL_K} เฉลี่ย: {mean_p:.2f} (ไม่รวม Q5)",
    "",
    "| ข้อ | วิธี | รหัสที่ถูกต้อง | top-k | P@5 | คะแนน | โหมด |",
    "|---|---|---|---|---|---|---|",
]
for r in rows:
    p = "-" if r["precision_at_k"] is None else f"{r['precision_at_k']:.2f}"
    md.append(f"| {r['qid']} | {r['method']} | {r['relevant_ids']} | {r['top_k_ids']} | {p} | {r['score_0_to_3']} | {r['answer_mode']} |")
md += ["", "## คำตอบแต่ละข้อ", ""]
for r in rows:
    md += [f"### {r['qid']} {r['question']}", "", f"คะแนน {r['score_0_to_3']}: {r['score_reason']}", "", r["answer"], ""]
with open("cn230_w14_eval_results.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("บันทึก cn230_w14_eval_results.csv และ cn230_w14_eval_results.md แล้ว")
