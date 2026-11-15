"""ทดสอบเว็บด้วยโมเดล MiniLM ตัวจริง ครบ 8 เกณฑ์ของบท 6.5 และทดสอบ Gemini จริงถ้าตั้งค่าคีย์ไว้

วิธีใช้ ในโฟลเดอร์ web ที่เปิด venv แล้ว
    python3 tests/selftest.py
สคริปต์นี้ไม่พิมพ์คีย์ออกมา แคปผลทั้งหมดส่งกลับได้เลย
ถ้าต้องการทดสอบเร็วโดยไม่โหลดโมเดลจริง ให้ตั้ง S02G07_TEST_EMBEDDER=1 (คะแนนความคล้ายจะไม่ตรงกับเล่ม)
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ให้ import app จากโฟลเดอร์ web ได้

t0 = time.perf_counter()
import app  # noqa: E402  loads MiniLM and the 40 chunks KC001-KC040

print(f"โหลดโมเดลและข้อมูล {time.perf_counter() - t0:.1f} วินาที")
print("โมเดล", app.EMBEDDER.name, "| Chunk", len(app.CHUNK_IDS), "| k", app.TOP_K, "| เกณฑ์", app.SIMILARITY_THRESHOLD)
print("Gemini", f"ตั้งค่าคีย์แล้ว รุ่น {app.GEMINI_MODEL} สำรอง {app.GEMINI_FALLBACK}" if app.gemini_ready() else "ยังไม่ได้ตั้งค่าคีย์ จะข้ามการทดสอบ Gemini")
print()

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print("ผ่าน " if ok else "ต้องดู", "|", name, ("| " + detail) if detail else "")


# เกณฑ์ 4 และ 5: Chunk มีแหล่งที่มา และค้นห้าอันดับพร้อม source_id
check("Chunk 40 รายการ ทุกรายการมี source_id", len(app.CHUNK_IDS) == 40 and app.CPROV["source_id"].notna().all())

# เกณฑ์ 7: คำถามเวกเตอร์ Q1 ถึง Q3 และ Q5
for item in app.EVAL_QUESTIONS:
    if item["method"] == "graph":
        continue
    r, hits = app.search_result(item["question"])
    top = [h["chunk_id"] for h in hits]
    if item["method"] == "not_found":
        check(f"{item['qid']} ต้องตอบว่าไม่พบ", r["mode"] in ("not_found", "check"), f"สูงสุด {r['top_similarity']:.4f}")
    else:
        check(f"{item['qid']} หลักฐานที่ถูกอยู่อันดับ 1", top[:1] == item["relevant"][:1],
              f"top5 {', '.join(top)} สูงสุด {r['top_similarity']:.4f} (เล่มรายงาน Q1 0.7368 Q2 0.6778 Q3 0.8342)")
        check(f"{item['qid']} มีค่า {item['keywords'][0]} ในคำตอบ", item["keywords"][0] in r["answer"], r["answer"][:120])

for q in ["โนบิตะผลิตปาล์มน้ำมันได้เท่าไหร่", "ราคาปาล์มวันนี้เท่าไร"]:
    r, _ = app.search_result(q)
    check(f"คำถามนอกคลัง {q}", r["mode"] in ("not_found", "check"), r["mode"])

# เกณฑ์ 6: Graph RAG Q4 ต้องพบชื่อวารสารผ่านการเดินกราฟ 2 ระดับ
q4 = next(i for i in app.EVAL_QUESTIONS if i["method"] == "graph")
g = app.graph_result(q4["question"])
check("Q4 เวกเตอร์อย่างเดียวไม่มีชื่อวารสารในหลักฐาน", not g["vector"]["journals_in_evidence"])
check("Q4 กราฟพบ Journal of Cleaner Production", "Journal of Cleaner Production" in g["graph"]["answer"],
      f"{len(g['paths'])} เส้นทาง ระดับ 2 จำนวน {sum(p['hops'] == 2 for p in g['paths'])}")

# เกณฑ์ 7: รันประเมินทั้งชุดแบบไม่ใช้ AI
ev = app.run_eval(False)
check("ประเมิน 5 ข้อ คะแนนเฉลี่ย", ev["mean_score"] >= 2.5, f"เฉลี่ย {ev['mean_score']} P@5 {ev['mean_precision']} รายข้อ " +
      " ".join(f"{r['qid']}={r['score']}" for r in ev["rows"]))

# เกณฑ์ 8: แก้ KC022 แล้ว UPSERT จำนวนแถวต้องเท่าเดิม แล้วคืนค่า
q1 = app.EVAL_QUESTIONS[0]["question"]
u = app.upsert_chunk("KC022", "6.20", q1)
check("UPSERT KC022 5.95 เป็น 6.20 จำนวนแถวคงเดิม", u["rows_before"] == u["rows_after"] == 40 and u["rows_with_id"] == 1,
      f"โคไซน์ {u['cosine_old_new']}")
check("คำตอบหลังแก้มีค่า 6.20", "6.20" in u["after"]["answer"], u["after"]["answer"][:120])
rs = app.reset_chunks()
check("คืนค่าเดิมครบ ข้อความและเวกเตอร์ตรงกับตอนเริ่ม", rs["matches_original"] and "5.95" in app.CHUNKS.at[app.CHUNK_INDEX["KC022"], "text"])

if app.gemini_ready():
    print("\nทดสอบ Gemini จริง")
    for item in app.EVAL_QUESTIONS[:3]:
        r = app.gemini_result(item["question"])
        check(f"Gemini {item['qid']}", r.get("ok") and item["keywords"][0] in r.get("answer", ""),
              f"รุ่น {r.get('model', '-')} {r.get('generate_ms', 0)} ms {r.get('answer') or r.get('notice')}")
    gg = app.graph_result(q4["question"], use_gemini=True)
    check("Gemini Q4 แบบกราฟ", "Journal of Cleaner Production" in gg["graph"]["answer"], gg["notice"] or gg["graph"]["answer"][:160])

print(f"\nผ่าน {sum(results)} จาก {len(results)} ข้อ")
