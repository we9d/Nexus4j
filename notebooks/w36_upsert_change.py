# ===== W36: เปลี่ยนค่าในข้อมูลของทีม สร้าง embedding ใหม่ UPSERT แล้วถามซ้ำ =====
# ต้องรันต่อจากขั้นตอนที่ 3 (model, embeddings, df), 7 (conn) และ 19 (rag_answer)
# ค่าใหม่เป็นค่าสมมติเพื่อสาธิตเท่านั้น ไม่ใช่ข้อมูลจาก SRC-04 และไม่ได้แก้ตาราง 5NF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

CHANGE_ID = "KC022"
OLD_VALUE = "5.95 kg"
NEW_VALUE = "6.20 kg"            # ค่าสมมติเพื่อสาธิต
CHANGE_QUESTION = "การสกัดน้ำมันปาล์มดิบชุดหลักใช้ผลปาล์มทะลายสดกี่ kg ต่อน้ำมันปาล์มดิบ 1 kg"
RESTORE_AFTER = True             # คืนค่าเดิมหลังสาธิต เพื่อไม่ให้ค่าสมมติค้างในคลัง


def fetch_row(chunk_id):
    with conn.cursor() as cur:
        cur.execute("""SELECT content, source_id, embedding::text, model_name, updated_at
                       FROM public.knowledge_chunks WHERE chunk_id = %s""", (chunk_id,))
        return cur.fetchone()


def count_rows(chunk_id=None):
    with conn.cursor() as cur:
        if chunk_id:
            cur.execute("SELECT COUNT(*) FROM public.knowledge_chunks WHERE chunk_id = %s", (chunk_id,))
        else:
            cur.execute("SELECT COUNT(*) FROM public.knowledge_chunks")
        return cur.fetchone()[0]


def upsert_chunk(chunk_id, content, source_id, vector_text, model_name):
    with conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO public.knowledge_chunks (chunk_id, content, source_id, embedding, model_name)
                VALUES (%s, %s, %s, %s::extensions.vector, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    source_id = EXCLUDED.source_id,
                    embedding = EXCLUDED.embedding,
                    model_name = EXCLUDED.model_name,
                    updated_at = CURRENT_TIMESTAMP
            """, (chunk_id, content, source_id, vector_text, model_name))


# ขั้นที่ 1 ถามก่อนเปลี่ยน
print("ขั้นที่ 1 ถามก่อนเปลี่ยนค่า")
before = rag_answer(CHANGE_QUESTION)
rows_before = count_rows()

# ขั้นที่ 2 สำรองแถวเดิม แล้วแก้ค่าในข้อความ
old_text, old_source, old_vec_text, old_model, old_updated = fetch_row(CHANGE_ID)
assert OLD_VALUE in old_text, f"ไม่พบ '{OLD_VALUE}' ใน {CHANGE_ID} ตรวจค่าที่ตั้งไว้"
new_text = old_text.replace(OLD_VALUE, NEW_VALUE)
print("ขั้นที่ 2 แก้ค่าใน", CHANGE_ID, ":", OLD_VALUE, "->", NEW_VALUE, "(ค่าสมมติเพื่อสาธิต)")

# ขั้นที่ 3 สร้าง embedding ใหม่ด้วยโมเดลเดิม แล้ว UPSERT ทับ chunk_id เดิม
new_vec = model.encode([new_text], normalize_embeddings=True)[0]
old_vec = np.fromstring(old_vec_text.strip("[]"), sep=",")
upsert_chunk(CHANGE_ID, new_text, old_source, to_pgvector(new_vec), MODEL_NAME)
_, _, _, _, new_updated = fetch_row(CHANGE_ID)
rows_after = count_rows()
print("ขั้นที่ 3 UPSERT แล้ว | cosine(เวกเตอร์เดิม, ใหม่) =", round(float(old_vec @ new_vec), 4))
print(f"  จำนวนแถวในคลัง ก่อน {rows_before} หลัง {rows_after} | แถว {CHANGE_ID}: {count_rows(CHANGE_ID)}")
print(f"  updated_at ก่อน {old_updated} หลัง {new_updated}")
assert rows_after == rows_before and count_rows(CHANGE_ID) == 1, "UPSERT สร้างแถวซ้ำ"

# ขั้นที่ 4 ถามคำถามเดิมซ้ำ
print("ขั้นที่ 4 ถามคำถามเดิมซ้ำหลัง UPSERT")
after = rag_answer(CHANGE_QUESTION)

changed = before["answer"] != after["answer"]
uses_new = NEW_VALUE.split()[0] in after["answer"]
print("=" * 70)
print("คำตอบก่อนอัปเดต:", before["answer"])
print("คำตอบหลังอัปเดต:", after["answer"])
print("คำตอบเปลี่ยนหรือไม่:", "เปลี่ยน" if changed else "ยังเหมือนเดิม",
      "| คำตอบใหม่มีค่า", NEW_VALUE, ":", "มี" if uses_new else "ไม่มี")
print("อันดับหนึ่ง ก่อน:", before["hits"][0]["chunk_id"], before["hits"][0]["score"],
      "| หลัง:", after["hits"][0]["chunk_id"], after["hits"][0]["score"])

pd.DataFrame([
    {"item": f"ค่าใน {CHANGE_ID}", "before": OLD_VALUE, "after": f"{NEW_VALUE} (ค่าสมมติเพื่อสาธิต)"},
    {"item": "จำนวนแถวในคลัง", "before": rows_before, "after": rows_after},
    {"item": "อันดับหนึ่งและคะแนน", "before": f"{before['hits'][0]['chunk_id']} {before['hits'][0]['score']}",
     "after": f"{after['hits'][0]['chunk_id']} {after['hits'][0]['score']}"},
    {"item": "โหมดการตอบ", "before": before["mode"], "after": after["mode"]},
    {"item": "คำตอบ", "before": before["answer"], "after": after["answer"]},
]).to_csv("upsert_before_after.csv", index=False, encoding="utf-8-sig")
print("บันทึก upsert_before_after.csv แล้ว")

# ขั้นที่ 5 ภาพ PCA ตำแหน่งของ KC022 ก่อนและหลังเปลี่ยนค่า (fit กับ 40 เวกเตอร์เดิมจากขั้นตอนที่ 3)
idx = df.index[df["chunk_id"] == CHANGE_ID][0]
pca = PCA(n_components=2).fit(embeddings)
all_xy = pca.transform(embeddings)
b_xy, a_xy = pca.transform(np.vstack([old_vec, new_vec]))
q_xy = pca.transform(model.encode([CHANGE_QUESTION], normalize_embeddings=True))[0]
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(all_xy[:, 0], all_xy[:, 1], s=25, c="#bbbbbb", label="other chunks")
ax.scatter(*b_xy, s=120, c="#1f77b4", marker="o", label=f"{CHANGE_ID} before ({OLD_VALUE})")
ax.scatter(*a_xy, s=160, c="#d62728", marker="*", label=f"{CHANGE_ID} after ({NEW_VALUE}, hypothetical)")
ax.scatter(*q_xy, s=120, c="#2ca02c", marker="X", label="question")
ax.annotate("", xy=a_xy, xytext=b_xy, arrowprops=dict(arrowstyle="->", color="#555"))
ax.set_title(f"PCA: {CHANGE_ID} before and after UPSERT")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("pca_upsert_before_after.png", dpi=200); plt.show()
print("บันทึก pca_upsert_before_after.png แล้ว")

# ขั้นที่ 6 คืนค่าเดิม เพื่อไม่ให้ค่าสมมติค้างในคลังความรู้
if RESTORE_AFTER:
    upsert_chunk(CHANGE_ID, old_text, old_source, old_vec_text, old_model)
    assert fetch_row(CHANGE_ID)[0] == old_text and count_rows() == rows_before
    print("ขั้นที่ 6 คืนค่าเดิมของ", CHANGE_ID, "แล้ว | จำนวนแถว:", count_rows())
