# Nexus4j: Palm LCI ฐานข้อมูล 5NF, ฐานข้อมูลเวกเตอร์ และ RAG

รายวิชา CN230 กลุ่ม S02-G07 โครงงานฐานข้อมูลบัญชีรายการวัฏจักรชีวิต (LCI) ของน้ำมันปาล์มดิบ

## โครงสร้าง repository

```
Nexus4j/
├── data/            ข้อมูลตาราง 5NF ทั้ง 29 ตาราง (CSV)
├── web/             เว็บต้นแบบ Palm LCI Portal (Soft UI) ดูวิธีรันใน web/README.txt
│   ├── app.py
│   ├── data/        ข้อมูลที่เว็บใช้ (สำเนาบางตารางจาก data/, eval_results, knowledge_chunks)
│   ├── static/      หน้าเว็บ ฟอนต์ และภาพ (สถาปัตยกรรม, Heatmap, PCA)
│   └── tests/       selftest.py และ modelcheck.py
├── notebooks/       โค้ด Colab ของบท 6.5 (W32, W33, W35, W36)
├── results/         ผลการทดลองบท 6.5 ทั้งไฟล์ CSV, MD และภาพ
└── docs/            คู่มือวิธีรันเว็บ และคู่มือทีม (PDF)
```

## บท 6.5 ฐานข้อมูลเวกเตอร์ การค้นหาเชิงความหมาย และ Graph RAG

- ข้อมูล: Chunk 40 รายการ (KC001 ถึง KC040) จากตาราง LCI_EXCHANGE, SOURCE_ITEM, PROCESS และ SOURCE
- ฐานข้อมูลเวกเตอร์: Supabase PostgreSQL 17 + pgvector ตาราง `public.knowledge_chunks`
- โมเดลเวกเตอร์: `paraphrase-multilingual-MiniLM-L12-v2` (384 มิติ)
- โมเดลสร้างคำตอบ: Gemini `gemini-3.5-flash` (เว็บสำรองด้วย `gemini-3.5-flash-lite`)
- กราฟ: NetworkX จาก `PROCESS.csv`, `PROCESS_LINK.csv`, `PROCESS_SOURCE.csv`, `SOURCE.csv` ในโฟลเดอร์ `data/`

### DDL ตาราง knowledge_chunks (ตรงกับ Supabase)

```sql
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;  -- รุ่น 0.8.2

CREATE TABLE IF NOT EXISTS public.knowledge_chunks (
    chunk_id    VARCHAR(50)  PRIMARY KEY,
    content     TEXT         NOT NULL,
    source_id   VARCHAR(50)  NOT NULL,
    embedding   extensions.vector(384) NOT NULL,
    model_name  TEXT         NOT NULL,
    updated_at  TIMESTAMPTZ  DEFAULT CURRENT_TIMESTAMP
);
```

ดัชนีมีเพียงคีย์หลัก `knowledge_chunks_pkey` (B-tree) ไม่สร้างดัชนี HNSW หรือ IVFFlat เพราะมี 40 แถว ค้นครบทุกแถวด้วย `<=>`

### โค้ด Colab

| ไฟล์โค้ด | งาน | ขั้นตอนในสมุดบันทึก |
|---|---|---|
| `notebooks/w32_rag_answer.py` | W32 ค้นคืน 5 อันดับแล้วตอบจากหลักฐาน | 18, 19 |
| `notebooks/w33_graph_rag.py` | W33 Graph RAG เดินกราฟ 2 ระดับ | 20 |
| `notebooks/w35_eval.py` | W35 ประเมิน 5 คำถาม คะแนน 0 ถึง 3 และ Precision@5 | 21 |
| `notebooks/w36_upsert_change.py` | W36 เปลี่ยนค่า สร้างเวกเตอร์ใหม่ UPSERT แล้วถามซ้ำ | 22 |

โค้ดในโฟลเดอร์ `notebooks/` ต้องรันต่อจากขั้นตอนที่ 1 ถึง 9 ของสมุดบันทึกใน runtime เดียวกัน

## ผลการทดลองสรุป

- คะแนนเฉลี่ย 3.00 จาก 3 และ Precision@5 เฉลี่ย 0.20 (ไม่รวมข้อที่ต้องปฏิเสธการตอบ)
- คำถาม Multi-hop: ค้นด้วยเวกเตอร์อย่างเดียวตอบว่าไม่พบ เมื่อเดินกราฟตอบได้ว่า Journal of Cleaner Production
- UPSERT: คำตอบเปลี่ยนจาก 5.95 kg เป็น 6.20 kg (ค่าสมมติ) จำนวนแถวคงที่ 40 แถว และคืนค่าเดิมแล้ว

## ข้อควรระวัง

- ห้าม commit คีย์ Gemini รหัสผ่าน Supabase หรือไฟล์ `.env` ใช้ Colab Secrets หรือ `export` ใน Terminal เท่านั้น
- `notebooks/w33_graph_rag.py` อ่าน CSV 5NF จากโฟลเดอร์ `data/` ของ repo นี้ผ่าน GitHub หากย้ายไฟล์ต้องแก้ `REPO_RAW` ในไฟล์ด้วย
- ข้อมูลทั้งหมดเป็นข้อมูลจากงานวิจัยที่เผยแพร่แล้ว ไม่มีข้อมูลส่วนบุคคล ค่า 6.20 kg ในการทดลอง UPSERT เป็นค่าสมมติ
