# ===== W33: Graph RAG เทียบการค้นด้วยเวกเตอร์อย่างเดียวกับการเดินกราฟ 2 ระดับ =====
# ต้องรันต่อจากช่อง W32 (ใช้ vector_search, rag_answer, call_gemini, build_prompt)
# กราฟสร้างจากตาราง 5NF ของทีมใน repo: PROCESS, PROCESS_LINK, PROCESS_SOURCE, SOURCE
import re
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

REPO_RAW = "https://raw.githubusercontent.com/we9d/Neexus4j/main/data/"


def load_csv(name):
    return pd.read_csv(REPO_RAW + name, encoding="utf-8-sig", dtype=str).fillna("")


process_df = load_csv("PROCESS.csv")
link_df = load_csv("PROCESS_LINK.csv")
psrc_df = load_csv("PROCESS_SOURCE.csv")
source_df = load_csv("SOURCE.csv")

graph = nx.MultiDiGraph()
for r in process_df.itertuples(index=False):
    graph.add_node(r.process_id, kind="process", name=r.process_name)
for r in source_df.itertuples(index=False):
    graph.add_node(r.source_id, kind="source", name=r.source_short, title=r.title, journal=r.journal)
# PROCESS_LINK: กระบวนการต้นน้ำ -ส่งผลผลิตไปยัง-> กระบวนการปลายน้ำ
for r in link_df.itertuples(index=False):
    graph.add_edge(r.from_process_id, r.to_process_id, rel="ส่งผลผลิตไปยัง", flow=r.flow_name,
                   table="PROCESS_LINK", evidence=f"{r.source_id} {r.source_location}")
# PROCESS_SOURCE: กระบวนการ -ใช้ข้อมูลบัญชีรายการจาก-> แหล่งข้อมูล (รวมตำแหน่งที่ซ้ำเป็นเส้นเดียว)
for (pid, sid), g in psrc_df.groupby(["process_id", "source_id"]):
    graph.add_edge(pid, sid, rel="ใช้ข้อมูลบัญชีรายการจาก", table="PROCESS_SOURCE",
                   evidence="; ".join(g["source_location"]))
print("จำนวนโหนด:", graph.number_of_nodes(), "| จำนวนเส้นเชื่อม:", graph.number_of_edges())


def pname(n):
    return graph.nodes[n].get("name", n)


def seeds_from_hits(hits):
    """โหนดตั้งต้น = รหัสกระบวนการ P-xx ที่อยู่ในข้อความของผลค้นด้วยเวกเตอร์"""
    seeds = []
    for h in hits:
        for pid in re.findall(r"P-\d{2}", h["content"]):
            if pid in graph and pid not in seeds:
                seeds.append(pid)
    return seeds


def walk_to_sources(seeds):
    """เดินกราฟไม่เกิน 2 ระดับจากโหนดตั้งต้นไปยังแหล่งข้อมูล
    1 ระดับ: กระบวนการ -> แหล่งข้อมูล
    2 ระดับ: กระบวนการ <-ส่งผลผลิต- กระบวนการต้นน้ำ -> แหล่งข้อมูล"""
    rows = []
    for seed in seeds:
        for _, sid, d in graph.out_edges(seed, data=True):
            if d["rel"] == "ใช้ข้อมูลบัญชีรายการจาก":
                rows.append({"seed": seed, "hops": 1, "upstream": "", "source_id": sid,
                             "journal": graph.nodes[sid].get("journal", ""),
                             "path": f"{seed} ({pname(seed)}) -ใช้ข้อมูลจาก-> {sid} [PROCESS_SOURCE {d['evidence']}]"})
        for up, _, d1 in graph.in_edges(seed, data=True):
            if d1["rel"] != "ส่งผลผลิตไปยัง":
                continue
            for _, sid, d2 in graph.out_edges(up, data=True):
                if d2["rel"] == "ใช้ข้อมูลบัญชีรายการจาก":
                    rows.append({"seed": seed, "hops": 2, "upstream": up, "source_id": sid,
                                 "journal": graph.nodes[sid].get("journal", ""),
                                 "path": (f"{up} ({pname(up)}) -ส่ง{d1['flow']}ไปยัง-> {seed} [PROCESS_LINK {d1['evidence']}] ; "
                                          f"{up} -ใช้ข้อมูลจาก-> {sid} [PROCESS_SOURCE {d2['evidence']}]")})
    return pd.DataFrame(rows, columns=["seed", "hops", "upstream", "source_id", "journal", "path"])


def graph_evidence(path_df):
    """แปลงเส้นทางเป็นบรรทัดหลักฐาน [G1], [G2], ... รวมชื่อวารสารจากตาราง SOURCE"""
    lines, seen = [], set()
    for r in path_df.itertuples(index=False):
        s = graph.nodes[r.source_id]
        key = (r.seed, r.upstream, r.source_id)
        if key in seen:
            continue
        seen.add(key)
        journal = f" ตีพิมพ์ในวารสาร {s['journal']}" if s.get("journal") else ""
        lines.append(f"[G{len(lines) + 1}] {r.path} ; {r.source_id} ({s['name']}) ชื่อเรื่อง {s['title']}{journal} [SOURCE]")
    return lines


def contains(text, name):
    return name.lower() in str(text).lower()


# TODO: ตรวจคำถามและคำตอบที่คาดไว้กับทีม
COMPARE_QUESTION = "ผลปาล์มทะลายสดที่ขนส่งไปโรงสกัด มาจากการปลูกที่อ้างอิงงานวิจัยซึ่งตีพิมพ์ในวารสารใด"
EXPECTED_NAME = "Journal of Cleaner Production"

print("=" * 70)
print("คำถามเดียวกัน:", COMPARE_QUESTION)
print("คำตอบที่คาดไว้:", EXPECTED_NAME)
print("=" * 70)

print("วิธีที่ 1: ค้นด้วยเวกเตอร์อย่างเดียว (top-5)")
vec = rag_answer(COMPARE_QUESTION)
vector_hits = vec["hits"]
vector_found_in_evidence = any(contains(h["content"], EXPECTED_NAME) for h in vector_hits)
print("พบชื่อวารสารที่คาดไว้ในหลักฐานจากเวกเตอร์:", "พบ" if vector_found_in_evidence else "ไม่พบ")

print("วิธีที่ 2: ค้นด้วยเวกเตอร์ แล้วเดินกราฟ 2 ระดับ")
seeds = seeds_from_hits(vector_hits)
print("โหนดตั้งต้นจากผลค้นหา:", ", ".join(seeds) if seeds else "-")
path_df = walk_to_sources(seeds)
print("เส้นทางที่เดินได้")
display(path_df)
g_lines = graph_evidence(path_df)
graph_found_in_evidence = any(contains(x, EXPECTED_NAME) for x in g_lines)
print("พบชื่อวารสารที่คาดไว้หลังเดินกราฟ:", "พบ" if graph_found_in_evidence else "ไม่พบ")

if not vector_hits or vector_hits[0]["score"] < SIM_THRESHOLD:
    graph_answer, graph_mode = NOT_FOUND, "below_threshold"
else:
    lines = evidence_from_hits(vector_hits) + g_lines
    generated = call_gemini(build_prompt(COMPARE_QUESTION, lines))
    if generated:
        graph_answer, graph_mode = generated, "llm"
    else:
        journals = sorted({j for j in path_df["journal"] if j})
        graph_answer = ("จากการเดินกราฟ พบวารสาร: " + ", ".join(journals)) if journals else NOT_FOUND
        graph_mode = "template"
print("วิธีได้คำตอบ:", graph_mode)
print("คำตอบจากเวกเตอร์ร่วมกับกราฟ:", graph_answer)

comparison = pd.DataFrame([
    {"method": "vector_only", "question": COMPARE_QUESTION,
     "found_expected_in_evidence": vector_found_in_evidence,
     "found_expected_in_answer": contains(vec["answer"], EXPECTED_NAME),
     "evidence": " | ".join(f"{h['chunk_id']}({h['source_id']}, {h['score']:.4f})" for h in vector_hits),
     "mode": vec["mode"], "answer": vec["answer"]},
    {"method": "vector_then_2hop_graph", "question": COMPARE_QUESTION,
     "found_expected_in_evidence": graph_found_in_evidence,
     "found_expected_in_answer": contains(graph_answer, EXPECTED_NAME),
     "evidence": " | ".join(g_lines),
     "mode": graph_mode, "answer": graph_answer},
])
print("-" * 70)
print("สรุปเปรียบเทียบสองวิธี")
display(comparison[["method", "found_expected_in_evidence", "found_expected_in_answer", "mode"]])
comparison.to_csv("graph_rag_comparison.csv", index=False, encoding="utf-8-sig")
path_df.to_csv("graph_rag_paths.csv", index=False, encoding="utf-8-sig")
print("บันทึก graph_rag_comparison.csv และ graph_rag_paths.csv แล้ว")

# ภาพเส้นทาง 2 ระดับ (ป้ายเป็นรหัสและภาษาอังกฤษ เพื่อไม่ให้ฟอนต์ไทยเป็นกล่องสี่เหลี่ยม)
if not path_df.empty:
    pg = nx.DiGraph()
    for r in path_df.itertuples(index=False):
        if r.upstream:
            pg.add_edge(r.upstream, r.seed, label="supplies FFB")
            pg.add_edge(r.upstream, r.source_id, label="uses data from")
        else:
            pg.add_edge(r.seed, r.source_id, label="uses data from")
    for sid in [n for n in pg if n in source_df["source_id"].values]:
        j = graph.nodes[sid].get("journal")
        if j:
            pg.add_edge(sid, j, label="published in")
    colors = ["#ffe0b2" if n in seeds else "#c8e6c9" if str(n).startswith("SRC") else
              "#bbdefb" if str(n).startswith("P-") else "#f8bbd0" for n in pg]
    suppliers = {u for u, v, d in pg.edges(data=True) if d["label"] == "supplies FFB"}
    for n in pg:
        pg.nodes[n]["layer"] = (0 if n in suppliers else 1 if str(n).startswith("P-") else
                                2 if str(n).startswith("SRC") else 3)
    fig, ax = plt.subplots(figsize=(13, 7))
    pos = nx.multipartite_layout(pg, subset_key="layer")
    nx.draw_networkx(pg, pos, ax=ax, node_color=colors, node_size=1800, font_size=8, arrowsize=14, edgecolors="#555")
    nx.draw_networkx_edge_labels(pg, pos, edge_labels=nx.get_edge_attributes(pg, "label"), ax=ax, font_size=7)
    ax.set_title("Graph RAG 2-hop: orange = process found by vector search, blue = process reached by graph walk, "
                 "green = source, pink = journal", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig("graph_rag_2hop_path.png", dpi=200)
    plt.show()
    print("บันทึก graph_rag_2hop_path.png แล้ว")
