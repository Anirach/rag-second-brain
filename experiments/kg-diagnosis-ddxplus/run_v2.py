#!/usr/bin/env python3
"""
V2: Fixed S2D (proper KG) + GPT-4o ablation on DDXPlus
"""
import os, sys, json, csv, math, time, hashlib, random, logging, re
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
import numpy as np

from openai import OpenAI
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

EMBEDDING_MODEL = "text-embedding-3-small"
API_DELAY = 0.1
MAX_RETRIES = 3
TOP_K = 5

BASE_DIR = Path("/home/clawdbot/clawd/experiments/kg-diagnosis-ddxplus")
DATA_DIR = BASE_DIR / "data" / "ddxplus"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = Path("/tmp/llm_cache_v2")
for d in [RESULTS_DIR, CACHE_DIR]: d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v2")
fh = logging.FileHandler(RESULTS_DIR / "experiment_v2_log.txt", mode='w')
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log.addHandler(fh)

client = OpenAI()

# ============================================================
# Cache
# ============================================================
def cache_key(prompt, model):
    return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()

def cached_llm_call(prompt, model="gpt-4o-mini"):
    ck = cache_key(prompt, model)
    cf = CACHE_DIR / f"{ck}.json"
    if cf.exists():
        return json.loads(cf.read_text())["response"]
    # Also check old cache
    old = Path("/tmp/llm_cache") / f"{ck}.json"
    if old.exists():
        data = json.loads(old.read_text())
        cf.write_text(old.read_text())
        return data["response"]
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(API_DELAY)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=500,
            )
            text = resp.choices[0].message.content.strip()
            cf.write_text(json.dumps({"prompt": prompt[:500], "model": model, "response": text}))
            return text
        except Exception as e:
            log.warning(f"API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return ""

def cached_embedding(text, model=EMBEDDING_MODEL):
    ck = cache_key(text, model)
    cf = CACHE_DIR / f"emb_{ck}.json"
    if cf.exists():
        return json.loads(cf.read_text())["embedding"]
    old = Path("/tmp/llm_cache") / f"emb_{ck}.json"
    if old.exists():
        data = json.loads(old.read_text())
        cf.write_text(old.read_text())
        return data["embedding"]
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(API_DELAY)
            resp = client.embeddings.create(model=model, input=text)
            emb = resp.data[0].embedding
            cf.write_text(json.dumps({"text": text[:200], "model": model, "embedding": emb}))
            return emb
        except Exception as e:
            log.warning(f"Embedding error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None

# ============================================================
# Helpers
# ============================================================
def norm(name):
    return re.sub(r'[^a-z0-9\s]', '', name.lower()).strip()

def match(pred, true):
    p, t = norm(pred), norm(true)
    if p == t: return True
    if len(p) > 3 and len(t) > 3 and (p in t or t in p): return True
    pw, tw = set(p.split()), set(t.split())
    if pw and tw and len(pw & tw) / max(len(pw), len(tw)) >= 0.6: return True
    return False

def parse_preds(text):
    try:
        m = re.search(r'\[.*?\]', text, re.DOTALL)
        if m:
            arr = json.loads(m.group())
            return [str(d).strip() for d in arr if d][:5]
    except: pass
    lines = text.strip().split('\n')
    out = []
    for l in lines:
        l = re.sub(r'^\d+[\.\)]\s*', '', l.strip())
        l = re.sub(r'[\*\-]+\s*', '', l).strip()
        if l and len(l) > 2: out.append(l)
    return out[:5]

def calc_metrics(results, known_diseases):
    t1 = t3 = t5 = 0
    ndcgs, f1s = [], []
    hall = total_p = 0
    disease_vocab = set(norm(d) for d in known_diseases)
    for r in results:
        preds, true = r["predictions"], r["true_diagnosis"]
        ms = [match(p, true) for p in preds[:5]]
        if any(ms[:1]): t1 += 1
        if any(ms[:3]): t3 += 1
        if any(ms[:5]): t5 += 1
        dcg = sum(1.0/math.log2(i+2) for i, m in enumerate(ms[:5]) if m)
        ndcgs.append(dcg / (1.0/math.log2(2)))
        tp = 1 if any(ms[:5]) else 0
        fp = sum(1 for m in ms[:5] if not m)
        prec = tp/(tp+fp) if tp+fp else 0
        rec = tp/(tp+(1-tp)) if tp+(1-tp) else 0
        f1s.append(2*prec*rec/(prec+rec) if prec+rec else 0)
        for p in preds:
            total_p += 1
            pn = norm(p)
            if not any(pn in dv or dv in pn for dv in disease_vocab):
                hall += 1
    n = len(results)
    return {
        "top1": round(t1/n, 4) if n else 0,
        "top3": round(t3/n, 4) if n else 0,
        "top5": round(t5/n, 4) if n else 0,
        "ndcg5": round(float(np.mean(ndcgs)), 4) if ndcgs else 0,
        "f1": round(float(np.mean(f1s)), 4) if f1s else 0,
        "halluc": round(hall/total_p, 4) if total_p else 0,
        "n": n,
    }

# ============================================================
# Build retrieval system for a specific domain
# ============================================================
class RetrievalSystem:
    def __init__(self, diseases, disease_descs, ppmi_data):
        self.diseases = diseases
        self.disease_descs = disease_descs
        self.ppmi = ppmi_data
        
        # Embeddings
        self.disease_embs = {}
        for d in diseases:
            emb = cached_embedding(disease_descs.get(d, d))
            if emb: self.disease_embs[d] = np.array(emb)
        
        # TF-IDF
        self.disease_list = list(diseases)
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.tfidf.fit_transform([disease_descs.get(d, d) for d in self.disease_list])
    
    def dense_retrieve(self, text, k=TOP_K):
        qe = cached_embedding(text)
        if qe is None: return []
        qe = np.array(qe)
        scores = []
        for d, e in self.disease_embs.items():
            sim = np.dot(qe, e) / (np.linalg.norm(qe) * np.linalg.norm(e) + 1e-8)
            scores.append((d, float(sim)))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]
    
    def bm25_retrieve(self, text, k=TOP_K):
        qv = self.tfidf.transform([text])
        sims = cosine_similarity(qv, self.tfidf_matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]
        return [(self.disease_list[i], float(sims[i])) for i in top_idx]
    
    def ppmi_rerank(self, syms, candidates, k=TOP_K):
        scores = {}
        for d, bs in candidates:
            ps = sum(self.ppmi.get(s, {}).get(d, 0) for s in syms)
            scores[d] = bs + 0.5 * ps
        return sorted(scores.items(), key=lambda x: -x[1])[:k]
    
    def multi_source(self, text, syms, k=TOP_K):
        dr = self.dense_retrieve(text, k * 2)
        br = self.bm25_retrieve(text, k * 2)
        ps = defaultdict(float)
        for s in syms:
            for d in self.ppmi.get(s, {}):
                ps[d] += self.ppmi[s][d]
        combined = defaultdict(float)
        for d, s in dr: combined[d] += 0.4 * s
        for d, s in br: combined[d] += 0.3 * s
        mx = max(ps.values()) if ps else 1
        for d, s in ps.items(): combined[d] += 0.3 * (s / mx)
        return sorted(combined.items(), key=lambda x: -x[1])[:k]

# ============================================================
# Conditions (parameterized by model and retrieval system)
# ============================================================
def run_conditions(vignettes, retrieval, model, name, extract_syms_fn):
    results = {"LLM-Only": [], "Dense RAG": [], "BM25+PPMI": [], "Multi-Source KG": []}
    
    for i, v in enumerate(vignettes):
        st = v["symptom_text"]
        syms = extract_syms_fn(v)
        true = v["true_diagnosis"]
        log.info(f"[{name}] {i+1}/{len(vignettes)} | {true}")
        
        # C1: LLM-Only
        p1 = f'Given these symptoms: {st}\n\nWhat are the top 5 most likely diagnoses? Return as a JSON array of strings, most likely first.\nExample: ["Diagnosis 1", "Diagnosis 2", "Diagnosis 3", "Diagnosis 4", "Diagnosis 5"]'
        r1 = cached_llm_call(p1, model)
        
        # C2: Dense RAG
        ret = retrieval.dense_retrieve(st)
        ctx = "\n".join([f"- {d} (sim: {s:.3f}): {retrieval.disease_descs.get(d, d)}" for d, s in ret])
        p2 = f'Given these symptoms: {st}\n\nRelevant disease information:\n{ctx}\n\nWhat are the top 5 most likely diagnoses? Return as a JSON array.\nExample: ["Diagnosis 1", "Diagnosis 2"]'
        r2 = cached_llm_call(p2, model)
        
        # C3: BM25+PPMI
        br = retrieval.bm25_retrieve(st, 10)
        rr = retrieval.ppmi_rerank(syms, br)
        ctx3 = "\n".join([f"- {d} (score: {s:.3f})" for d, s in rr])
        p3 = f'Given these symptoms: {st}\n\nStatistical co-occurrence evidence:\n{ctx3}\n\nWhat are the top 5 most likely diagnoses? Return as a JSON array.\nExample: ["Diagnosis 1", "Diagnosis 2"]'
        r3 = cached_llm_call(p3, model)
        
        # C4: Multi-Source KG
        fused = retrieval.multi_source(st, syms)
        ctx4 = "\n".join([f"- {d} (confidence: {s:.3f}): {retrieval.disease_descs.get(d, d)}" for d, s in fused])
        p4 = f'Given these symptoms: {st}\n\nMulti-source medical knowledge (embedding + statistical + KG):\n{ctx4}\n\nWhat are the top 5 most likely diagnoses? Return as a JSON array.\nExample: ["Diagnosis 1", "Diagnosis 2"]'
        r4 = cached_llm_call(p4, model)
        
        for cname, raw in [("LLM-Only", r1), ("Dense RAG", r2), ("BM25+PPMI", r3), ("Multi-Source KG", r4)]:
            preds = parse_preds(raw)
            results[cname].append({"id": v.get("id", i), "symptom_text": st,
                                   "true_diagnosis": true, "predictions": preds, "raw": raw})
    
    return results

# ============================================================
# PART 1: Load DDXPlus (reuse from v1)
# ============================================================
log.info("=" * 60)
log.info("Loading DDXPlus...")

with open(DATA_DIR / "release_evidences.json") as f:
    evidences = json.load(f)
with open(DATA_DIR / "release_conditions.json") as f:
    conditions = json.load(f)

cond_names = {k: v.get("cond-name-eng", v.get("condition_name", k)) for k, v in conditions.items()}
ddx_diseases = sorted(set(cond_names.values()))

def evidence_to_name(code):
    if code in evidences:
        return evidences[code].get("question_en", evidences[code].get("name", code))
    return code

def parse_evidences(evidence_str):
    symptoms = []
    if not evidence_str or evidence_str == "[]": return symptoms
    try:
        evidence_str = evidence_str.strip("[]")
        for item in evidence_str.split(","):
            item = item.strip().strip("'\"")
            if "_@_" in item:
                code, value = item.split("_@_", 1)
                if value.lower() in ("true", "1", "yes"):
                    symptoms.append(evidence_to_name(code))
                elif value.lower() not in ("false", "0", "no", "n"):
                    name = evidence_to_name(code)
                    symptoms.append(f"{name}: {value}")
            else:
                symptoms.append(evidence_to_name(item))
    except: pass
    return symptoms

patients = []
with open(DATA_DIR / "release_test_patients.csv", newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f): patients.append(row)
log.info(f"Loaded {len(patients)} DDXPlus test patients")

# Stratified sample
pathology_groups = defaultdict(list)
for p in patients: pathology_groups[p["PATHOLOGY"]].append(p)
sampled = []
n_per = max(1, 200 // len(pathology_groups))
for path in sorted(pathology_groups.keys()):
    grp = pathology_groups[path]
    random.shuffle(grp)
    sampled.extend(grp[:n_per])
if len(sampled) < 200:
    sampled_ids = set(id(p) for p in sampled)
    rest = [p for p in patients if id(p) not in sampled_ids]
    random.shuffle(rest)
    sampled.extend(rest[:200 - len(sampled)])
sampled = sampled[:200]
random.shuffle(sampled)

ddx_vignettes = []
for p in sampled:
    symptoms = parse_evidences(p.get("EVIDENCES", ""))
    true_dx = cond_names.get(p["PATHOLOGY"], p["PATHOLOGY"])
    parts = []
    if p.get("AGE"): parts.append(f"Age: {p['AGE']}")
    if p.get("SEX"): parts.append(f"Sex: {p['SEX']}")
    parts.extend(symptoms)
    ddx_vignettes.append({
        "id": len(ddx_vignettes),
        "symptoms": symptoms,
        "symptom_text": ", ".join(parts),
        "true_diagnosis": true_dx,
    })
log.info(f"DDXPlus: {len(ddx_vignettes)} vignettes")

# Build DDXPlus KG
kg_patients = patients[:10000]
ddx_sd = defaultdict(lambda: defaultdict(int))
ddx_s = defaultdict(int)
ddx_d = defaultdict(int)
N_ddx = 0
for p in kg_patients:
    syms = parse_evidences(p.get("EVIDENCES", ""))
    dis = cond_names.get(p["PATHOLOGY"], p["PATHOLOGY"])
    ddx_d[dis] += 1; N_ddx += 1
    for s in syms: ddx_s[s] += 1; ddx_sd[s][dis] += 1

ddx_ppmi = defaultdict(lambda: defaultdict(float))
for s in ddx_sd:
    for d in ddx_sd[s]:
        p_sd = ddx_sd[s][d] / N_ddx
        p_s = ddx_s[s] / N_ddx
        p_d = ddx_d[d] / N_ddx
        if p_s > 0 and p_d > 0 and p_sd > 0:
            val = math.log2(p_sd / (p_s * p_d))
            if val > 0: ddx_ppmi[s][d] = val

ddx_descs = {}
for dis in ddx_diseases:
    related = [(s, ddx_ppmi[s][dis]) for s in ddx_ppmi if ddx_ppmi[s].get(dis, 0) > 0]
    related.sort(key=lambda x: -x[1])
    tops = [s for s, _ in related[:15]]
    ddx_descs[dis] = f"{dis}: associated with {', '.join(tops)}" if tops else dis

log.info("Building DDXPlus retrieval system...")
ddx_retrieval = RetrievalSystem(ddx_diseases, ddx_descs, ddx_ppmi)

# ============================================================
# PART 2: S2D with PROPER KG
# ============================================================
log.info("=" * 60)
log.info("Loading Symptom2Disease with proper KG...")

s2d_all = []
for split in ["train", "test"]:
    try:
        ds = load_dataset("gretelai/symptom_to_diagnosis", split=split)
        for row in ds:
            text = row.get("text", row.get("input_text", ""))
            label = row.get("output_text", row.get("label", ""))
            s2d_all.append({"text": text, "label": label})
    except Exception as e:
        log.warning(f"S2D split {split}: {e}")

# Split: use 70% for KG building, 30% for testing
random.shuffle(s2d_all)
split_idx = int(len(s2d_all) * 0.7)
s2d_train = s2d_all[:split_idx]
s2d_test = s2d_all[split_idx:]

# Get S2D disease list
s2d_diseases = sorted(set(r["label"] for r in s2d_all))
log.info(f"S2D diseases: {len(s2d_diseases)}: {s2d_diseases}")

# Build S2D KG from training data
s2d_sd = defaultdict(lambda: defaultdict(int))
s2d_s = defaultdict(int)
s2d_d = defaultdict(int)
N_s2d = 0
for r in s2d_train:
    dis = r["label"]
    # Extract keywords from symptom text as pseudo-symptoms
    words = set(re.findall(r'\b[a-z]{3,}\b', r["text"].lower()))
    stop = {'the','and','has','have','had','been','was','are','for','with','that','this','from',
            'also','very','much','lot','lot','its','but','not','can','will','may','could','would',
            'about','into','when','what','some','more','them','they','their','your','been','being',
            'each','she','her','his','him','just','like','than','then','now','only','over','such',
            'after','before','other','well','back','even','still','last','long','going','feeling',
            'experiencing','days','day','weeks','time','started','noticed','body','doctor','symptoms'}
    words -= stop
    s2d_d[dis] += 1; N_s2d += 1
    for w in words:
        s2d_s[w] += 1
        s2d_sd[w][dis] += 1

s2d_ppmi = defaultdict(lambda: defaultdict(float))
for s in s2d_sd:
    for d in s2d_sd[s]:
        p_sd = s2d_sd[s][d] / N_s2d
        p_s = s2d_s[s] / N_s2d
        p_d = s2d_d[d] / N_s2d
        if p_s > 0 and p_d > 0 and p_sd > 0:
            val = math.log2(p_sd / (p_s * p_d))
            if val > 0: s2d_ppmi[s][d] = val

log.info(f"S2D PPMI: {len(s2d_ppmi)} terms × {len(s2d_d)} diseases")

# Build S2D disease descriptions from co-occurrence
s2d_descs = {}
for dis in s2d_diseases:
    related = [(s, s2d_ppmi[s][dis]) for s in s2d_ppmi if s2d_ppmi[s].get(dis, 0) > 0]
    related.sort(key=lambda x: -x[1])
    tops = [s for s, _ in related[:20]]
    s2d_descs[dis] = f"{dis}: commonly presents with {', '.join(tops)}" if tops else dis

log.info("Building S2D retrieval system...")
s2d_retrieval = RetrievalSystem(s2d_diseases, s2d_descs, s2d_ppmi)

# Prepare S2D test vignettes
s2d_vignettes = []
for r in s2d_test:
    s2d_vignettes.append({
        "id": len(s2d_vignettes),
        "symptom_text": r["text"],
        "true_diagnosis": r["label"],
    })
log.info(f"S2D test set: {len(s2d_vignettes)} vignettes")

# ============================================================
# PART 3: Run experiments
# ============================================================

# DDXPlus with gpt-4o-mini
log.info("=" * 60)
log.info("Running DDXPlus with gpt-4o-mini...")
ddx_mini = run_conditions(ddx_vignettes, ddx_retrieval, "gpt-4o-mini", "DDX-mini",
                          lambda v: v["symptoms"])
ddx_mini_met = {c: calc_metrics(r, ddx_diseases) for c, r in ddx_mini.items()}

# DDXPlus with gpt-4o (ablation)
log.info("=" * 60)
log.info("Running DDXPlus with gpt-4o (ablation)...")
ddx_4o = run_conditions(ddx_vignettes, ddx_retrieval, "gpt-4o", "DDX-4o",
                        lambda v: v["symptoms"])
ddx_4o_met = {c: calc_metrics(r, ddx_diseases) for c, r in ddx_4o.items()}

# S2D with gpt-4o-mini (fixed KG)
log.info("=" * 60)
log.info("Running S2D with gpt-4o-mini (fixed KG)...")
s2d_mini = run_conditions(s2d_vignettes, s2d_retrieval, "gpt-4o-mini", "S2D-mini",
                          lambda v: re.findall(r'\b[a-z]{3,}\b', v["symptom_text"].lower()))
s2d_mini_met = {c: calc_metrics(r, s2d_diseases) for c, r in s2d_mini.items()}

# ============================================================
# PART 4: Save results
# ============================================================
def fmt(v): return f"{v:.3f}" if isinstance(v, float) else str(v)

lines = [
    "# Experiment Results V2 — Publication Quality", "",
    f"**Date:** {datetime.now().isoformat()}", f"**Seed:** {SEED}", "",
    "## DDXPlus Dataset (n=200) — GPT-4o-mini", "",
    "| Method | Top-1 | Top-3 | Top-5 | NDCG@5 | F1 | Halluc. |",
    "|--------|-------|-------|-------|--------|-----|---------|",
]
for c in ["LLM-Only", "Dense RAG", "BM25+PPMI", "Multi-Source KG"]:
    m = ddx_mini_met[c]
    lines.append(f"| {c} | {fmt(m['top1'])} | {fmt(m['top3'])} | {fmt(m['top5'])} | {fmt(m['ndcg5'])} | {fmt(m['f1'])} | {fmt(m['halluc'])} |")

lines += ["", "## DDXPlus Dataset (n=200) — GPT-4o (Ablation)", "",
    "| Method | Top-1 | Top-3 | Top-5 | NDCG@5 | F1 | Halluc. |",
    "|--------|-------|-------|-------|--------|-----|---------|"]
for c in ["LLM-Only", "Dense RAG", "BM25+PPMI", "Multi-Source KG"]:
    m = ddx_4o_met[c]
    lines.append(f"| {c} | {fmt(m['top1'])} | {fmt(m['top3'])} | {fmt(m['top5'])} | {fmt(m['ndcg5'])} | {fmt(m['f1'])} | {fmt(m['halluc'])} |")

lines += ["", f"## Symptom2Disease Dataset (n={len(s2d_vignettes)}) — GPT-4o-mini (Fixed KG)", "",
    "| Method | Top-1 | Top-3 | Top-5 | NDCG@5 | F1 | Halluc. |",
    "|--------|-------|-------|-------|--------|-----|---------|"]
for c in ["LLM-Only", "Dense RAG", "BM25+PPMI", "Multi-Source KG"]:
    m = s2d_mini_met[c]
    lines.append(f"| {c} | {fmt(m['top1'])} | {fmt(m['top3'])} | {fmt(m['top5'])} | {fmt(m['ndcg5'])} | {fmt(m['f1'])} | {fmt(m['halluc'])} |")

summary = "\n".join(lines) + "\n"
with open(RESULTS_DIR / "summary_v2.md", "w") as f:
    f.write(summary)

all_results = {
    "ddxplus_mini": {"metrics": ddx_mini_met, "results": {c: [{"predictions": r["predictions"], "true": r["true_diagnosis"]} for r in rs] for c, rs in ddx_mini.items()}},
    "ddxplus_4o": {"metrics": ddx_4o_met, "results": {c: [{"predictions": r["predictions"], "true": r["true_diagnosis"]} for r in rs] for c, rs in ddx_4o.items()}},
    "s2d_mini": {"metrics": s2d_mini_met, "results": {c: [{"predictions": r["predictions"], "true": r["true_diagnosis"]} for r in rs] for c, rs in s2d_mini.items()}},
}
with open(RESULTS_DIR / "all_results_v2.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

log.info("EXPERIMENT V2 COMPLETE")
print("\n" + summary)
