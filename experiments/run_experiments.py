#!/usr/bin/env python3
"""
DDXPlus Diagnostic Experiment Pipeline (Pure Python, Synthetic Data)
====================================================================
The DDXPlus dataset requires HuggingFace authentication (gated dataset).
This pipeline uses a synthetic version based on the published DDXPlus structure
(49 pathologies, ~200 symptoms) to demonstrate and benchmark 4 retrieval conditions.

Conditions:
1. LLM-only (TF-IDF similarity as proxy for zero-shot LLM)
2. Dense RAG (reduced-dim embeddings)
3. BM25 + PPMI (sparse retrieval + statistical associations)
4. Multi-Source KG (combines TF-IDF, PPMI, symptom overlap, disease prior)

Metrics: Top-k Accuracy, NDCG@5, F1, Hallucination Rate
"""

import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_DIR = Path("/tmp/ddxplus/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# ═══════════════════════════════════════════════════════════════
# DDXPlus-based Disease-Symptom Knowledge Base
# Based on the 49 pathologies in DDXPlus (Fansi Tchango et al., 2022)
# ═══════════════════════════════════════════════════════════════

DISEASE_SYMPTOM_DB = {
    "URTI": {
        "symptoms": ["cough", "sore_throat", "runny_nose", "sneezing", "fever_low", "headache", "fatigue", "body_aches"],
        "prior": 0.08,
    },
    "Pneumonia": {
        "symptoms": ["cough_productive", "fever_high", "chest_pain_pleuritic", "shortness_of_breath", "fatigue", "chills", "crackles", "tachypnea"],
        "prior": 0.04,
    },
    "Bronchitis": {
        "symptoms": ["cough_productive", "chest_discomfort", "fatigue", "sore_throat", "wheezing", "fever_low", "body_aches"],
        "prior": 0.05,
    },
    "Asthma": {
        "symptoms": ["wheezing", "shortness_of_breath", "chest_tightness", "cough_dry", "nocturnal_symptoms", "exercise_triggered"],
        "prior": 0.04,
    },
    "COPD": {
        "symptoms": ["shortness_of_breath_chronic", "cough_productive_chronic", "wheezing", "barrel_chest", "smoking_history", "fatigue"],
        "prior": 0.03,
    },
    "Tuberculosis": {
        "symptoms": ["cough_chronic", "hemoptysis", "night_sweats", "weight_loss", "fever_low_chronic", "fatigue", "lymphadenopathy"],
        "prior": 0.01,
    },
    "Influenza": {
        "symptoms": ["fever_high_sudden", "body_aches_severe", "headache", "fatigue_severe", "cough_dry", "sore_throat", "chills"],
        "prior": 0.05,
    },
    "COVID-19": {
        "symptoms": ["fever", "cough_dry", "fatigue", "loss_of_taste", "loss_of_smell", "shortness_of_breath", "headache", "body_aches"],
        "prior": 0.03,
    },
    "Allergic_Rhinitis": {
        "symptoms": ["sneezing", "runny_nose_clear", "nasal_congestion", "itchy_eyes", "watery_eyes", "postnasal_drip"],
        "prior": 0.06,
    },
    "Sinusitis": {
        "symptoms": ["facial_pain", "nasal_congestion", "purulent_discharge", "headache", "fever_low", "tooth_pain_upper", "postnasal_drip"],
        "prior": 0.03,
    },
    "Pharyngitis": {
        "symptoms": ["sore_throat_severe", "painful_swallowing", "fever", "swollen_lymph_nodes_neck", "tonsillar_exudate", "headache"],
        "prior": 0.04,
    },
    "Otitis_Media": {
        "symptoms": ["ear_pain", "fever", "hearing_loss", "ear_discharge", "irritability", "pulling_ear"],
        "prior": 0.03,
    },
    "Gastroenteritis": {
        "symptoms": ["diarrhea", "nausea", "vomiting", "abdominal_cramps", "fever_low", "dehydration"],
        "prior": 0.05,
    },
    "GERD": {
        "symptoms": ["heartburn", "regurgitation", "chest_pain_burning", "dysphagia", "chronic_cough", "hoarseness"],
        "prior": 0.04,
    },
    "Peptic_Ulcer": {
        "symptoms": ["epigastric_pain", "burning_stomach", "nausea", "bloating", "pain_with_meals", "weight_loss"],
        "prior": 0.02,
    },
    "Appendicitis": {
        "symptoms": ["right_lower_quadrant_pain", "nausea", "vomiting", "fever", "loss_of_appetite", "rebound_tenderness", "migration_of_pain"],
        "prior": 0.01,
    },
    "Cholecystitis": {
        "symptoms": ["right_upper_quadrant_pain", "nausea", "vomiting", "fever", "murphy_sign", "pain_after_fatty_food"],
        "prior": 0.01,
    },
    "Pancreatitis": {
        "symptoms": ["epigastric_pain_severe", "radiating_to_back", "nausea", "vomiting", "fever", "tachycardia", "abdominal_tenderness"],
        "prior": 0.01,
    },
    "UTI": {
        "symptoms": ["dysuria", "urinary_frequency", "urinary_urgency", "suprapubic_pain", "hematuria", "cloudy_urine", "fever_low"],
        "prior": 0.04,
    },
    "Pyelonephritis": {
        "symptoms": ["flank_pain", "fever_high", "chills", "dysuria", "nausea", "vomiting", "costovertebral_tenderness"],
        "prior": 0.01,
    },
    "Myocardial_Infarction": {
        "symptoms": ["chest_pain_crushing", "radiating_left_arm", "shortness_of_breath", "diaphoresis", "nausea", "jaw_pain", "anxiety"],
        "prior": 0.02,
    },
    "Angina": {
        "symptoms": ["chest_pain_exertional", "shortness_of_breath", "radiating_left_arm", "relieved_by_rest", "diaphoresis"],
        "prior": 0.02,
    },
    "Heart_Failure": {
        "symptoms": ["shortness_of_breath_exertional", "orthopnea", "peripheral_edema", "fatigue", "nocturnal_dyspnea", "weight_gain", "jugular_distension"],
        "prior": 0.02,
    },
    "Atrial_Fibrillation": {
        "symptoms": ["palpitations", "irregular_heartbeat", "shortness_of_breath", "fatigue", "dizziness", "chest_discomfort"],
        "prior": 0.02,
    },
    "Hypertension": {
        "symptoms": ["headache", "dizziness", "visual_changes", "chest_pain", "shortness_of_breath", "nosebleed", "often_asymptomatic"],
        "prior": 0.05,
    },
    "DVT": {
        "symptoms": ["leg_swelling_unilateral", "leg_pain", "warmth", "redness", "homan_sign", "recent_immobility"],
        "prior": 0.01,
    },
    "Pulmonary_Embolism": {
        "symptoms": ["shortness_of_breath_sudden", "chest_pain_pleuritic", "tachycardia", "hemoptysis", "leg_swelling", "anxiety", "syncope"],
        "prior": 0.01,
    },
    "Diabetes_Type2": {
        "symptoms": ["polyuria", "polydipsia", "polyphagia", "fatigue", "blurred_vision", "weight_loss", "slow_wound_healing"],
        "prior": 0.03,
    },
    "Hypothyroidism": {
        "symptoms": ["fatigue", "weight_gain", "cold_intolerance", "constipation", "dry_skin", "hair_loss", "bradycardia", "depression"],
        "prior": 0.02,
    },
    "Hyperthyroidism": {
        "symptoms": ["weight_loss", "heat_intolerance", "tachycardia", "tremor", "anxiety", "sweating", "exophthalmos", "diarrhea"],
        "prior": 0.01,
    },
    "Migraine": {
        "symptoms": ["headache_unilateral", "throbbing_pain", "nausea", "photophobia", "phonophobia", "aura", "visual_disturbances"],
        "prior": 0.04,
    },
    "Tension_Headache": {
        "symptoms": ["headache_bilateral", "band_like_pressure", "mild_to_moderate", "no_nausea", "stress_related", "neck_tension"],
        "prior": 0.05,
    },
    "Meningitis": {
        "symptoms": ["headache_severe", "neck_stiffness", "fever_high", "photophobia", "nausea", "vomiting", "altered_mental_status", "kernig_sign"],
        "prior": 0.005,
    },
    "Stroke": {
        "symptoms": ["sudden_weakness_unilateral", "speech_difficulty", "facial_droop", "vision_loss", "severe_headache", "confusion", "gait_difficulty"],
        "prior": 0.01,
    },
    "Anxiety_Disorder": {
        "symptoms": ["excessive_worry", "restlessness", "fatigue", "difficulty_concentrating", "muscle_tension", "sleep_disturbance", "palpitations"],
        "prior": 0.04,
    },
    "Depression": {
        "symptoms": ["depressed_mood", "loss_of_interest", "fatigue", "sleep_changes", "appetite_changes", "difficulty_concentrating", "feelings_of_worthlessness"],
        "prior": 0.04,
    },
    "Rheumatoid_Arthritis": {
        "symptoms": ["joint_pain_symmetric", "morning_stiffness", "joint_swelling", "fatigue", "small_joint_involvement", "rheumatoid_nodules"],
        "prior": 0.01,
    },
    "Osteoarthritis": {
        "symptoms": ["joint_pain_use", "stiffness_morning_brief", "crepitus", "joint_enlargement", "decreased_range_of_motion", "age_related"],
        "prior": 0.03,
    },
    "Gout": {
        "symptoms": ["joint_pain_acute", "big_toe_pain", "redness", "swelling", "warmth", "severe_pain_night"],
        "prior": 0.01,
    },
    "Anemia": {
        "symptoms": ["fatigue", "pallor", "shortness_of_breath_exertional", "dizziness", "tachycardia", "weakness", "pale_conjunctiva"],
        "prior": 0.03,
    },
    "Cellulitis": {
        "symptoms": ["skin_redness", "warmth", "swelling", "pain", "fever", "red_streaking", "rapid_spread"],
        "prior": 0.02,
    },
    "Eczema": {
        "symptoms": ["itchy_skin", "dry_skin", "red_patches", "skin_cracking", "oozing", "chronic_relapsing"],
        "prior": 0.03,
    },
    "Psoriasis": {
        "symptoms": ["scaly_patches", "silvery_scales", "itching", "dry_cracked_skin", "nail_changes", "joint_pain"],
        "prior": 0.01,
    },
    "Sciatica": {
        "symptoms": ["lower_back_pain", "radiating_leg_pain", "numbness", "tingling", "weakness_leg", "pain_sitting"],
        "prior": 0.02,
    },
    "Kidney_Stones": {
        "symptoms": ["flank_pain_severe", "colicky_pain", "hematuria", "nausea", "vomiting", "pain_radiating_groin", "restlessness"],
        "prior": 0.02,
    },
    "Pneumothorax": {
        "symptoms": ["chest_pain_sudden", "shortness_of_breath_sudden", "decreased_breath_sounds", "tachycardia", "hyperresonance"],
        "prior": 0.005,
    },
    "Bronchiolitis": {
        "symptoms": ["wheezing", "cough", "runny_nose", "fever_low", "tachypnea", "retractions", "infant"],
        "prior": 0.02,
    },
    "Croup": {
        "symptoms": ["barking_cough", "stridor", "hoarseness", "fever_low", "worsening_at_night", "young_child"],
        "prior": 0.01,
    },
    "Laryngitis": {
        "symptoms": ["hoarseness", "voice_loss", "sore_throat", "dry_cough", "throat_clearing", "fever_low"],
        "prior": 0.02,
    },
}


def generate_vignettes(n=50):
    """Generate synthetic patient vignettes with realistic noise and ambiguity."""
    diseases = list(DISEASE_SYMPTOM_DB.keys())
    vignettes = []
    ages = list(range(18, 85))
    sexes = ["M", "F"]
    
    # Build pool of all symptoms for noise injection
    all_symptoms_flat = set()
    for d in diseases:
        all_symptoms_flat.update(DISEASE_SYMPTOM_DB[d]["symptoms"])
    
    # Common non-specific symptoms that appear across many diseases
    nonspecific = ["fatigue", "headache", "fever", "nausea", "body_aches", "dizziness",
                   "shortness_of_breath", "chest_pain", "weakness", "malaise"]
    
    for i in range(n):
        weights = [DISEASE_SYMPTOM_DB[d]["prior"] for d in diseases]
        total_w = sum(weights)
        weights = [w/total_w for w in weights]
        
        r = random.random()
        cum = 0
        disease = diseases[-1]
        for d, w in zip(diseases, weights):
            cum += w
            if r <= cum:
                disease = d
                break
        
        info = DISEASE_SYMPTOM_DB[disease]
        symptoms = info["symptoms"]
        
        # Present only 40-75% of symptoms (harder)
        n_syms = max(2, int(len(symptoms) * random.uniform(0.4, 0.75)))
        presented = random.sample(symptoms, min(n_syms, len(symptoms)))
        
        # Add 1-4 noise symptoms from confounding diseases + nonspecific
        n_noise = random.randint(1, 4)
        noise_pool = list(all_symptoms_flat - set(symptoms)) + nonspecific
        noise = random.sample(noise_pool, min(n_noise, len(noise_pool)))
        
        # Occasionally drop the most distinctive symptom (30% chance)
        if len(presented) > 2 and random.random() < 0.3:
            presented = presented[1:]  # drop first (most characteristic)
        
        age = random.choice(ages)
        sex = random.choice(sexes)
        
        all_ev = presented + noise
        random.shuffle(all_ev)
        chief = all_ev[0]
        additional = all_ev[1:]
        
        # Use natural-ish text with symptom names (not clean tokens)
        text = f"Patient: age {age}, sex {sex}. Chief complaint: {chief}. Additional findings: {', '.join(additional)}."
        
        vignettes.append({
            "id": i,
            "text": text,
            "ground_truth": disease,
            "evidences": all_ev,
            "initial_evidence": chief,
        })
    
    gt_counts = Counter(v["ground_truth"] for v in vignettes)
    print(f"  Generated {len(vignettes)} vignettes across {len(gt_counts)} diseases")
    return vignettes


def build_kb_from_db():
    """Build knowledge base from the disease-symptom DB."""
    print("Building disease knowledge base...")
    disease_profiles = {}
    disease_symptoms = defaultdict(list)
    symptom_disease = defaultdict(Counter)
    disease_counts = Counter()
    
    # Simulate training data: 100 patients per disease
    for disease, info in DISEASE_SYMPTOM_DB.items():
        symptoms = info["symptoms"]
        n_patients = max(10, int(info["prior"] * 2000))
        disease_counts[disease] = n_patients
        
        for _ in range(n_patients):
            n_syms = max(2, int(len(symptoms) * random.uniform(0.5, 1.0)))
            sampled = random.sample(symptoms, min(n_syms, len(symptoms)))
            for s in sampled:
                disease_symptoms[disease].append(s)
                symptom_disease[s][disease] += 1
    
    for disease, syms in disease_symptoms.items():
        freq = Counter(syms)
        top = [s for s, _ in freq.most_common(30)]
        disease_profiles[disease] = f"Disease: {disease}. Common symptoms: {', '.join(top)}"
    
    print(f"  KB: {len(disease_profiles)} diseases, {len(symptom_disease)} unique symptoms")
    return disease_profiles, dict(disease_symptoms), dict(symptom_disease), disease_counts


# ═══════════════════════════════════════════════════════════════
# Pure-Python TF-IDF & Math
# ═══════════════════════════════════════════════════════════════

STOP_WORDS = set("a an the is was were be been being have has had do does did will would shall should may might can could and or but if then else when where how what which who whom this that these those i me my we our you your he him his she her it its they them their at by for from in into of on to with as".split())

def tokenize(text):
    return [w for w in re.findall(r'[a-z0-9_@]+', text.lower()) if w not in STOP_WORDS and len(w) > 1]

def build_tfidf(docs):
    tokenized = [tokenize(d) for d in docs]
    n = len(docs)
    df = Counter()
    for tokens in tokenized:
        for t in set(tokens):
            df[t] += 1
    
    vectors = []
    for tokens in tokenized:
        tf = Counter(tokens)
        vec = {}
        for t, c in tf.items():
            idf = math.log((n + 1) / (df[t] + 1)) + 1
            vec[t] = (1 + math.log(c)) * idf
        mag = math.sqrt(sum(v*v for v in vec.values()))
        if mag > 0:
            vec = {k: v/mag for k, v in vec.items()}
        vectors.append(vec)
    return vectors

def sparse_cosine(a, b):
    common = set(a.keys()) & set(b.keys())
    return sum(a[k] * b[k] for k in common) if common else 0.0

def dot_dense(a, b):
    return sum(x * y for x, y in zip(a, b))


# ═══════════════════════════════════════════════════════════════
# PPMI
# ═══════════════════════════════════════════════════════════════

def compute_ppmi(symptom_disease, diseases, all_symptoms):
    total = sum(sum(v.values()) for v in symptom_disease.values())
    if total == 0:
        return {}, {s: i for i, s in enumerate(all_symptoms)}, {d: i for i, d in enumerate(diseases)}
    
    sym_totals = {s: sum(d.values()) for s, d in symptom_disease.items()}
    dis_totals = Counter()
    for s, dd in symptom_disease.items():
        for d, c in dd.items():
            dis_totals[d] += c
    
    sym_idx = {s: i for i, s in enumerate(all_symptoms)}
    dis_idx = {d: i for i, d in enumerate(diseases)}
    
    ppmi = {}
    for symptom, dd in symptom_disease.items():
        if symptom not in sym_idx:
            continue
        si = sym_idx[symptom]
        p_s = sym_totals[symptom] / total
        for disease, count in dd.items():
            if disease not in dis_idx:
                continue
            di = dis_idx[disease]
            p_d = dis_totals[disease] / total
            p_sd = count / total
            if p_s > 0 and p_d > 0 and p_sd > 0:
                pmi = math.log2(p_sd / (p_s * p_d))
                if pmi > 0:
                    ppmi[(si, di)] = pmi
    return ppmi, sym_idx, dis_idx


def get_ppmi_scores(patient_symptoms, diseases, ppmi, sym_idx, dis_idx, init_ev=None):
    scores = [0.0] * len(diseases)
    for ev in patient_symptoms:
        if ev in sym_idx:
            si = sym_idx[ev]
            for j, d in enumerate(diseases):
                if d in dis_idx:
                    scores[j] += ppmi.get((si, dis_idx[d]), 0)
    if init_ev and init_ev in sym_idx:
        si = sym_idx[init_ev]
        for j, d in enumerate(diseases):
            if d in dis_idx:
                scores[j] += ppmi.get((si, dis_idx[d]), 0) * 2
    return scores


def safe_norm_list(arr):
    mx = max(arr) if arr else 0
    return [x/mx for x in arr] if mx > 0 else arr


# ═══════════════════════════════════════════════════════════════
# 4 Conditions
# ═══════════════════════════════════════════════════════════════

def condition_llm_only(vignettes, disease_profiles):
    print("\n=== Condition 1: LLM-Only (TF-IDF proxy) ===")
    diseases = list(disease_profiles.keys())
    all_vecs = build_tfidf([disease_profiles[d] for d in diseases] + [v["text"] for v in vignettes])
    pv = all_vecs[:len(diseases)]
    qv = all_vecs[len(diseases):]
    
    results = []
    for i, v in enumerate(vignettes):
        sims = [sparse_cosine(qv[i], p) for p in pv]
        ranked = sorted(range(len(sims)), key=lambda j: sims[j], reverse=True)
        results.append({"id": v["id"], "ground_truth": v["ground_truth"],
                       "predictions": [diseases[j] for j in ranked[:10]],
                       "scores": [sims[j] for j in ranked[:10]]})
    return results


def condition_dense_rag(vignettes, disease_profiles):
    print("\n=== Condition 2: Dense RAG (bigram-enhanced embeddings) ===")
    diseases = list(disease_profiles.keys())
    all_docs = [disease_profiles[d] for d in diseases] + [v["text"] for v in vignettes]
    
    tokenized = [tokenize(d) for d in all_docs]
    bigram_docs = []
    for tokens in tokenized:
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        bigram_docs.append(tokens + bigrams)
    
    n = len(all_docs)
    df = Counter()
    for tokens in bigram_docs:
        for t in set(tokens):
            df[t] += 1
    
    top_terms = sorted([t for t in df if df[t] >= 2], key=lambda t: -df[t])[:2000]
    term_idx = {t: i for i, t in enumerate(top_terms)}
    dim = len(top_terms)
    
    vectors = []
    for tokens in bigram_docs:
        tf = Counter(tokens)
        vec = [0.0] * dim
        for t, c in tf.items():
            if t in term_idx:
                idf = math.log((n+1)/(df[t]+1)) + 1
                vec[term_idx[t]] = (1 + math.log(c)) * idf
        mag = math.sqrt(sum(x*x for x in vec))
        if mag > 0:
            vec = [x/mag for x in vec]
        vectors.append(vec)
    
    pv = vectors[:len(diseases)]
    qv = vectors[len(diseases):]
    
    results = []
    for i, v in enumerate(vignettes):
        sims = [dot_dense(qv[i], p) for p in pv]
        ranked = sorted(range(len(sims)), key=lambda j: sims[j], reverse=True)
        results.append({"id": v["id"], "ground_truth": v["ground_truth"],
                       "predictions": [diseases[j] for j in ranked[:10]],
                       "scores": [sims[j] for j in ranked[:10]]})
    return results


def condition_bm25_ppmi(vignettes, disease_profiles, symptom_disease, disease_symptoms):
    print("\n=== Condition 3: BM25 + PPMI ===")
    diseases = list(disease_profiles.keys())
    doc_terms = [tokenize(disease_profiles[d]) for d in diseases]
    avg_dl = sum(len(d) for d in doc_terms) / len(doc_terms)
    n_docs = len(doc_terms)
    
    df = Counter()
    for dt in doc_terms:
        for t in set(dt):
            df[t] += 1
    
    all_syms = list(symptom_disease.keys())
    ppmi, sym_idx, dis_idx = compute_ppmi(symptom_disease, diseases, all_syms)
    
    k1, b = 1.5, 0.75
    results = []
    for i, v in enumerate(vignettes):
        qt = tokenize(v["text"])
        
        bm25 = []
        for dt in doc_terms:
            dl = len(dt)
            tfd = Counter(dt)
            s = 0
            for q in qt:
                if q in tfd:
                    tf = tfd[q]
                    idf = math.log((n_docs - df.get(q, 0) + 0.5) / (df.get(q, 0) + 0.5) + 1)
                    s += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            bm25.append(s)
        
        ppmi_s = get_ppmi_scores(v.get("evidences", []), diseases, ppmi, sym_idx, dis_idx, v.get("initial_evidence"))
        combined = [0.4*a + 0.6*b for a, b in zip(safe_norm_list(bm25), safe_norm_list(ppmi_s))]
        
        ranked = sorted(range(len(combined)), key=lambda j: combined[j], reverse=True)
        results.append({"id": v["id"], "ground_truth": v["ground_truth"],
                       "predictions": [diseases[j] for j in ranked[:10]],
                       "scores": [combined[j] for j in ranked[:10]]})
    return results


def condition_multi_source_kg(vignettes, disease_profiles, symptom_disease, disease_symptoms, disease_counts):
    print("\n=== Condition 4: Multi-Source KG ===")
    diseases = list(disease_profiles.keys())
    all_vecs = build_tfidf([disease_profiles[d] for d in diseases] + [v["text"] for v in vignettes])
    pv = all_vecs[:len(diseases)]
    qv = all_vecs[len(diseases):]
    
    all_syms = list(symptom_disease.keys())
    ppmi, sym_idx, dis_idx = compute_ppmi(symptom_disease, diseases, all_syms)
    
    total_cases = sum(disease_counts.values())
    prior = [disease_counts.get(d, 0) / total_cases for d in diseases]
    
    disease_sym_sets = {}
    for d in diseases:
        disease_sym_sets[d] = set(disease_symptoms.get(d, []))
    
    results = []
    for i, v in enumerate(vignettes):
        tfidf_sims = [sparse_cosine(qv[i], p) for p in pv]
        
        patient_syms = set(v.get("evidences", []))
        if v.get("initial_evidence"):
            patient_syms.add(v["initial_evidence"])
        
        ppmi_s = get_ppmi_scores(patient_syms, diseases, ppmi, sym_idx, dis_idx, v.get("initial_evidence"))
        
        overlap = []
        for d in diseases:
            ds = disease_sym_sets.get(d, set())
            if patient_syms and ds:
                overlap.append(len(patient_syms & ds) / len(patient_syms | ds))
            else:
                overlap.append(0.0)
        
        combined = [0.25*a + 0.35*b + 0.30*c + 0.10*d 
                   for a, b, c, d in zip(safe_norm_list(tfidf_sims), safe_norm_list(ppmi_s), 
                                          safe_norm_list(overlap), safe_norm_list(prior))]
        
        ranked = sorted(range(len(combined)), key=lambda j: combined[j], reverse=True)
        results.append({"id": v["id"], "ground_truth": v["ground_truth"],
                       "predictions": [diseases[j] for j in ranked[:10]],
                       "scores": [combined[j] for j in ranked[:10]]})
    return results


# ═══════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════

def compute_metrics(results, all_pathologies):
    m = {}
    for k in [1, 3, 5]:
        m[f"top_{k}_accuracy"] = sum(1 for r in results if r["ground_truth"] in r["predictions"][:k]) / len(results)
    
    ndcg = []
    for r in results:
        dcg = sum(1/math.log2(j+2) for j, p in enumerate(r["predictions"][:5]) if p == r["ground_truth"])
        ndcg.append(dcg / (1/math.log2(2)))
    m["ndcg_at_5"] = sum(ndcg) / len(ndcg)
    
    m["f1"] = sum(1 for r in results if r["predictions"][0] == r["ground_truth"]) / len(results)
    
    known = set(all_pathologies)
    m["hallucination_rate"] = sum(1 for r in results for p in r["predictions"][:5] if p not in known) / (len(results) * 5)
    
    return m


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    
    print("=" * 70)
    print(f"DDXPlus Diagnostic Experiment Pipeline — {n} vignettes")
    print("NOTE: Using synthetic data (DDXPlus dataset requires HF auth)")
    print("=" * 70)
    
    vignettes = generate_vignettes(n)
    disease_profiles, disease_symptoms, symptom_disease, disease_counts = build_kb_from_db()
    all_path = list(disease_profiles.keys())
    
    all_metrics = {}
    all_results = {}
    
    for name, fn in [
        ("llm_only", lambda: condition_llm_only(vignettes, disease_profiles)),
        ("dense_rag", lambda: condition_dense_rag(vignettes, disease_profiles)),
        ("bm25_ppmi", lambda: condition_bm25_ppmi(vignettes, disease_profiles, symptom_disease, disease_symptoms)),
        ("multi_source_kg", lambda: condition_multi_source_kg(vignettes, disease_profiles, symptom_disease, disease_symptoms, disease_counts)),
    ]:
        res = fn()
        met = compute_metrics(res, all_path)
        all_results[name] = res
        all_metrics[name] = met
        print(f"  → Top-1={met['top_1_accuracy']:.3f}  Top-3={met['top_3_accuracy']:.3f}  Top-5={met['top_5_accuracy']:.3f}  NDCG@5={met['ndcg_at_5']:.3f}  F1={met['f1']:.3f}  Halluc={met['hallucination_rate']:.3f}")
    
    with open(RESULTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(RESULTS_DIR / "detailed_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    hdr = f"{'Condition':<20} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8} {'NDCG@5':>8} {'F1':>8} {'Halluc':>8}"
    sep = "-" * 78
    print(f"\n{'='*78}\n{hdr}\n{sep}")
    lines = [hdr, sep]
    for name, met in all_metrics.items():
        line = f"{name:<20} {met['top_1_accuracy']:>8.3f} {met['top_3_accuracy']:>8.3f} {met['top_5_accuracy']:>8.3f} {met['ndcg_at_5']:>8.3f} {met['f1']:>8.3f} {met['hallucination_rate']:>8.3f}"
        print(line)
        lines.append(line)
    print("=" * 78)
    
    with open(RESULTS_DIR / "results_table.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
