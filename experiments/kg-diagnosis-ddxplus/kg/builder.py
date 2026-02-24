"""Knowledge Graph builder from DDXPlus training data."""
import json
import logging
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data.loader import parse_evidences, parse_differential

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """SQLite-backed medical knowledge graph."""

    def __init__(self, db_path: str = "cache/kg.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_tables()
        self._ppmi_matrix: Dict[Tuple[str, str], float] = {}
        self._disease_symptoms: Dict[str, List[str]] = {}
        self._symptom_diseases: Dict[str, List[str]] = {}

    def _init_tables(self):
        c = self.conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                type TEXT  -- 'symptom', 'disease', 'risk_factor', 'antecedent'
            );
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                target TEXT,
                relation TEXT,  -- 'symptom_of', 'risk_factor_for', 'co_occurs_with', 'differential_of'
                weight REAL DEFAULT 1.0
            );
            CREATE TABLE IF NOT EXISTS ppmi (
                symptom TEXT,
                disease TEXT,
                score REAL,
                PRIMARY KEY (symptom, disease)
            );
            CREATE INDEX IF NOT EXISTS idx_rel_source ON relations(source);
            CREATE INDEX IF NOT EXISTS idx_rel_target ON relations(target);
            CREATE INDEX IF NOT EXISTS idx_rel_relation ON relations(relation);
            CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(type);
        """)
        self.conn.commit()

    def build_from_dataframe(self, df: pd.DataFrame):
        """Build KG from DDXPlus training data."""
        logger.info(f"Building KG from {len(df)} training samples...")

        symptom_disease_counts = Counter()
        disease_counts = Counter()
        symptom_counts = Counter()
        differential_pairs = set()
        co_occurrence = Counter()
        total_samples = 0

        diseases = set()
        symptoms = set()

        for _, row in df.iterrows():
            pathology = row.get("PATHOLOGY", "")
            if not pathology:
                continue

            diseases.add(pathology)
            disease_counts[pathology] += 1
            total_samples += 1

            evidences = parse_evidences(row.get("EVIDENCES", ""))
            row_symptoms = []
            for key, val in evidences.items():
                if key.startswith("@"):
                    continue
                if val == "1" or val == 1 or val is True or val == "True":
                    symptoms.add(key)
                    row_symptoms.append(key)
                    symptom_counts[key] += 1
                    symptom_disease_counts[(key, pathology)] += 1
                elif isinstance(val, str) and val not in ("0", "False", "N", ""):
                    full_key = f"{key}_{val}"
                    symptoms.add(full_key)
                    row_symptoms.append(full_key)
                    symptom_counts[full_key] += 1
                    symptom_disease_counts[(full_key, pathology)] += 1

            # Co-occurrence between symptoms
            for i, s1 in enumerate(row_symptoms):
                for s2 in row_symptoms[i + 1:]:
                    pair = tuple(sorted([s1, s2]))
                    co_occurrence[pair] += 1

            # Differential diagnosis pairs
            diff = parse_differential(row.get("DIFFERENTIAL_DIAGNOSIS", ""))
            diff_diseases = [d[0] for d in diff]
            for i, d1 in enumerate(diff_diseases):
                for d2 in diff_diseases[i + 1:]:
                    differential_pairs.add(tuple(sorted([d1, d2])))

        # Insert entities
        c = self.conn.cursor()
        c.execute("DELETE FROM entities")
        c.execute("DELETE FROM relations")
        c.execute("DELETE FROM ppmi")

        for d in diseases:
            c.execute("INSERT OR IGNORE INTO entities (name, type) VALUES (?, 'disease')", (d,))
        for s in symptoms:
            c.execute("INSERT OR IGNORE INTO entities (name, type) VALUES (?, 'symptom')", (s,))

        # Insert symptom-of relations
        for (symptom, disease), count in symptom_disease_counts.items():
            weight = count / max(disease_counts[disease], 1)
            c.execute(
                "INSERT INTO relations (source, target, relation, weight) VALUES (?, ?, 'symptom_of', ?)",
                (symptom, disease, weight),
            )

        # Insert co-occurs-with
        for (s1, s2), count in co_occurrence.most_common(5000):
            if count >= 3:
                c.execute(
                    "INSERT INTO relations (source, target, relation, weight) VALUES (?, ?, 'co_occurs_with', ?)",
                    (s1, s2, count),
                )

        # Insert differential-of
        for d1, d2 in differential_pairs:
            c.execute(
                "INSERT INTO relations (source, target, relation, weight) VALUES (?, ?, 'differential_of', 1.0)",
                (d1, d2),
            )

        # Compute PPMI
        logger.info("Computing PPMI matrix...")
        for (symptom, disease), count in symptom_disease_counts.items():
            p_sd = count / total_samples
            p_s = symptom_counts[symptom] / total_samples
            p_d = disease_counts[disease] / total_samples
            pmi = math.log2(max(p_sd / (p_s * p_d), 1e-10))
            ppmi = max(pmi, 0)
            if ppmi > 0:
                c.execute(
                    "INSERT OR REPLACE INTO ppmi (symptom, disease, score) VALUES (?, ?, ?)",
                    (symptom, disease, ppmi),
                )
                self._ppmi_matrix[(symptom, disease)] = ppmi

        self.conn.commit()

        # Build in-memory indices
        self._build_indices()

        stats = self.get_statistics()
        logger.info(f"KG built: {stats}")
        return stats

    def _build_indices(self):
        """Build in-memory lookup indices."""
        c = self.conn.cursor()
        self._disease_symptoms = defaultdict(list)
        self._symptom_diseases = defaultdict(list)

        for row in c.execute(
            "SELECT source, target, weight FROM relations WHERE relation='symptom_of'"
        ):
            self._symptom_diseases[row[0]].append((row[1], row[2]))
            self._disease_symptoms[row[1]].append((row[0], row[2]))

        # Load PPMI into memory
        self._ppmi_matrix = {}
        for row in c.execute("SELECT symptom, disease, score FROM ppmi"):
            self._ppmi_matrix[(row[0], row[1])] = row[2]

    def get_diseases_for_symptom(self, symptom: str) -> List[Tuple[str, float]]:
        """Get diseases associated with a symptom, sorted by weight."""
        results = self._symptom_diseases.get(symptom, [])
        return sorted(results, key=lambda x: x[1], reverse=True)

    def get_symptoms_for_disease(self, disease: str) -> List[Tuple[str, float]]:
        """Get symptoms associated with a disease."""
        return self._disease_symptoms.get(disease, [])

    def get_ppmi(self, symptom: str, disease: str) -> float:
        """Get PPMI score for symptom-disease pair."""
        return self._ppmi_matrix.get((symptom, disease), 0.0)

    def get_differentials(self, disease: str) -> List[str]:
        """Get differential diagnoses for a disease."""
        c = self.conn.cursor()
        results = []
        for row in c.execute(
            "SELECT target FROM relations WHERE source=? AND relation='differential_of' "
            "UNION SELECT source FROM relations WHERE target=? AND relation='differential_of'",
            (disease, disease),
        ):
            results.append(row[0])
        return results

    def get_co_occurring_symptoms(self, symptom: str) -> List[Tuple[str, float]]:
        """Get symptoms that co-occur with given symptom."""
        c = self.conn.cursor()
        results = []
        for row in c.execute(
            "SELECT target, weight FROM relations WHERE source=? AND relation='co_occurs_with' "
            "UNION SELECT source, weight FROM relations WHERE target=? AND relation='co_occurs_with'",
            (symptom, symptom),
        ):
            results.append((row[0], row[1]))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def graph_traverse(self, seed_symptoms: List[str], max_hops: int = 2) -> Dict[str, float]:
        """Multi-hop graph traversal from seed symptoms to find candidate diseases."""
        disease_scores = defaultdict(float)

        for symptom in seed_symptoms:
            # Direct symptom -> disease
            for disease, weight in self.get_diseases_for_symptom(symptom):
                disease_scores[disease] += weight

            # 2-hop: symptom -> co-occurring symptom -> disease
            if max_hops >= 2:
                for co_symptom, co_weight in self.get_co_occurring_symptoms(symptom)[:10]:
                    for disease, weight in self.get_diseases_for_symptom(co_symptom):
                        disease_scores[disease] += weight * 0.5 * (co_weight / 100)

        # Add differential connections
        top_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        for disease, score in top_diseases:
            for diff in self.get_differentials(disease):
                disease_scores[diff] += score * 0.3

        return dict(disease_scores)

    def get_all_diseases(self) -> List[str]:
        c = self.conn.cursor()
        return [r[0] for r in c.execute("SELECT name FROM entities WHERE type='disease'")]

    def get_statistics(self) -> Dict:
        c = self.conn.cursor()
        entity_count = c.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        relation_count = c.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        disease_count = c.execute("SELECT COUNT(*) FROM entities WHERE type='disease'").fetchone()[0]
        symptom_count = c.execute("SELECT COUNT(*) FROM entities WHERE type='symptom'").fetchone()[0]
        ppmi_count = c.execute("SELECT COUNT(*) FROM ppmi").fetchone()[0]

        return {
            "entities": entity_count,
            "relations": relation_count,
            "diseases": disease_count,
            "symptoms": symptom_count,
            "ppmi_entries": ppmi_count,
            "density": relation_count / max(entity_count * (entity_count - 1) / 2, 1),
        }

    def close(self):
        self.conn.close()
