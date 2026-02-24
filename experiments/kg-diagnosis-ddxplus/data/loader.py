"""DDXPlus dataset loader and vignette sampler."""
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# DDXPlus pathology to disease group mapping
PATHOLOGY_GROUPS = {
    # Cardiovascular
    "cardiovascular": [
        "Myocardial infarction", "Unstable angina", "Stable angina",
        "Atrial fibrillation", "Pulmonary embolism", "SVT",
        "Pericarditis", "Myocarditis", "Acute pulmonary edema",
        "Chagas", "SLE", "Sarcoidosis",
        "Anemia", "HIV (initial infection)", "Scombroid food poisoning",
        "Anaphylaxis", "Boerhaave",
    ],
    # Respiratory
    "respiratory": [
        "Pneumonia", "Bronchitis", "URTI", "Tuberculosis",
        "Bronchiectasis", "Croup", "Acute laryngitis",
        "Epiglottitis", "Influenza", "Whooping cough",
        "Bronchospasm / acute asthma exacerbation",
        "Possible NSTEMI / NSTEMI", "GERD", "Acute COPD exacerbation / infection",
        "Viral pharyngitis", "Allergic sinusitis",
        "Chronic rhinosinusitis", "Acute rhinosinusitis",
    ],
    # Neurological
    "neurological": [
        "Panic attack", "Cluster headache", "Spontaneous pneumothorax",
        "Guillain-Barré syndrome", "Acute dystonic reactions",
        "Inguinal hernia", "Spontaneous rib fracture",
        "Pancreatic neoplasm", "PSVT", "Larygospasm",
        "Localized edema", "Ebola", "Acute otitis media",
    ],
}

# Flatten for reverse lookup
PATHOLOGY_TO_GROUP = {}
for group, pathologies in PATHOLOGY_GROUPS.items():
    for p in pathologies:
        PATHOLOGY_TO_GROUP[p] = group

ALL_KNOWN_PATHOLOGIES = set()


def load_ddxplus(cache_dir: str = "cache/data") -> dict:
    """Load DDXPlus dataset from HuggingFace or cache."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    cached_file = cache_path / "ddxplus_all.parquet"
    if cached_file.exists():
        logger.info("Loading DDXPlus from cache...")
        df = pd.read_parquet(cached_file)
        return {"data": df}
    
    logger.info("Downloading DDXPlus dataset from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("auvipy/ddxplus", trust_remote_code=True)
        # Combine train/validate/test
        frames = []
        for split_name in ds:
            split_df = ds[split_name].to_pandas()
            split_df["_split"] = split_name
            frames.append(split_df)
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(cached_file)
        logger.info(f"Cached {len(df)} samples to {cached_file}")
        return {"data": df}
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def parse_evidences(evidences_str: str) -> Dict[str, str]:
    """Parse DDXPlus evidence string into dict."""
    if not evidences_str or pd.isna(evidences_str):
        return {}
    try:
        if isinstance(evidences_str, str):
            return json.loads(evidences_str.replace("'", '"'))
        return dict(evidences_str)
    except:
        return {}


def parse_differential(diff_str: str) -> List[Tuple[str, float]]:
    """Parse differential diagnosis string into list of (pathology, probability)."""
    if not diff_str or pd.isna(diff_str):
        return []
    try:
        if isinstance(diff_str, str):
            items = json.loads(diff_str.replace("'", '"'))
        else:
            items = list(diff_str)
        return [(item[0] if isinstance(item, (list, tuple)) else item, 
                 float(item[1]) if isinstance(item, (list, tuple)) and len(item) > 1 else 1.0)
                for item in items]
    except:
        return []


def build_vignette(row: pd.Series) -> Dict:
    """Convert a DDXPlus row into a standardized vignette dict."""
    evidences = parse_evidences(row.get("EVIDENCES", ""))
    differential = parse_differential(row.get("DIFFERENTIAL_DIAGNOSIS", ""))
    
    pathology = row.get("PATHOLOGY", "Unknown")
    age = row.get("AGE", "Unknown")
    sex = row.get("SEX", "Unknown")
    
    # Build symptom list from evidences
    symptoms = []
    antecedents = []
    for key, val in evidences.items():
        if key.startswith("@"):
            continue
        if val == "1" or val == 1 or val is True or val == "True":
            # Binary symptom present
            symptoms.append(key)
        elif isinstance(val, str) and val not in ("0", "False", "N"):
            symptoms.append(f"{key}: {val}")
    
    initial_evidence = row.get("INITIAL_EVIDENCE", "")
    
    return {
        "id": row.name if hasattr(row, 'name') else id(row),
        "age": age,
        "sex": sex,
        "initial_evidence": initial_evidence,
        "symptoms": symptoms,
        "antecedents": antecedents,
        "evidences_raw": evidences,
        "pathology": pathology,
        "differential": differential,
        "group": PATHOLOGY_TO_GROUP.get(pathology, "other"),
    }


def sample_vignettes(
    df: pd.DataFrame,
    n_per_group: int = 300,
    groups: List[str] = ["cardiovascular", "respiratory", "neurological"],
    seed: int = 42,
) -> List[Dict]:
    """Sample balanced vignettes across disease groups."""
    random.seed(seed)
    
    # Map pathologies to groups
    df = df.copy()
    df["_group"] = df["PATHOLOGY"].map(lambda p: PATHOLOGY_TO_GROUP.get(p, "other"))
    
    # Update global pathology set
    global ALL_KNOWN_PATHOLOGIES
    ALL_KNOWN_PATHOLOGIES = set(df["PATHOLOGY"].unique())
    
    vignettes = []
    for group in groups:
        group_df = df[df["_group"] == group]
        if len(group_df) == 0:
            logger.warning(f"No samples found for group '{group}'. Available pathologies: {df['PATHOLOGY'].unique()[:10]}")
            # Try fuzzy matching
            continue
        
        n_sample = min(n_per_group, len(group_df))
        sampled = group_df.sample(n=n_sample, random_state=seed)
        logger.info(f"Sampled {n_sample} vignettes for {group} from {len(group_df)} available")
        
        for _, row in sampled.iterrows():
            vignettes.append(build_vignette(row))
    
    logger.info(f"Total vignettes sampled: {len(vignettes)}")
    return vignettes


def split_vignettes(
    vignettes: List[Dict], test_ratio: float = 0.8, seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Split vignettes into test and validation sets."""
    random.seed(seed)
    shuffled = vignettes.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * test_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def get_all_pathologies(df: pd.DataFrame) -> set:
    """Return set of all pathology names in dataset."""
    return set(df["PATHOLOGY"].unique())


def get_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return training split for KG construction."""
    if "_split" in df.columns:
        train = df[df["_split"] == "train"]
        if len(train) > 0:
            return train
    # fallback: use first 70%
    return df.iloc[: int(len(df) * 0.7)]
