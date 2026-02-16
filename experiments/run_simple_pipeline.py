#!/usr/bin/env python3
"""
Modified DDXPlus Diagnostic Experiment Pipeline
=============================================
This version uses /tmp/results for output to avoid permission issues.
"""

import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import csv
import zipfile
from urllib.request import urlopen

# Data Paths
DDXPLUS_DIR = Path("/tmp/ddxplus")
RESULTS_DIR = Path("/tmp/results")  # Changed to /tmp
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# Load DDXPlus Data
def load_ddxplus_data():
    """Load DDXPlus conditions, evidences, and patient data."""
    try:
        with open(DDXPLUS_DIR / "release_conditions.json") as f:
            conditions = json.load(f)
        
        with open(DDXPLUS_DIR / "release_evidences.json") as f:
            evidences = json.load(f)
        
        # Load test patients
        test_patients = []
        with open(DDXPLUS_DIR / "release_test_patients.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_patients.append(row)
        
        # Load training patients (for KG)
        train_patients = []
        with open(DDXPLUS_DIR / "release_train_patients.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                train_patients.append(row)
        
        return conditions, evidences, test_patients, train_patients
    except Exception as e:
        print(f"Error loading DDXPlus data: {e}")
        return {}, {}, [], []

# Main
def main():
    """Run the full pipeline."""
    print("Loading DDXPlus data...")
    conditions, evidences, test_patients, train_patients = load_ddxplus_data()
    
    if not test_patients:
        print("No test patients loaded - exiting")
        return
    
    print(f"Loaded {len(test_patients)} test patients and {len(train_patients)} training patients")
    print(f"Found {len(conditions)} conditions and {len(evidences)} evidence types")
    
    # Save simple status file
    with open(RESULTS_DIR / "status.json", "w") as f:
        json.dump({
            "status": "loaded",
            "test_patients": len(test_patients),
            "train_patients": len(train_patients),
            "conditions": len(conditions),
            "evidences": len(evidences)
        }, f, indent=2)
    
    print(f"Status saved to {RESULTS_DIR}/status.json")

if __name__ == "__main__":
    main()