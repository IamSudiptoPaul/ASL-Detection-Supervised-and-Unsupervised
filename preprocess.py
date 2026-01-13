# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical Assignment On Supervised And Unsupervised Learning Project
Coursework 002 for: CMP-7058A Artificial Intelligence

Feature Extraction

@author: C102 ( 100525654 , 100525448, 100538928 )
@date:   11/1/2026

"""
# Import Libraries
import csv
from collections import Counter

# Define Input And Output Files
INPUT_CSV  = "extracted_features.csv"
OUTPUT_CSV = "preprocessed_extracted_features.csv"

LABEL_COL = "Label"
EXPECTED_LANDMARKS = 21

# ===============================
# Helper functions
# ===============================
def parse_row_to_feats(row_dict):
    """Parse lm0..lm20 'x,y,z' strings into a 63D float list"""
    feats = []
    for i in range(EXPECTED_LANDMARKS):
        s = row_dict[f"lm{i}"].strip()
        parts = s.split(",")
        if len(parts) != 3:
            raise ValueError(f"Bad landmark format: {s}")
        feats.extend([float(parts[0]), float(parts[1]), float(parts[2])])
    return feats

def is_all_zero(feats):
    """No hand detected"""
    return all(abs(v) < 1e-12 for v in feats)

def is_landmark_valid(feats, z_abs_max=1.0):
    """Geometric sanity check"""
    for i in range(0, len(feats), 3):
        x, y, z = feats[i], feats[i+1], feats[i+2]
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False
        if abs(z) > z_abs_max:
            return False
    return True

# ===============================
# Cleaning + Deduplication
# ===============================
rows_clean = []
seen = set()          # for deduplication
reasons = Counter()   # for reporting

with open(INPUT_CSV, "r", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames

    # Basic header validation
    if LABEL_COL not in fieldnames:
        raise ValueError("Missing Label column.")
    for i in range(EXPECTED_LANDMARKS):
        if f"lm{i}" not in fieldnames:
            raise ValueError(f"Missing lm{i} column.")

    for row in reader:
        # ---- Parse features ----
        try:
            feats = parse_row_to_feats(row)
        except Exception:
            reasons["bad_format"] += 1
            continue

        # ---- Remove no-hand samples ----
        if is_all_zero(feats):
            reasons["all_zero_no_hand"] += 1
            continue

        # ---- Remove invalid landmarks ----
        if not is_landmark_valid(feats):
            reasons["invalid_coords"] += 1
            continue

        # ---- Deduplication (exact duplicate rows) ----
        key = tuple(row[col] for col in fieldnames)
        if key in seen:
            reasons["duplicate_sample"] += 1
            continue
        seen.add(key)

        rows_clean.append(row)

# ===============================
# Write cleaned CSV (same schema)
# ===============================
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows_clean:
        writer.writerow(r)

# ===============================
# Summary (for console / report)
# ===============================
print("Cleaning finished.")
print(f"Kept samples   : {len(rows_clean)}")
print("Removed samples:")
for k, v in reasons.items():
    print(f"  - {k}: {v}")
