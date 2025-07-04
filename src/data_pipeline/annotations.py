# In file: src/data_pipeline/annotations.py

import os
import re
from typing import Dict, List

def parse_chb_summary_txt(file_path: str) -> Dict[str, List[Dict[str, int]]]:
    """
    Parses a CHB-MIT summary.txt file to extract seizure times for all files listed within it.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    seizure_info = {}
    file_blocks = re.finditer(r"File Name: (chb\d{2,3}[a-zA-Z]?_\d{2,3}\.edf)[\s\S]*?(?=File Name:|$)", content, re.IGNORECASE)

    for block in file_blocks:
        file_name = block.group(1)
        seizures = []
        seizure_times = re.finditer(r"Seizure Start Time: (\d+) seconds\nSeizure End Time: (\d+) seconds", block.group(0), re.IGNORECASE)
        for seizure in seizure_times:
            seizures.append({'start': int(seizure.group(1)), 'end': int(seizure.group(2))})
        if seizures:
            seizure_info[file_name.lower()] = seizures # Use lowercase for consistent matching
            
    return seizure_info


def load_annotations_for_file(edf_file_path: str) -> List[Dict[str, int]]:
    """
    Master annotation loader for a single EDF file.
    This version now ONLY searches for a patient-level summary.txt file.
    """
    base_name_lower = os.path.basename(edf_file_path).lower()
    patient_dir = os.path.dirname(edf_file_path)
    patient_id = os.path.basename(patient_dir)

    # We only trust the summary.txt files.
    summary_file_path = os.path.join(patient_dir, f"{patient_id}-summary.txt")
    if os.path.exists(summary_file_path):
        all_patient_seizures = parse_chb_summary_txt(summary_file_path)
        return all_patient_seizures.get(base_name_lower, [])

    # Return empty list if no summary file is found
    return []