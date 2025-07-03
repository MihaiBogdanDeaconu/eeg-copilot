# In file: src/data_pipeline/annotations.py

import os
import re
from typing import Dict, List, Optional

def parse_chb_summary_txt(file_path: str) -> Dict[str, List[Dict[str, int]]]:
    """
    Parses a CHB-MIT summary.txt file to extract seizure times for all files listed within it.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    seizure_info = {}
    file_blocks = re.finditer(r"File Name: (chb\d{2}_\d{2,3}\.edf)[\s\S]*?(?=File Name:|$)", content)

    for block in file_blocks:
        file_name = block.group(1)
        seizures = []
        seizure_times = re.finditer(r"Seizure Start Time: (\d+) seconds\nSeizure End Time: (\d+) seconds", block.group(0))
        for seizure in seizure_times:
            seizures.append({'start': int(seizure.group(1)), 'end': int(seizure.group(2))})
        if seizures:
            seizure_info[file_name] = seizures
            
    return seizure_info

def parse_edf_seizures_file(file_path: str) -> List[Dict[str, int]]:
    """
    Parses a .edf.seizures file.
    Assumes a simple format of 'start_time_seconds end_time_seconds' per line.
    NOTE: This is a hypothesized format. Adjust if the actual format differs.
    """
    seizures = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                start_str, end_str = line.split()
                seizures.append({'start': int(start_str), 'end': int(end_str)})
            except ValueError:
                print(f"WARNING: Could not parse line in {file_path}: '{line}'")
    return seizures


def load_annotations_for_file(edf_file_path: str) -> List[Dict[str, int]]:
    """
    Master annotation loader for a single EDF file.
    It intelligently searches for corresponding annotation files (-summary.txt or .edf.seizures)
    and returns a standardized list of seizure intervals.
    
    Returns:
        A list of seizure dictionaries, e.g., [{'start': 2996, 'end': 3036}], or an empty list if no seizures.
    """
    base_name = os.path.basename(edf_file_path)
    patient_dir = os.path.dirname(edf_file_path)
    patient_id = os.path.basename(patient_dir)

    # Priority 1: Check for a .edf.seizures file specific to this EDF.
    seizure_file_path = edf_file_path + '.seizures'
    if os.path.exists(seizure_file_path):
        return parse_edf_seizures_file(seizure_file_path)

    # Priority 2: Check for a patient-level summary.txt file.
    summary_file_path = os.path.join(patient_dir, f"{patient_id}-summary.txt")
    if os.path.exists(summary_file_path):
        all_patient_seizures = parse_chb_summary_txt(summary_file_path)
        return all_patient_seizures.get(base_name, [])

    # Return empty list if no annotation file is found
    return []