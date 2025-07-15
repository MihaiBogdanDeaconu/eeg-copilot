import os
import re
from typing import Dict, List

def parse_summary_file(file_path: str) -> Dict[str, List[Dict[str, int]]]:
    """
    Parses a TUH or CHB-MIT summary file (.txt, .tse) to extract seizure times.
    """
    seizure_info = {}
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # This regex is robust for both TUH and CHB-MIT formats
        file_blocks = re.finditer(r"File Name: ([\w\d_\-\.]+\.edf)[\s\S]*?(?=File Name:|$)", content, re.IGNORECASE)

        for block in file_blocks:
            file_name = block.group(1)
            seizures = []
            # This regex captures "Seizure Start Time", "Seizure 1 Start Time", etc.
            seizure_times = re.finditer(r"Seizure(?: \w+)? Start Time: ([\d\.]+) seconds\nSeizure(?: \w+)? End Time: ([\d\.]+) seconds", block.group(0), re.IGNORECASE)
            for seizure in seizure_times:
                try:
                    start_time = float(seizure.group(1))
                    end_time = float(seizure.group(2))
                    seizures.append({'start': int(start_time), 'end': int(end_time)})
                except ValueError:
                    continue # Skip if times are not valid numbers
            if seizures:
                seizure_info[file_name.lower()] = seizures
    except Exception as e:
        print(f"Warning: Could not read or parse summary file {file_path}. Error: {e}")
            
    return seizure_info

def load_annotations_for_file(edf_file_path: str) -> List[Dict[str, int]]:
    """
    Master annotation loader. It looks for a corresponding summary/annotation file
    in the patient's directory (.txt or .tse).
    """
    base_name_lower = os.path.basename(edf_file_path).lower()
    patient_dir = os.path.dirname(edf_file_path)

    # Search for any .txt or .tse file in the directory
    try:
        for f_name in os.listdir(patient_dir):
            if f_name.lower().endswith(('.txt', '.tse')):
                summary_file_path = os.path.join(patient_dir, f_name)
                all_patient_seizures = parse_summary_file(summary_file_path)
                # Check if our specific edf file has annotations in this summary
                if base_name_lower in all_patient_seizures:
                    return all_patient_seizures[base_name_lower]
    except FileNotFoundError:
        pass # It's normal for many directories to not have annotation files.
    except Exception as e:
        print(f"An unexpected error occurred while loading annotations for {edf_file_path}: {e}")

    # Return empty list if no annotations are found for this specific file
    return []