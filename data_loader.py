"""
data_loader.py -- Loads the bundled reference dataset.

Extracted from server.py as part of the modularization requested in
mentor review Comments 1, 3, and 13 (Overall Organization, Code Structure,
Scalability). Previously this was a few lines inline at the top of
server.py; pulling it out means the dataset can be loaded and inspected
independently (e.g. from a notebook or the evaluation-report script)
without booting the whole Flask app.
"""
import json
from pathlib import Path

BASE = Path(__file__).parent


def load_disease_data(path=None):
    """Load disease_data.json into the same dict shape server.py has always used:
       {disease_name: {years, case_series, death_series, immunization_series,
                        total_cases, total_deaths, n_locations, ...}}"""
    p = path or (BASE / 'disease_data.json')
    with open(p) as f:
        return json.load(f)


# Loaded once at import time, exactly like the old inline version in server.py --
# every module that needs the dataset imports DISEASE_DATA from here rather than
# each re-reading the file.
DISEASE_DATA = load_disease_data()
