from pathlib import Path

import yaml

PROFILE_PATH = Path(__file__).parent / "candidate_profile.yaml"


def get_profile():
    # Load Candidate profile
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(f"Profile not found: {PROFILE_PATH}")

    with PROFILE_PATH.open("r", encoding="utf-8") as f:
        candidate_profile = yaml.safe_load(f)

    if not isinstance(candidate_profile, dict):
        raise ValueError(f"Invalid candidate profile: {PROFILE_PATH}")

    return candidate_profile
