"""
Service layer — sits between the FastAPI router and core_engine.
All direct calls to core_engine.analyze_url() go through here so
the router stays thin and the business logic stays testable.

Key fix: the SQLite cache (get_cached_result) does not store gambling_warning
because it is not a DB column. When a cached result is returned, that field
is simply absent from the dict. We re-generate the warning using the same
get_gambling_warning() function from core_engine so Pydantic always receives
a complete, valid response dict — whether the result was cached or fresh.
"""
import sys
from pathlib import Path

# Make sure core_engine (wherever it lives) is importable.
CORE_DIR = Path(__file__).parent.parent.parent / "core"
sys.path.insert(0, str(CORE_DIR))

from core_engine import analyze_url, get_gambling_warning   # main fn + warning helper
from database import initialize_database                      # called once at startup


def ensure_db_ready():
    """
    Called once at application startup (see main.py lifespan).
    Initialises the SQLite schema if it does not already exist.
    """
    initialize_database()


def _patch_gambling_warning(result: dict) -> dict:
    """
    gambling_warning is NOT a column in the url_analysis table, so
    get_cached_result() never includes it. Fresh results from analyze_url()
    always include it. This function ensures the key always exists.

    For cached results we reconstruct the warning from the scores that ARE
    stored in the DB, using the identical logic core_engine uses.
    """
    if "gambling_warning" in result:
        # Fresh result — field is already present (may be None)
        return result

    # Build a minimal features-like dict from what the DB did store
    fake_features = {
        "total_score":        result.get("total_score", 0),
        "keyword_score":      result.get("keyword_score", 0),
        "is_gambling":        result.get("risk_type") == "Gambling/Betting",
        "inferred_risk_type": result.get("risk_type", "Unknown"),
    }
    result["gambling_warning"] = get_gambling_warning(fake_features)
    return result


def run_analysis(url: str) -> dict:
    """
    Call core_engine.analyze_url() and return its result dict.
    Always guarantees the returned dict contains 'gambling_warning'.

    Raises:
        ValueError   – if core_engine signals a bad URL via an 'error' key
        RuntimeError – for unexpected exceptions inside the analysis engine
    """
    try:
        result = analyze_url(url)
    except Exception as exc:
        raise RuntimeError(f"Analysis engine failure: {exc}") from exc

    # core_engine signals an invalid URL by setting result['error']
    if "error" in result:
        raise ValueError(result["error"])

    # Guarantee gambling_warning exists (handles both fresh + cached paths)
    result = _patch_gambling_warning(result)

    return result