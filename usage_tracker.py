import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import streamlit as st


USAGE_FILE_PATH = Path(__file__).resolve().parent / "usage_metrics.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_metrics() -> Dict[str, Any]:
    return {
        "last_updated": _utc_now_iso(),
        "overall": {
            "app_sessions": 0,
            "page_views_total": 0,
            "api_calls_total": 0,
        },
        "pages": {},
    }


def _load_metrics() -> Dict[str, Any]:
    if not USAGE_FILE_PATH.exists():
        return _default_metrics()

    try:
        with open(USAGE_FILE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            return _default_metrics()
        if "overall" not in data or "pages" not in data:
            return _default_metrics()
        return data
    except Exception:
        return _default_metrics()


def _save_metrics(metrics: Dict[str, Any]) -> None:
    metrics["last_updated"] = _utc_now_iso()
    temp_path = USAGE_FILE_PATH.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    os.replace(temp_path, USAGE_FILE_PATH)


def _ensure_page(metrics: Dict[str, Any], page_name: str) -> None:
    if page_name not in metrics["pages"]:
        metrics["pages"][page_name] = {
            "page_views": 0,
            "api_calls": 0,
        }


def _ensure_session_id() -> str:
    if "usage_session_id" not in st.session_state:
        st.session_state["usage_session_id"] = str(uuid.uuid4())
    return st.session_state["usage_session_id"]


def track_page_visit(page_name: str) -> None:
    _ensure_session_id()

    if not st.session_state.get("usage_app_session_logged", False):
        metrics = _load_metrics()
        metrics["overall"]["app_sessions"] = metrics["overall"].get("app_sessions", 0) + 1
        _save_metrics(metrics)
        st.session_state["usage_app_session_logged"] = True

    page_flag = f"usage_page_logged::{page_name}"
    if st.session_state.get(page_flag, False):
        return

    metrics = _load_metrics()
    _ensure_page(metrics, page_name)
    metrics["pages"][page_name]["page_views"] += 1
    metrics["overall"]["page_views_total"] = metrics["overall"].get("page_views_total", 0) + 1
    _save_metrics(metrics)
    st.session_state[page_flag] = True


def track_api_call(page_name: str) -> None:
    metrics = _load_metrics()
    _ensure_page(metrics, page_name)
    metrics["pages"][page_name]["api_calls"] += 1
    metrics["overall"]["api_calls_total"] = metrics["overall"].get("api_calls_total", 0) + 1
    _save_metrics(metrics)


def get_usage_metrics() -> Dict[str, Any]:
    return _load_metrics()
