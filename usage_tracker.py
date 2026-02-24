import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st


USAGE_FILE_PATH = Path(__file__).resolve().parent / "usage_metrics.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_metrics() -> Dict[str, Any]:
    return {
        "last_updated": _utc_now_iso(),
        "overall": {
            "total_sessions": 0,
            "total_api_calls": 0,
            "input_tokens_total": 0,
            "output_tokens_total": 0,
            "total_documents": 0,
        },
        "sessions": [],
    }


def _load_metrics() -> Dict[str, Any]:
    if not USAGE_FILE_PATH.exists():
        return _default_metrics()

    try:
        with open(USAGE_FILE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            return _default_metrics()
        if "overall" not in data or "sessions" not in data:
            return _default_metrics()
        return data
    except Exception:
        return _default_metrics()


def _save_metrics(metrics: Dict[str, Any]) -> None:
    metrics["last_updated"] = _utc_now_iso()
    try:
        temp_path = USAGE_FILE_PATH.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
        os.replace(temp_path, USAGE_FILE_PATH)
    except Exception as e:
        print(f"[usage_tracker] Failed to save metrics: {e}")
        print(f"   Attempted to write to: {USAGE_FILE_PATH}")


def _ensure_session_id() -> str:
    if "usage_session_id" not in st.session_state:
        st.session_state["usage_session_id"] = str(uuid.uuid4())
    return st.session_state["usage_session_id"]


def _get_or_create_session(page_name: str, validate_by_documents: bool = False) -> Dict[str, Any]:
    """Get or create current session entry.
    
    Args:
        page_name: The name of the page
        validate_by_documents: If True, only count session if it has documents
    """
    session_id = _ensure_session_id()
    
    metrics = _load_metrics()
    
    # Find existing session
    for session in metrics["sessions"]:
        if session["uuid"] == session_id and session["page"] == page_name:
            return session
    
    # Create new session with readable name
    session_num = len(metrics["sessions"]) + 1
    session = {
        "session_id": f"Session_{session_num}",
        "uuid": session_id,
        "page": page_name,
        "started_at": _utc_now_iso(),
        "document_count": 0,
        "api_calls": [],
        "input_tokens_total": 0,
        "output_tokens_total": 0,
    }
    
    metrics["sessions"].append(session)
    # Only count sessions that have documents (or if not validating by documents)
    if not validate_by_documents:
        metrics["overall"]["total_sessions"] = len(metrics["sessions"])
    _save_metrics(metrics)
    
    return session


def track_page_visit(page_name: str) -> None:
    """Track a page visit and ensure session ID exists.
    
    Note: Does NOT create a session record. Session records are only created
    when documents are uploaded. This prevents counting sessions without uploads.
    """
    try:
        _ensure_session_id()
        print(f"[usage_tracker] Page '{page_name}' visited - session ID ensured")
    except Exception as e:
        print(f"[usage_tracker] Failed to track page visit: {e}")


def track_api_call(page_name: str, input_tokens: int = 0, output_tokens: int = 0, purpose: str = "") -> None:
    """Track an API call with token counts."""
    try:
        session_id = _ensure_session_id()
        metrics = _load_metrics()
        
        # Find the session by UUID (not session_id which is the display name)
        session = None
        for s in metrics["sessions"]:
            if s["uuid"] == session_id and s["page"] == page_name:
                session = s
                break
        
        if not session:
            session = _get_or_create_session(page_name)
        
        # Record API call
        api_call = {
            "timestamp": _utc_now_iso(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        if purpose:
            api_call["purpose"] = purpose
        
        session["api_calls"].append(api_call)
        session["input_tokens_total"] = session.get("input_tokens_total", 0) + input_tokens
        session["output_tokens_total"] = session.get("output_tokens_total", 0) + output_tokens
        
        # Update overall metrics
        metrics["overall"]["total_api_calls"] = metrics["overall"].get("total_api_calls", 0) + 1
        metrics["overall"]["input_tokens_total"] = metrics["overall"].get("input_tokens_total", 0) + input_tokens
        metrics["overall"]["output_tokens_total"] = metrics["overall"].get("output_tokens_total", 0) + output_tokens
        
        _save_metrics(metrics)
        print(f"[usage_tracker] API call tracked for '{page_name}' ({input_tokens} in / {output_tokens} out)")
    except Exception as e:
        print(f"[usage_tracker] Failed to track API call: {e}")


def track_document_upload(page_name: str, count: int = 1) -> None:
    """Track document uploads in a session (only increment once per upload batch).
    Only count as a session if at least 1 document is uploaded.
    """
    try:
        # Only track if count > 0
        if count <= 0:
            return
            
        session_id = _ensure_session_id()
        metrics = _load_metrics()
        
        # Find the session
        session = None
        for s in metrics["sessions"]:
            if s["uuid"] == session_id and s["page"] == page_name:
                session = s
                break
        
        if not session:
            session = _get_or_create_session(page_name, validate_by_documents=True)
        
        # Only increment if it's the first upload (marking session as valid/countable)
        prev_count = session.get("document_count", 0)
        session["document_count"] = prev_count + count
        
        # Only increment total_sessions if this is the first document in this session
        if prev_count == 0:
            metrics["overall"]["total_sessions"] = metrics["overall"].get("total_sessions", 0) + 1
            
        metrics["overall"]["total_documents"] = metrics["overall"].get("total_documents", 0) + count
        
        _save_metrics(metrics)
        print(f"[usage_tracker] Uploaded {count} document(s) to {page_name} (total in session: {session['document_count']})")
    except Exception as e:
        print(f"[usage_tracker] Failed to track document upload: {e}")


def get_usage_metrics() -> Dict[str, Any]:
    """Get all usage metrics."""
    return _load_metrics()


def reset_all_sessions() -> None:
    """Reset all sessions and overall metrics (admin function)."""
    try:
        metrics = _default_metrics()
        _save_metrics(metrics)
        print("[usage_tracker] All sessions and metrics have been reset")
    except Exception as e:
        print(f"[usage_tracker] Failed to reset sessions: {e}")


def delete_session(session_id: str, page_name: str) -> bool:
    """Delete a specific session by UUID and page name.
    Returns True if deleted, False if not found.
    """
    try:
        metrics = _load_metrics()
        
        # Find and remove the session
        original_count = len(metrics["sessions"])
        metrics["sessions"] = [
            s for s in metrics["sessions"]
            if not (s["uuid"] == session_id and s["page"] == page_name)
        ]
        
        if len(metrics["sessions"]) < original_count:
            # Session was deleted; recalculate totals
            deleted_session = next(
                (s for s in metrics["sessions"] if s["uuid"] == session_id and s["page"] == page_name),
                None
            )
            
            # Recalculate overall metrics from remaining sessions
            total_sessions = len(metrics["sessions"])
            total_docs = sum(s.get("document_count", 0) for s in metrics["sessions"])
            total_api_calls = sum(len(s.get("api_calls", [])) for s in metrics["sessions"])
            total_input_tokens = sum(s.get("input_tokens_total", 0) for s in metrics["sessions"])
            total_output_tokens = sum(s.get("output_tokens_total", 0) for s in metrics["sessions"])
            
            metrics["overall"]["total_sessions"] = total_sessions
            metrics["overall"]["total_documents"] = total_docs
            metrics["overall"]["total_api_calls"] = total_api_calls
            metrics["overall"]["input_tokens_total"] = total_input_tokens
            metrics["overall"]["output_tokens_total"] = total_output_tokens
            
            _save_metrics(metrics)
            print(f"[usage_tracker] Session {session_id[:8]}... deleted from page '{page_name}'")
            return True
        else:
            print(f"[usage_tracker] Session not found: {session_id}")
            return False
    except Exception as e:
        print(f"[usage_tracker] Failed to delete session: {e}")
        return False
