import os
import json
import streamlit as st
from dotenv import load_dotenv

from usage_tracker import get_usage_metrics, track_page_visit


load_dotenv()


def _get_admin_password() -> str:
    password_from_secrets = ""
    try:
        password_from_secrets = st.secrets.get("ADMIN_DASHBOARD_PASSWORD", "")
    except Exception:
        password_from_secrets = ""

    return password_from_secrets or os.getenv("ADMIN_DASHBOARD_PASSWORD", "admin123")


def _is_authenticated() -> bool:
    return st.session_state.get("admin_dashboard_authenticated", False)


def _show_login() -> None:
    st.title("🔐 Admin Dashboard")
    st.info("Enter admin password to view usage analytics.")

    configured_password = _get_admin_password()
    if configured_password == "admin123":
        st.warning("Using default admin password. Set ADMIN_DASHBOARD_PASSWORD in .env or Streamlit secrets.")

    password = st.text_input("Password", type="password")
    if st.button("Login", type="primary", use_container_width=True):
        if password == configured_password:
            st.session_state["admin_dashboard_authenticated"] = True
            st.success("Access granted.")
            st.rerun()
        else:
            st.error("Invalid password.")


def _render_dashboard() -> None:
    track_page_visit("Admin Dashboard")

    st.title("📈 Admin Usage Dashboard")

    metrics = get_usage_metrics()
    overall = metrics.get("overall", {})
    pages = metrics.get("pages", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("App Sessions", overall.get("app_sessions", 0))
    with col2:
        st.metric("Total Page Views", overall.get("page_views_total", 0))
    with col3:
        st.metric("Total API Calls", overall.get("api_calls_total", 0))

    st.markdown("---")
    st.subheader("Per Page Usage")

    if pages:
        page_rows = []
        for page_name, page_data in sorted(pages.items()):
            page_rows.append(
                {
                    "Page": page_name,
                    "Page Views": page_data.get("page_views", 0),
                    "API Calls": page_data.get("api_calls", 0),
                }
            )

        st.dataframe(page_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No usage data captured yet.")

    st.caption(f"Last Updated (UTC): {metrics.get('last_updated', 'N/A')}")

    st.download_button(
        label="📥 Download Raw Usage JSON",
        data=json.dumps(metrics, indent=2),
        file_name="usage_metrics.json",
        mime="application/json",
        use_container_width=True,
    )

    if st.button("Logout", use_container_width=True):
        st.session_state["admin_dashboard_authenticated"] = False
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Admin Dashboard", page_icon="🔐", layout="wide")

    if not _is_authenticated():
        _show_login()
        return

    _render_dashboard()


if __name__ == "__main__":
    main()
