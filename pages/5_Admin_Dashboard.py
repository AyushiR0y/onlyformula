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
    st.title("📈 Admin Usage Dashboard")

    metrics = get_usage_metrics()
    overall = metrics.get("overall", {})
    sessions = metrics.get("sessions", [])

    # GPT-4o pricing per 1000 tokens (converted from per 1M)
    INPUT_COST_PER_1K_USD = 2.50 / 1000
    OUTPUT_COST_PER_1K_USD = 10.00 / 1000
    
    # USD to INR conversion rate
    USD_TO_INR = 83.0
    
    # Convert to INR
    INPUT_COST_PER_1K_INR = INPUT_COST_PER_1K_USD * USD_TO_INR
    OUTPUT_COST_PER_1K_INR = OUTPUT_COST_PER_1K_USD * USD_TO_INR

    # Calculate overall cost
    total_input_tokens = overall.get("input_tokens_total", 0)
    total_output_tokens = overall.get("output_tokens_total", 0)
    total_cost_inr = ((total_input_tokens / 1000) * INPUT_COST_PER_1K_INR) + ((total_output_tokens / 1000) * OUTPUT_COST_PER_1K_INR)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Sessions", overall.get("total_sessions", 0))
    with col2:
        st.metric("Total API Calls", overall.get("total_api_calls", 0))
    with col3:
        st.metric("Total Documents", overall.get("total_documents", 0))
    with col4:
        st.metric("Input Tokens", f"{total_input_tokens:,}")
    with col5:
        st.metric("Output Tokens", f"{total_output_tokens:,}")
    with col6:
        st.metric("Total Cost (INR)", f"₹{total_cost_inr:.2f}")

    st.markdown("---")
    st.subheader("Per Session Breakdown")

    if sessions:
        session_rows = []
        for session in sorted(sessions, key=lambda x: x["started_at"], reverse=True):
            session_id = session["session_id"]
            page = session["page"]
            started_at = session["started_at"]
            doc_count = session.get("document_count", 0)
            api_calls = len(session.get("api_calls", []))
            input_tokens = session.get("input_tokens_total", 0)
            output_tokens = session.get("output_tokens_total", 0)
            total_tokens = input_tokens + output_tokens
            cost_inr = ((input_tokens / 1000) * INPUT_COST_PER_1K_INR) + ((output_tokens / 1000) * OUTPUT_COST_PER_1K_INR)
            
            session_rows.append(
                {
                    "Session": session_id,
                    "Page": page,
                    "Started": started_at[:16],  # Show date and time only
                    "Documents": doc_count,
                    "API Calls": api_calls,
                    "Input Tokens": f"{input_tokens:,}",
                    "Output Tokens": f"{output_tokens:,}",
                    "Total Tokens": f"{total_tokens:,}",
                    "Cost (INR)": f"₹{cost_inr:.2f}",
                }
            )

        st.dataframe(session_rows, use_container_width=True, hide_index=True)
        
        # Expandable section for detailed API calls per session
        st.markdown("---")
        st.subheader("Session API Call Details")
        
        selected_session_id = st.selectbox(
            "Select a session to view API call details:",
            options=[s["session_id"] for s in sessions],
            format_func=lambda x: f"{x} ({next((s['page'] for s in sessions if s['session_id'] == x), 'Unknown')})"
        )
        
        if selected_session_id:
            selected_session = next((s for s in sessions if s["session_id"] == selected_session_id), None)
            if selected_session:
                st.write(f"**Session {selected_session_id}** - {selected_session['page']}")
                st.write(f"Started: {selected_session['started_at']}")
                st.write(f"Documents uploaded: {selected_session.get('document_count', 0)}")
                
                api_calls = selected_session.get("api_calls", [])
                if api_calls:
                    call_rows = []
                    for i, call in enumerate(api_calls, 1):
                        timestamp = call["timestamp"][:19]  # Show date and time
                        in_tokens = call.get("input_tokens", 0)
                        out_tokens = call.get("output_tokens", 0)
                        purpose = call.get("purpose", "general")
                        call_cost = ((in_tokens / 1000) * INPUT_COST_PER_1K_INR) + ((out_tokens / 1000) * OUTPUT_COST_PER_1K_INR)
                        
                        call_rows.append({
                            "Call #": i,
                            "Timestamp": timestamp,
                            "Purpose": purpose,
                            "Input": f"{in_tokens:,}",
                            "Output": f"{out_tokens:,}",
                            "Cost (INR)": f"₹{call_cost:.4f}",
                        })
                    
                    st.dataframe(call_rows, use_container_width=True, hide_index=True)
                else:
                    st.info("No API calls recorded for this session.")
    else:
        st.info("No usage data captured yet.")

    st.markdown("---")
    st.caption(f"Last Updated (UTC): {metrics.get('last_updated', 'N/A')}")
    st.caption(f"GPT-4o Pricing: ₹{INPUT_COST_PER_1K_INR:.4f} per 1K input tokens | ₹{OUTPUT_COST_PER_1K_INR:.4f} per 1K output tokens (USD to INR: {USD_TO_INR})")

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
