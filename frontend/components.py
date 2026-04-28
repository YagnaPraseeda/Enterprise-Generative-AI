"""
Reusable Streamlit UI components.
"""

import streamlit as st


def render_source_card(source: dict):
    """Render a document source reference card."""
    doc_name = source.get("document", "Unknown")
    file_type = source.get("file_type", "")
    icon = {"pdf": "📄", "docx": "📝", "txt": "📃", "md": "📋"}.get(file_type, "📎")

    st.markdown(
        f"""<div class="source-card">
            {icon} <strong>{doc_name}</strong>
            <span class="badge">{file_type.upper()}</span>
        </div>""",
        unsafe_allow_html=True,
    )


def render_governance_badge(governance: dict):
    """Render governance status badges."""
    if not governance:
        return

    confidence = governance.get("confidence_score", 0)
    warnings = governance.get("warnings", [])

    # Color based on confidence
    if confidence >= 0.7:
        color, icon, label = "#10b981", "✅", "Safe"
    elif confidence >= 0.4:
        color, icon, label = "#f59e0b", "⚠️", "Caution"
    else:
        color, icon, label = "#ef4444", "🚨", "Warning"

    st.markdown(
        f"""<div style="display:flex; align-items:center; gap:12px; 
                margin:8px 0; padding:10px 16px; 
                background: {color}15; border-left:3px solid {color}; 
                border-radius:8px;">
            <span style="font-size:1.2em">{icon}</span>
            <div>
                <div style="font-weight:600; color:{color}">{label} — Confidence: {confidence:.0%}</div>
                {''.join(f'<div style="font-size:0.85em; color:#94a3b8; margin-top:2px">• {w}</div>' for w in warnings)}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_metrics_card(title: str, value: str, subtitle: str = "", icon: str = "📊"):
    """Render a dashboard metric card."""
    st.markdown(
        f"""<div class="metric-card">
            <div style="font-size:1.5em; margin-bottom:4px">{icon}</div>
            <div style="font-size:1.8em; font-weight:700; color:#e2e8f0">{value}</div>
            <div style="font-size:0.9em; font-weight:600; color:#94a3b8; margin-top:2px">{title}</div>
            <div style="font-size:0.75em; color:#64748b; margin-top:2px">{subtitle}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_model_comparison_card(result: dict):
    """Render a model comparison result."""
    model = result.get("model", "Unknown")
    answer = result.get("answer", "")
    latency = result.get("latency_ms", 0)
    cost_tier = result.get("cost_tier", "unknown")
    status = result.get("status", "unknown")

    tier_colors = {"low": "#10b981", "medium": "#f59e0b", "high": "#ef4444"}
    tier_color = tier_colors.get(cost_tier, "#94a3b8")

    status_icon = "✅" if status == "success" else "❌"

    st.markdown(
        f"""<div style="background:#1e293b; border-radius:12px; padding:20px; 
                margin:8px 0; border:1px solid #334155;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div style="font-weight:700; font-size:1.1em; color:#e2e8f0">
                    {status_icon} {model}
                </div>
                <div style="display:flex; gap:12px; font-size:0.85em;">
                    <span style="color:#94a3b8">⏱ {latency}ms</span>
                    <span style="color:{tier_color}; font-weight:600">
                        💰 {cost_tier.upper()}
                    </span>
                </div>
            </div>
            <div style="color:#cbd5e1; line-height:1.6; font-size:0.95em;">
                {answer[:500]}{'...' if len(answer) > 500 else ''}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
