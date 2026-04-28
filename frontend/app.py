"""
Enterprise GenAI Knowledge Platform — Streamlit Frontend
Premium chat interface with model selection, governance, and document management.
"""

import streamlit as st
import requests
from pathlib import Path

# ──────────── Page Config ────────────

st.set_page_config(
    page_title="Enterprise GenAI Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = Path(__file__).parent / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# Import components
from components import (
    render_source_card,
    render_governance_badge,
    render_metrics_card,
    render_model_comparison_card,
)

# ──────────── Config ────────────

API_BASE = "http://localhost:8000"


def api_call(method: str, endpoint: str, **kwargs) -> dict | None:
    """Make an API call and handle errors."""
    try:
        url = f"{API_BASE}{endpoint}"
        resp = getattr(requests, method)(url, timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API server. Make sure it's running on port 8000.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API error: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


# ──────────── Session State ────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama-3.3-70b-versatile"


# ──────────── Sidebar ────────────

with st.sidebar:
    st.markdown("## 🧠 GenAI Platform")
    st.markdown("---")

    # Model Selection
    st.markdown("### 🤖 Model Selection")
    models_data = api_call("get", "/api/models")
    if models_data:
        model_names = [m["name"] for m in models_data]
        st.session_state.selected_model = st.selectbox(
            "Choose Model",
            model_names,
            index=model_names.index(st.session_state.selected_model)
            if st.session_state.selected_model in model_names
            else 0,
        )

        # Show model details
        selected = next(
            (m for m in models_data if m["name"] == st.session_state.selected_model),
            None,
        )
        if selected:
            st.markdown(
                f"<div style='font-size:0.85em; color:#94a3b8; padding:8px 0;'>"
                f"<strong>Provider:</strong> {selected.get('provider', 'N/A')}<br>"
                f"<strong>Cost:</strong> {selected.get('cost_tier', 'N/A').upper()}<br>"
                f"<strong>Status:</strong> {selected.get('status', 'N/A')}"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Governance Toggle
    st.markdown("### 🛡️ Governance")
    enable_governance = st.toggle("Enable Safety Checks", value=False)
    top_k = st.slider("Documents to Retrieve", 1, 15, 5)

    st.markdown("---")

    # Document Upload
    st.markdown("### 📁 Documents")

    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "docx", "txt", "md"],
        help="Upload enterprise documents for the knowledge base",
    )

    if uploaded_file:
        if st.button("📤 Process Document", use_container_width=True):
            with st.spinner("Processing..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                result = api_call("post", "/api/documents/upload", files=files)
                if result:
                    st.success(
                        f"✅ {result.get('filename')} — "
                        f"{result.get('chunks_created')} chunks created"
                    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📚 Load Samples", use_container_width=True):
            with st.spinner("Ingesting samples..."):
                result = api_call("post", "/api/documents/ingest-samples")
                if result:
                    st.success(f"✅ {result.get('chunks_created')} chunks loaded")

    with col2:
        if st.button("📊 Stats", use_container_width=True):
            stats = api_call("get", "/api/documents/stats")
            if stats:
                st.json(stats)

    st.markdown("---")

    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("❤️ Health Check", use_container_width=True):
        health = api_call("get", "/api/health")
        if health:
            st.json(health)


# ──────────── Main Content ────────────

# Header
st.markdown(
    """
    <div style="text-align:center; padding:20px 0 30px;">
        <h1 style="font-size:2.2em; margin:0;">🧠 Enterprise GenAI Knowledge Platform</h1>
        <p style="color:#94a3b8; font-size:1.05em; margin-top:8px;">
            AI-powered enterprise knowledge assistant with RAG, governance, and model management
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main tabs
tab_chat, tab_compare, tab_docs, tab_governance = st.tabs(
    ["💬 Chat", "⚖️ Model Compare", "📁 Documents", "🛡️ Governance"]
)

# ━━━━━━━━━━ Chat Tab ━━━━━━━━━━
with tab_chat:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show metadata for assistant messages
            if msg["role"] == "assistant" and msg.get("metadata"):
                meta = msg["metadata"]

                # Governance badge
                if meta.get("governance"):
                    render_governance_badge(meta["governance"])

                # Source citations
                if meta.get("sources"):
                    with st.expander(f"📎 Sources ({len(meta['sources'])})"):
                        for src in meta["sources"]:
                            render_source_card(src)

                # Response info
                st.markdown(
                    f"<div style='font-size:0.8em; color:#64748b; margin-top:8px;'>"
                    f"🤖 {meta.get('model', 'N/A')} · "
                    f"⏱ {meta.get('latency_ms', 0)}ms · "
                    f"📄 {meta.get('chunks', 0)} chunks"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Chat input
    if prompt := st.chat_input("Ask about enterprise documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = api_call(
                    "post",
                    "/api/chat",
                    json={
                        "question": prompt,
                        "model_name": st.session_state.selected_model,
                        "top_k": top_k,
                        "enable_governance": enable_governance,
                    },
                )

                if result:
                    answer = result.get("answer", "No response")
                    st.markdown(answer)

                    # Metadata
                    metadata = {
                        "model": result.get("model_used"),
                        "latency_ms": result.get("latency_ms"),
                        "chunks": result.get("retrieved_chunks"),
                        "sources": result.get("sources", []),
                        "governance": result.get("governance"),
                    }

                    # Governance badge
                    if metadata.get("governance"):
                        render_governance_badge(metadata["governance"])

                    # Sources
                    if metadata["sources"]:
                        with st.expander(f"📎 Sources ({len(metadata['sources'])})"):
                            for src in metadata["sources"]:
                                render_source_card(src)

                    st.markdown(
                        f"<div style='font-size:0.8em; color:#64748b; margin-top:8px;'>"
                        f"🤖 {metadata['model']} · "
                        f"⏱ {metadata['latency_ms']}ms · "
                        f"📄 {metadata['chunks']} chunks"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": metadata,
                    })
                else:
                    st.error("Failed to get response")

# ━━━━━━━━━━ Model Compare Tab ━━━━━━━━━━
with tab_compare:
    st.markdown("### ⚖️ Compare Model Responses")
    st.markdown(
        "<p style='color:#94a3b8;'>Run the same question across multiple models to compare quality, speed, and cost.</p>",
        unsafe_allow_html=True,
    )

    compare_question = st.text_area(
        "Question to compare",
        placeholder="Enter a question to compare across models...",
        height=80,
    )

    if models_data:
        compare_models = st.multiselect(
            "Select models to compare",
            [m["name"] for m in models_data],
            default=[m["name"] for m in models_data][:2],
        )

        if st.button("🔄 Compare Models", use_container_width=True) and compare_question:
            if len(compare_models) < 2:
                st.warning("Select at least 2 models to compare")
            else:
                with st.spinner("Running comparison..."):
                    results = api_call(
                        "post",
                        "/api/models/compare",
                        json={
                            "question": compare_question,
                            "model_names": compare_models,
                        },
                    )
                    if results:
                        for r in results:
                            render_model_comparison_card(r)

# ━━━━━━━━━━ Documents Tab ━━━━━━━━━━
with tab_docs:
    st.markdown("### 📁 Document Knowledge Base")

    col1, col2, col3 = st.columns(3)

    stats = api_call("get", "/api/documents/stats")
    if stats:
        with col1:
            render_metrics_card(
                "Total Chunks",
                str(stats.get("total_documents", 0)),
                "Document chunks indexed",
                "📊"
            )
        with col2:
            render_metrics_card(
                "Source Files",
                str(stats.get("unique_sources", 0)),
                "Unique documents",
                "📁"
            )
        with col3:
            render_metrics_card(
                "Collection",
                stats.get("collection_name", "N/A"),
                "ChromaDB collection",
                "🗄️"
            )

        st.markdown("---")
        st.markdown("#### Indexed Documents")
        for doc in stats.get("source_files", []):
            st.markdown(
                f"<div class='source-card'>📄 {doc}</div>",
                unsafe_allow_html=True,
            )

        if not stats.get("source_files"):
            st.info(
                "📭 No documents indexed yet. Upload documents or click "
                "'Load Samples' in the sidebar to get started."
            )

# ━━━━━━━━━━ Governance Tab ━━━━━━━━━━
with tab_governance:
    st.markdown("### 🛡️ AI Governance Controls")

    # Config display
    gov_config = api_call("get", "/api/governance/config")
    if gov_config:
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "🟢 Active" if gov_config.get("toxicity_enabled") else "🔴 Disabled"
            render_metrics_card("Toxicity Filter", status, "Detects harmful content", "🧪")
        with col2:
            status = "🟢 Active" if gov_config.get("hallucination_enabled") else "🔴 Disabled"
            render_metrics_card("Hallucination Detection", status, "Verifies grounded responses", "🔍")
        with col3:
            status = "🟢 Active" if gov_config.get("prompt_guard_enabled") else "🔴 Disabled"
            render_metrics_card("Prompt Guard", status, "Blocks prompt injection", "🛡️")

    st.markdown("---")
    st.markdown("#### 🧪 Test Governance Controls")

    test_text = st.text_area(
        "Enter text to test",
        placeholder="Type or paste text to run governance checks...",
        height=100,
    )

    check_type = st.selectbox(
        "Check Type",
        ["all", "toxicity", "prompt_safety"],
    )

    if st.button("🔍 Run Check", use_container_width=True) and test_text:
        with st.spinner("Running governance checks..."):
            result = api_call(
                "post",
                "/api/governance/check",
                json={"text": test_text, "check_type": check_type},
            )
            if result:
                st.json(result.get("results", {}))


# ──────────── Footer ────────────

st.markdown("---")
st.markdown(
    """<div style="text-align:center; color:#64748b; font-size:0.85em; padding:20px 0;">
        🧠 Enterprise GenAI Knowledge Platform v1.0 · Built with FastAPI, LangChain, ChromaDB & Streamlit
    </div>""",
    unsafe_allow_html=True,
)
