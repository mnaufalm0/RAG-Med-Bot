import streamlit as st
from pipeline import MediRAG

st.set_page_config(page_title="MediRAG", page_icon="🩺", layout="centered")


if "history" not in st.session_state:
    st.session_state.history = []
if "bot" not in st.session_state:
    st.session_state.bot = None


with st.sidebar:
    st.title("MediRAG")
    st.caption("RAG health assistant")
    st.divider()

    groq_key = st.text_input("Groq API key", type="password", placeholder="gsk_...")
    model = st.selectbox("model", ["default", "fast", "mixtral"])
    show_sources = st.checkbox("show sources", value=True)

    if st.button("load", use_container_width=True):
        with st.spinner("loading..."):
            try:
                st.session_state.bot = MediRAG(
                    groq_api_key=groq_key or None,
                    model=model,
                    k=5,
)
                meta = st.session_state.bot.meta
                st.success(f"ready — {meta.get('disease_count', '?')} diseases loaded")
            except FileNotFoundError:
                st.error("index not found, run build_index.py first")
            except ValueError as e:
                st.error(str(e))

    st.divider()
    if st.button("clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()


st.header("MediRAG — Health Assistant")
st.caption("Ask about symptoms, diseases, or treatments. Answers are grounded in the dataset.")
st.divider()

if not st.session_state.history:
    st.write("Try asking:")
    cols = st.columns(2)
    suggestions = [
        "Symptoms of diabetes?",
        "How is pneumonia treated?",
        "What causes hypertension?",
        "Dengue fever precautions?",
    ]
    for i, q in enumerate(suggestions):
        if cols[i % 2].button(q, use_container_width=True):
            st.session_state._pending = q


for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and show_sources:
            if msg.get("diseases"):
                st.caption(f"diseases: {', '.join(msg['diseases'])}")
            if msg.get("sources"):
                st.caption("sources: " + " · ".join(msg["sources"]))


user_input = st.chat_input("ask something...")

if hasattr(st.session_state, "_pending"):
    user_input = st.session_state._pending
    del st.session_state._pending

if user_input:
    if st.session_state.bot is None:
        st.warning("load the pipeline first using the sidebar")
    else:
        st.session_state.history.append({"role": "user", "content": user_input})

        with st.spinner("searching..."):
            try:
                result = st.session_state.bot.ask(user_input)
                st.session_state.history.append({
                    "role":     "assistant",
                    "content":  result.answer,
                    "diseases": result.diseases,
                    "sources":  result.sources(),
                })
            except Exception as e:
                st.session_state.history.append({
                    "role":    "assistant",
                    "content": f"error: {e}",
                    "diseases": [],
                    "sources":  [],
                })

        st.rerun()

st.divider()
