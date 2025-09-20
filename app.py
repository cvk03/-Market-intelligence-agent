import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.core import MarketIntelligenceAgent
from src.utils.vector_store import SimpleVectorStore

# Page config
st.set_page_config(
    page_title="Market Intelligence Chatbot",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_resources():
    """Load agent and vector store"""
    try:
        agent = MarketIntelligenceAgent()
        vector_store = SimpleVectorStore()
        vector_store.load()
        # Extract types and regions for options, fallback to common ones if not found
        types = set()
        regions = set()
        for doc in getattr(vector_store, "texts", []):
            if isinstance(doc, dict):  # If texts are stored as dicts
                types.add(doc.get("insurance_type", "auto"))
                regions.add(doc.get("state", "CA"))
        if not types: types = {"auto", "home", "life", "health"}
        if not regions: regions = {"CA", "TX", "FL", "NY", "PA"}
        return agent, vector_store, sorted(types), sorted(regions), None
    except Exception as e:
        return None, None, [], [], str(e)

def filter_search_results(results, insurance_type, region):
    """Filter vector store results by type and region, unless 'all' is selected."""
    if insurance_type.lower() == "all" and region.lower() == "all":
        return results
    filtered = []
    for r in results:
        data = r.get('meta', {}) if 'meta' in r else r
        type_match = (insurance_type.lower() == "all" or data.get('insurance_type', '').lower() == insurance_type.lower())
        region_match = (region.lower() == "all" or data.get('state', '').lower() == region.lower())
        if type_match and region_match:
            filtered.append(r)
    return filtered if filtered else results  # Fallback to all if nothing matches

def main():
    st.title("🤖 Insurance Market Intelligence Chatbot")
    st.markdown("Chat with an AI agent for insurance market analysis, pricing, and trends.")

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ Google API key not found! Please set GOOGLE_API_KEY in your .env file.")
        st.info("Get your free API key from: https://makersuite.google.com/app/apikey")
        return

    # Load resources
    with st.spinner("Loading agent..."):
        agent, vector_store, types, regions, error = load_resources()

    if error:
        st.error(f"Failed to load resources: {error}")
        return

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.success("✅ Gemini API Connected")
        st.info("Using: gemini-pro model")
        if vector_store and getattr(vector_store, "index", None):
            st.metric("Documents Indexed", len(getattr(vector_store, "texts", [])))

    # Extract types/regions, add "All" at the top and ensure session state
    types = ["all"] + [t for t in types if t.lower() != "all"]
    regions = ["all"] + [r for r in regions if r.lower() != "all"]
    if "insurance_type" not in st.session_state:
        st.session_state["insurance_type"] = types[0]
    if "region" not in st.session_state:
        st.session_state["region"] = regions[0]

    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "text": "Hi! 👋 I'm your insurance market intelligence agent. Ask me anything about insurance rates, trends, or pricing strategies."}
        ]

    # User input (chat box)
    user_query = st.chat_input("Type your insurance question...")

    # Sidebar: Context Settings with dynamic options
    with st.sidebar:
        st.subheader("Context Settings")
        insurance_type = st.selectbox("Insurance Type:", types, index=types.index(st.session_state["insurance_type"]))
        region = st.selectbox("Region:", regions, index=regions.index(st.session_state["region"]))
        # Update session state if changed
        st.session_state["insurance_type"] = insurance_type
        st.session_state["region"] = region

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    if user_query:
        st.session_state["messages"].append({"role": "user", "text": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("🤖 Gemini is analyzing..."):
            try:
                # Search for relevant data
                search_results = vector_store.search(user_query, k=10)
                filtered_results = filter_search_results(
                    search_results, 
                    st.session_state["insurance_type"], 
                    st.session_state["region"]
                )
                if not filtered_results:
                    agent_reply = "No relevant data found. Try a different question or selection."
                else:
                    relevant_data = "\n".join([r['text'] for r in filtered_results[:5]])
                    # Pass "all" if selected
                    result = agent.process_query_sync(
                        query=user_query,
                        retrieved_data=relevant_data,
                        insurance_type=st.session_state["insurance_type"],
                        region=st.session_state["region"]
                    )
                    if result['success']:
                        agent_reply = f"**Query Type:** {result['query_type'].replace('_', ' ').title()}\n\n"
                        agent_reply += f"**Confidence:** {result['confidence']*100:.0f}%\n\n"
                        agent_reply += f"### 📊 Analysis Results\n{result['analysis']}"
                    else:
                        agent_reply = f"Analysis failed: {result.get('error', 'Unknown error')}"
            except Exception as e:
                agent_reply = f"An error occurred: {str(e)}"

        st.session_state["messages"].append({"role": "assistant", "text": agent_reply})
        st.experimental_rerun()

    # Footer
    st.markdown("---")
    st.caption("Powered by Google Gemini AI and Semantic Kernel")

if __name__ == "__main__":
    main()