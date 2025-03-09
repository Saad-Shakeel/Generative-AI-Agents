import streamlit as st
import time
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
# Initialize Gmail Agent
credentials = get_gmail_credentials(
    scopes=["https://mail.google.com/"],
    token_file="token.json",
    client_secrets_file="Credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)
tools = toolkit.get_tools()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt = """You are a Personal Email Agent with access to the user's Gmail via the Gmail API. Your tasks include:

Fetch Emails: Retrieve emails based on queries (unread, specific sender, attachments, etc.).
Summarize Emails: Provide concise summaries while preserving key details.
Categorize Emails: Classify emails (work, personal, promotions, spam).
Search Emails: Find emails by keywords, sender, date, or subject.
Draft Replies: Assist in composing context-aware, professional responses.
Manage Emails: Mark as read/unread, archive, label, or delete (with confirmation).
"""

agent_executor = create_react_agent(llm, tools, prompt=prompt,checkpointer=memory)

# Streamlit UI
st.set_page_config(page_title="Gmail Assistant", page_icon="📧", layout="wide")
st.title("📧 Gmail Personal Assistant")
st.caption("Manage your emails with AI-powered assistance")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Ask about your emails..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Process query with Gmail Agent
    config = {"configurable": {"thread_id": "1"}}
    events = agent_executor.stream({"messages": [("user", user_input)]}, stream_mode="values",config=config)
    
    response_placeholder = st.empty()
    final_response = ""
    for event in events:
        final_response = event["messages"][-1].content
        response_placeholder.markdown(final_response)
        time.sleep(0.1)
    
    st.session_state.messages.append({"role": "assistant", "content": final_response})
    st.rerun()
