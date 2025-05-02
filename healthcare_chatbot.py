import streamlit as st
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the page configuration
st.set_page_config(
    page_title="HealthCare AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
if "messages" not in st.session_state:
    st.session_state.messages = []

# System prompt for the healthcare assistant
sys_msg = """
You are a highly knowledgeable and empathetic healthcare assistant powered by advanced AI. Your role is to provide accurate, concise, and professional responses to health-related questions. Always prioritize user safety, recommend consulting a licensed medical professional for serious concerns, and avoid providing definitive diagnoses. If a question is outside your expertise, admit the limitation and suggest seeking professional help.
"""
API_KEY= os.getenv("API_KEY")
# Function to get response from Grok
def get_grok_response(user_input):
    try:
        model = ChatGroq(
            temperature=0.2,
            model_name="llama3-8b-8192",
            api_key=os.getenv("GROQ_API_KEY", API_KEY)
        )
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(sys_msg),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        chain = prompt_template | model
        runnable_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: st.session_state.chat_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        response = runnable_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "user_session"}}
        )
        return response.content
    except Exception as e:
        return f"Error: Unable to process your request. Please try again. ({str(e)})"

# Custom CSS for professional UI (original design with enhancements)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 20px;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 1px solid #007bff;
        padding: 12px;
        background-color: #ffffff;
        color: #000000;
        font-size: 14px;
        transition: border-color 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #0056b3;
        box-shadow: 0 0 5px rgba(0,123,255,0.3);
    }
    .stTextInput > div > div > input::placeholder {
        color: #666;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s ease;
        margin-top: 10px;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding: 10px;
        margin-bottom: 120px; /* Space for fixed input */
    }
    .chat-message {
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        animation: fadeIn 0.3s ease-in;
        max-width: 80%;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 0;
    }
    .bot-message {
        background-color: #ffffff;
        color: #333;
        margin-right: auto;
        border-bottom-left-radius: 0;
    }
    .timestamp {
        font-size: 10px;
        color: #999;
        margin-top: 5px;
        text-align: right;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #007bff;
        font-family: 'Arial', sans-serif;
    }
    .footer {
        text-align: center;
        color: #666;
        padding: 20px 0;
        font-size: 12px;
    }
    .input-container {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 90%;
        max-width: 800px;
        background-color: #f5f7fa;
        padding: 10px;
        z-index: 1000;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# Auto-scroll JavaScript
st.markdown("""
    <script>
    function scrollToBottom() {
        const chatContainer = document.querySelector(".chat-container");
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    window.addEventListener("load", scrollToBottom);
    </script>
""", unsafe_allow_html=True)

# Sidebar content (unchanged)
with st.sidebar:
    st.image("https://via.placeholder.com/100x100.png?text=🏥", use_column_width=True)
    st.header("HealthCare AI")
    st.markdown("Your trusted AI-powered healthcare assistant. Ask any health-related question, and get quick, reliable answers.")
    st.markdown("**Note:** For serious health concerns, always consult a licensed medical professional.")
    
    # Clear chat history button (fixed)
    if st.button("Clear Chat History"):
        st.session_state.chat_history = InMemoryChatMessageHistory()
        st.session_state.messages = []
        st.success("Chat history cleared!")

# Main content
st.title("🏥 HealthCare AI Assistant")
st.markdown("Welcome to your personal healthcare assistant. How can I help you today?")

# Chat container
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        timestamp = datetime.now().strftime("%H:%M")
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">\U0001F464 {message["content"]}<div class="timestamp">{timestamp}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">\U0001F916 {message["content"]}<div class="timestamp">{timestamp}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input (original alignment with fixed-bottom style)
with st.form(key="chat_form", clear_on_submit=True):
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    user_input = st.text_input("Ask a health-related question:", placeholder="E.g., What are the symptoms of a common cold?", key="user_input")
    submit_button = st.form_submit_button("Send")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if submit_button and user_input:
        with st.spinner("Thinking..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            bot_response = get_grok_response(user_input)
            st.session_state.messages.append({"role": "bot", "content": bot_response})
            st.markdown('<script>scrollToBottom();</script>', unsafe_allow_html=True)

# Footer (unchanged)
st.markdown('<div class="footer">© 2025 HealthCare AI. All rights reserved. Powered by xAI.</div>', unsafe_allow_html=True)
