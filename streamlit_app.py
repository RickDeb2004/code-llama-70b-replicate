import streamlit as st
import replicate
import os
import pandas as pd
import gettext

# Initialize translation
lang = "en_US"  # Default language is English
lang_path = os.path.join(os.path.dirname(__file__), "locales")
gettext.bindtextdomain("messages", lang_path)
gettext.textdomain("messages")
_ = gettext.gettext

# Constants
PAGE_TITLE = _("Code Llama 70B Instruct")
PAGE_ICON = ":llama:"
MODEL_NAME = "meta/codellama-70b-instruct:a279116fe47a0f65701a8817188601e2fe8f4b9e04a518789655ea7b995851bf"
DEFAULT_MAX_TOKENS = 2000
DEFAULT_SYSTEM_PROMPT = _(
    """You are a Python programming expert.
Your task is to help users by answering questions related to Python or by executing tasks requested of you.
You should ALWAYS structure code in markdown format before you respond.
If the task is unrelated to Python, then say 'I cannot help with anything not related to Python.'"""
)
UNWANTED_TEXT = _(" Source: assistant")


def initialize_streamlit():
    """Initializes Streamlit app configuration."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")


def display_icon(emoji: str):
    """Displays an emoji as a Notion-style page icon."""
    st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)


def setup_api_token():
    """Sets up the API token for Replicate."""
    api_token = st.secrets["REPLICATE_API_TOKEN"]
    os.environ["REPLICATE_API_TOKEN"] = api_token


def customize_input():
    """Handles custom inputs for the model."""
    with st.expander(_("🎛️ **Customize**"), expanded=False):
        max_tokens = st.number_input(_("Max tokens to return"), value=DEFAULT_MAX_TOKENS)
        system_prompt = st.text_area(_("System prompt"), value=DEFAULT_SYSTEM_PROMPT)
    return max_tokens, system_prompt


def display_chat_history():
    """Displays chat messages from the session history."""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.text(f"{_('You')}: {message['content']}")
        elif message["role"] == "assistant":
            st.text(f"{_('Assistant')}: {message['content']}")


def process_prompt(prompt: str, max_tokens: int, system_prompt: str):
    """Processes the user prompt and fetches response from the model."""
    output_generator = replicate.run(
        MODEL_NAME,
        input={
            "top_k": 10,
            "top_p": 0.95,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.8,
            "system_prompt": system_prompt,
            "repeat_penalty": 1.1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
    )
    return output_generator


def display_response(output_generator):
    """Displays the model's response in the chat and handles the chat history."""
    output_list = list(output_generator)
    if output_list:
        output_text = ''.join(output_list).rstrip(UNWANTED_TEXT)
        st.text(f"{_('Assistant')}: {output_text}")
    else:
        output_text = _("No output received.")
        st.write(output_text)
    return output_text


def main():
    """Main function to run the Streamlit app."""
    initialize_streamlit()
    display_icon(PAGE_ICON)
    st.title(PAGE_TITLE, anchor=False)
    setup_api_token()

    max_tokens, system_prompt = customize_input()
    st.divider()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    display_chat_history()

    # User input
    prompt = st.text_input(_("Enter your prompt..."))
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        output_generator = process_prompt(prompt, max_tokens, system_prompt)
        output_text = display_response(output_generator)
        st.session_state.messages.append({"role": "assistant", "content": output_text})

    # File Upload
    st.sidebar.header(_("File Upload"))
    uploaded_file = st.sidebar.file_uploader(_("Upload CSV or Excel file"), type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Process the uploaded file
        df = pd.read_csv(uploaded_file)  # Use pd.read_excel for Excel files
        st.dataframe(df)

        # Generate and download processed file
        if st.button(_("Generate Report")):
            # Perform data processing or analysis
            processed_df = df  # Example: No processing, just return original data

            # Download processed file (CSV)
            csv = processed_df.to_csv(index=False)
            st.download_button(label=_("Download CSV"), data=csv, file_name="processed_data.csv", mime="text/csv")


if __name__ == "__main__":
    main()
