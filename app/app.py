import streamlit as st
import pandas as pd
st.set_page_config(
    page_title="Hello Joseph",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a DATA from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
 """
)
# Gestion du fichier CSV via session_state
if "data" not in st.session_state:
    st.session_state.data = None
with st.sidebar:
    st.header("Upload and Select Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file,sep=";")
        st.success("File successfully uploaded!")