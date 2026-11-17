import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import urllib.parse

import streamlit as st

from utils.avatar import avatar_path
from utils.config import DEV_EMAIL, DEV_GITHUB, DEV_LINKEDIN, DEV_MAILTO, DEV_NAME, REPO_URL
from utils.disclaimer import show_full_disclaimer
from utils.layout import footer, setup

setup("Contact · BTC Econometrics Lab", "📞")
st.title("📞 Contact")

l, r = st.columns([1, 3])
l.image(avatar_path(), width=200)
with r:
    st.markdown(f"""
### {DEV_NAME}
Quantitative finance aspirant in **Bangalore, India** — M.Sc. Financial Engineering (WorldQuant University), M.Tech CSE (NSUT Delhi),
B.Tech IT (AKTU). Python · C++ · SQL · machine learning. Open to **full-time roles in quantitative research or financial engineering**.
""")
    b = st.columns(4)
    b[0].link_button("📧 Email", DEV_MAILTO)
    b[1].link_button("💼 LinkedIn", DEV_LINKEDIN)
    b[2].link_button("💻 GitHub", DEV_GITHUB)
    b[3].link_button("📦 This repo", REPO_URL)

st.header("Send feedback")
st.caption("This opens your own mail app with the message pre-filled — nothing is stored on this server.")
with st.form("feedback"):
    name = st.text_input("Your name")
    topic = st.selectbox("Topic", ["Feedback on the app", "Bug report", "Collaboration / role", "Question"])
    msg = st.text_area("Message", height=140)
    sent = st.form_submit_button("Prepare email")
if sent:
    if not msg.strip():
        st.warning("Please write a message first.")
    else:
        url = f"{DEV_MAILTO}?subject={urllib.parse.quote('BTC Econometrics Lab — ' + topic)}&body={urllib.parse.quote(f'From: {name}' + chr(10) + chr(10) + msg)}"
        st.success("Ready — click to open your mail app.")
        st.link_button("📧 Open email draft", url)

st.header("Disclaimer")
show_full_disclaimer()
footer()
