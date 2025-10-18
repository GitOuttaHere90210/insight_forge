# password_gate.py
import streamlit as st
import subprocess
import sys

# -------------------------
# PASSWORD CHECK
# -------------------------

APP_PASSWORD = st.secrets["APP_PASSWORD"]

st.title("🔒 InsightForge")

password = st.text_input("Enter password to access InsightForge:", type="password")

if password != APP_PASSWORD:
    st.error("Access denied 🚫")
    st.stop()

st.success("Access granted ✅")
st.write("Loading app...")

# -------------------------
# LAUNCH MAIN APP
# -------------------------
subprocess.run([sys.executable, "app.py"])
