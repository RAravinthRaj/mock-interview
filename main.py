import streamlit as st
from audio_recorder_streamlit import audio_recorder
from groq import Groq
from deepgram import Deepgram
from dotenv import load_dotenv
import tempfile
import time
import os

st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎤",
    layout="wide"
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not GROQ_API_KEY or not DEEPGRAM_API_KEY:
    st.error("❌ Missing API keys in .env file")
    st.stop()

# --------------------------------------------------
# CLIENTS
# --------------------------------------------------
GROQ_MODEL = "llama-3.1-8b-instant"
llm_client = Groq(api_key=GROQ_API_KEY)
dg_client = Deepgram(DEEPGRAM_API_KEY)

# --------------------------------------------------
# AI FUNCTIONS
# --------------------------------------------------
def generate_questions(role, experience, difficulty, topic, qtype):
    prompt = f"""
Generate exactly 5 {difficulty} {qtype} interview questions
for a {experience}-level {role}.
Focus on: {topic}
Return only a numbered list.
"""
    response = llm_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return [
        q.strip("0123456789. ").strip()
        for q in response.choices[0].message.content.split("\n")
        if q.strip()
    ]


def evaluate_interview(questions, answers):
    qa = ""
    for i, q in enumerate(questions):
        qa += f"\nQ{i+1}: {q}\nA{i+1}: {answers[i]}\n"

    prompt = f"""
Evaluate the interview below and provide:
- Score out of 10
- Strengths
- Weaknesses
- Improvements
- Final summary

Interview:
{qa}
"""
    response = llm_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content


def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(audio_bytes)
        path = f.name

    with open(path, "rb") as audio:
        response = dg_client.transcription.sync_prerecorded(
            {"buffer": audio, "mimetype": "audio/webm"},
            {"punctuate": True, "language": "en", "model": "nova"}
        )

    results = response.get("results", {})
    channels = results.get("channels", [])
    if not channels:
        return ""
    alternatives = channels[0].get("alternatives", [])
    if not alternatives:
        return ""
    return alternatives[0].get("transcript", "").strip()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "end_interview" not in st.session_state:
    st.session_state.end_interview = False

# 🔥 Per-question audio tracking
if "last_audio_processed" not in st.session_state:
    st.session_state.last_audio_processed = {}

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🤖 AI-Powered Interview Preparation Guide")
st.caption("Internship Project · Groq LLaMA-3.1 · Deepgram STT")
st.divider()

st.sidebar.header("Interview Setup")
role = st.sidebar.text_input("Job Role", "Data Scientist")
experience = st.sidebar.selectbox("Experience Level", ["Beginner", "Intermediate", "Expert"])
difficulty = st.sidebar.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
question_type = st.sidebar.selectbox(
    "Question Type",
    ["Behavioral", "Technical", "Case Study", "Situational", "Leadership"]
)
topic = st.sidebar.text_area("Focus Topics", "Python, Machine Learning")
use_voice = st.sidebar.checkbox("🎤 Answer using voice")

if st.sidebar.button("🚀 Start Interview"):
    st.session_state.questions = generate_questions(
        role, experience, difficulty, topic, question_type
    )
    st.session_state.current_q = 0
    st.session_state.answers = {}
    st.session_state.end_interview = False
    st.session_state.last_audio_processed = {}

# --------------------------------------------------
# QUESTION FLOW
# --------------------------------------------------
if st.session_state.questions and not st.session_state.end_interview:
    i = st.session_state.current_q
    total = len(st.session_state.questions)

    st.subheader(f"Question {i + 1} of {total}")
    st.write(st.session_state.questions[i])

    answer = st.text_area(
        "✍️ Your Answer",
        value=st.session_state.answers.get(i, ""),
        height=160
    )

    st.session_state.answers[i] = answer

    if use_voice:
        audio = audio_recorder(pause_threshold=2.0)

        if audio:
            audio_id = hash(audio)

            # ✅ Transcribe only once per question per recording
            if st.session_state.last_audio_processed.get(i) != audio_id:
                st.session_state.last_audio_processed[i] = audio_id

                with st.spinner("Transcribing voice..."):
                    transcript = transcribe_audio(audio)

                if transcript:
                    st.session_state.answers[i] = transcript
                    st.success("Voice transcribed — you can edit it ✍️")
                    st.rerun()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬅️ Previous") and i > 0:
            st.session_state.current_q -= 1
            st.rerun()

    with col2:
        if st.button("💾 Save & Next") and i < total - 1:
            st.session_state.current_q += 1
            st.rerun()

    with col3:
        if st.button("🛑 End Interview"):
            st.session_state.end_interview = True
            st.rerun()

# --------------------------------------------------
# FINAL EVALUATION
# --------------------------------------------------
if st.session_state.questions and st.session_state.end_interview:
    qs, ans = [], []
    for k, v in st.session_state.answers.items():
        qs.append(st.session_state.questions[k])
        ans.append(v)

    if ans:
        st.subheader("✅ Interview Evaluation")
        with st.spinner("Evaluating interview..."):
            time.sleep(1)
            st.markdown(evaluate_interview(qs, ans))

