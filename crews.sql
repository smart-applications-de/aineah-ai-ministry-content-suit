

import streamlit as st
import os
import io
import re
import json
import wave
import asyncio
import time
from datetime import datetime

# --- Dependency Imports ---
import docx
import markdown2
import pypdf
from google import genai
from google.generativeai import types
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
import streamlit.components.v1 as components

# ==============================================================================
## 1. Helper Functions
# ==============================================================================

def create_downloadable_docx(content):
    """Converts text content to a downloadable DOCX file in memory."""
    doc = docx.Document()
    for line in content.split('\n'):
        if line.startswith('# '):
            doc.add_heading(line[2:], level=1)
        elif line.startswith('## '):
            doc.add_heading(line[3:], level=2)
        elif line.startswith('### '):
            doc.add_heading(line[4:], level=3)
        else:
            doc.add_paragraph(line)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

def parse_json_from_text(text):
    """Safely extracts a JSON object from a string."""
    match = re.search(r'```json\n({.*?})\n```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            st.error("AI returned an invalid JSON structure. Please try again.")
            return None
    st.warning("Could not parse the AI's JSON response. The structure might be incorrect.")
    return None

@st.cache_data
def get_available_models(_api_key, task="generateContent"):
    """Fetches and caches the list of available Gemini models for a specific task."""
    if not _api_key: return []
    try:
        genai.configure(api_key=_api_key)
        models = [m.name for m in genai.list_models() if task in m.supported_generation_methods]
        return sorted(models)
    except Exception:
        st.sidebar.error("Error fetching models: Invalid API Key.")
        return []

def render_download_buttons(content, filename_base):
    """Renders a consistent set of download buttons for text-based content."""
    st.markdown("---")
    st.subheader("Export Your Content")
    col1, col2, col3, col4 = st.columns(4)
    html_content = markdown2.markdown(content)
    with col1:
        st.download_button("⬇️ DOCX", create_downloadable_docx(content), f"{filename_base}.docx")
    with col2:
        st.download_button("⬇️ Markdown", content.encode('utf-8'), f"{filename_base}.md")
    with col3:
        st.download_button("⬇️ HTML", html_content.encode('utf-8'), f"{filename_base}.html")
    with col4:
        st.download_button("⬇️ Text", content.encode('utf-8'), f"{filename_base}.txt")

class LanguageAcademyCrew:
    def __init__(self, model_name, native_language, target_language, level):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.native_language = native_language
        self.target_language = target_language
        self.level = level

    def run_guide_crew(self):
        # ... (Implementation from previous steps)
        pass

class LanguagePracticeCrew:
    def __init__(self, model_name, native_language, target_language, level):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.native_language = native_language
        self.target_language = target_language
        self.level = level

    def run(self, practice_type):
        # ... (Implementation from previous steps)
        pass

class LanguageListeningCrew:
    def __init__(self, model_name, native_language, target_language, level):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.native_language = native_language
        self.target_language = target_language
        self.level = level

    def run(self, topic):
        # ... (Implementation from previous steps)
        pass

class StreetEvangelismCrew:
    def __init__(self, model_name, language):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        os.environ["SERPER_API_KEY"] = st.session_state.get('serper_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.language = language

    def run_apologetics_crew(self, question, religion):
        agents = [
            Agent(role='Christian Apologist', goal=f"Formulate a biblically sound, logical, and respectful answer to the question: '{question}', specifically addressing the worldview of a {religion}.", backstory="You are an experienced Christian apologist with a deep understanding of world religions and philosophical objections to Christianity.", llm=self.llm, tools=[SerperDevTool()], verbose=True),
            Agent(role='Pastoral Counselor', goal=f"Refine the apologist's answer to be more pastoral, compassionate, and easy to understand for a layperson.", backstory="You are a Pentecostal pastor with decades of experience in street evangelism. You know how to communicate complex truths in a simple, heartfelt way.", llm=self.llm, verbose=True)
        ]
        task1 = Task(description=f"Develop a well-reasoned answer to the question '{question}' from a {religion} perspective.", agent=agents[0], expected_output="A structured, well-supported answer in markdown format.")
        task2 = Task(description="Review the apologist's answer. Rewrite it to be warmer in tone, add practical analogies, and conclude with a gentle, encouraging call to consider the message of Jesus.", agent=agents[1], context=[task1], expected_output="The final, polished, and pastoral answer in markdown format.", output_file="evangelism_answer.md")
        crew = Crew(agents=agents, tasks=[task1, task2], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("evangelism_answer.md", "r", encoding="utf-8") as f:
            return f.read()

    def run_guide_crew(self, topic=None):
        agents = [
            Agent(role='Veteran Evangelism Trainer', goal="Create a comprehensive, practical guide for beginners to street evangelism.", backstory="You have trained thousands of Christians for street ministry. You are practical, encouraging, and deeply biblical.", llm=self.llm, tools=[SerperDevTool()], verbose=True),
            Agent(role='Pastoral Mentor', goal="Enrich the guide with wisdom, encouragement, and specific biblical grounding.", backstory="You are a loving pastor with a heart for the Great Commission. You provide the 'why' behind the 'how-to'.", llm=self.llm, verbose=True)
        ]

        description = f"Create a beginner's guide to street evangelism in {self.language}. It must include sections on Getting Started, The Core Message, Do's and Don'ts, and Key Bible Verses."
        if topic:
            description += f"\nInclude a special section with talking points and verses for sharing about '{topic}'."

        task1 = Task(description=description, agent=agents[0], expected_output="A well-structured markdown guide with all the requested sections.")
        task2 = Task(description="Review the guide. Add a heartfelt introduction and conclusion. Ensure the tone is encouraging and empowering.", agent=agents[1], context=[task1], expected_output="The final, polished guide, ready for a new evangelist.", output_file="evangelism_guide.md")

        crew = Crew(agents=agents, tasks=[task1, task2], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("evangelism_guide.md", "r", encoding="utf-8") as f:
            return f.read()

    def run_faq_crew(self):
        agents = [
            Agent(role='Global Apologetics Researcher', goal="Identify the top 15 most frequently asked questions that people from all religious and non-religious backgrounds ask Christians.", backstory="You are an expert researcher who monitors global religious dialogues and debates to identify common points of contention and curiosity.", llm=self.llm, tools=[SerperDevTool()], verbose=True),
            Agent(role='Apologetics Response Team', goal="Provide a concise, biblically sound, and respectful answer for each of the top 15 questions.", backstory="You are a team of theologians and pastors skilled in providing clear, well-reasoned answers to challenging questions about the Christian faith.", llm=self.llm, verbose=True)
        ]
        task1 = Task(description="Research and compile a list of the top 15 questions asked of Christians globally.", agent=agents[0], expected_output="A numbered list of 15 frequently asked questions.")
        task2 = Task(description=f"For each of the 15 questions, provide a thoughtful and biblically referenced answer. The final output should be a well-formatted Q&A document in {self.language}.", agent=agents[1], context=[task1], expected_output="A complete markdown document with each question followed by its answer.", output_file="evangelism_faq.md")

        crew = Crew(agents=agents, tasks=[task1, task2], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("evangelism_faq.md", "r", encoding="utf-8") as f:
            return f.read()


def render_language_academy_page():
    st.title("🗣️ AI Language Academy")
    st.markdown("Your interactive hub for mastering a new language.")

    tab1, tab2, tab3 = st.tabs(["**Study Guide**", "**Practice & Exams**", "**Listening Practice**"])

    with tab1:
        st.header("Generate a Comprehensive Study Guide")
        if 'language_guide' not in st.session_state: st.session_state.language_guide = None
        available_models = get_available_models(st.session_state.get('gemini_key'))
        LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")

        with st.form("guide_form"):
            col1, col2 = st.columns(2)
            native_language = col1.text_input("Your Language", "English", key="guide_native")
            target_language = col2.text_input("Language to Learn", "French", key="guide_target")
            col3, col4 = st.columns(2)
            level = col3.selectbox("Select Your Level (CEFR)", ["A1", "A2", "B1", "B2", "C1", "C2"], key="guide_level")
            selected_model = col4.selectbox("Choose AI Model", available_models, key="guide_model") if available_models else None

            if st.form_submit_button("Generate Study Guide", use_container_width=True):
                if not all([native_language, target_language, selected_model]):
                    st.error("Please fill all fields and select a model.")
                else:
                    with st.spinner(f"Building your {level} {target_language} curriculum..."):
                        crew = LanguageAcademyCrew(selected_model, native_language, target_language, level)
                        st.session_state.language_guide = crew.run_guide_crew()

        if st.session_state.get('language_guide'):
            st.markdown("---")
            st.subheader(f"Your {target_language} ({level}) Study Guide")
            st.markdown(st.session_state.language_guide)
            render_download_buttons(st.session_state.language_guide, f"{target_language}_{level}_guide")

    with tab2:
        st.header("Create Custom Exercises or a Final Exam")
        if 'practice_material' not in st.session_state: st.session_state.practice_material = None
        available_models = get_available_models(st.session_state.get('gemini_key'))

        with st.form("practice_form"):
            col1, col2 = st.columns(2)
            native_language_prac = col1.text_input("Your Language", "English", key="prac_native")
            target_language_prac = col2.text_input("Language to Learn", "French", key="prac_target")

            col3, col4 = st.columns(2)
            level_prac = col3.selectbox("Select Level (CEFR)", ["A1", "A2", "B1", "B2", "C1", "C2"], key="prac_level")
            practice_type = col4.selectbox("Select Practice Type", ["Exercises", "Final Exam"])

            selected_model_prac = st.selectbox("Choose AI Model", available_models, key="prac_model") if available_models else None

            if st.form_submit_button(f"Generate {practice_type}", use_container_width=True):
                if not all([native_language_prac, target_language_prac, selected_model_prac]):
                    st.error("Please fill all fields and select a model.")
                else:
                    with st.spinner(f"The AI Exam Committee is preparing your {practice_type}..."):
                        crew = LanguagePracticeCrew(selected_model_prac, native_language_prac, target_language_prac, level_prac)
                        st.session_state.practice_material = crew.run(practice_type)

        if st.session_state.get('practice_material'):
            st.markdown("---")
            st.subheader(f"Your {target_language_prac} ({level_prac}) {practice_type}")
            st.markdown(st.session_state.practice_material)
            render_download_buttons(st.session_state.practice_material, f"{target_language_prac}_{level_prac}_{practice_type}")

    with tab3:
        st.header("Create a Custom Listening Exercise")
        if 'listening_material' not in st.session_state: st.session_state.listening_material = None
        available_models = get_available_models(st.session_state.get('gemini_key'))

        with st.form("listening_form"):
            col1, col2 = st.columns(2)
            native_language_listen = col1.text_input("Your Language", "English", key="listen_native")
            target_language_listen = col2.text_input("Language to Learn", "French", key="listen_target")

            col3, col4 = st.columns(2)
            level_listen = col3.selectbox("Select Level (CEFR)", ["A1", "A2", "B1", "B2", "C1", "C2"], key="listen_level")
            topic_listen = col4.text_input("Topic for the dialogue", "Ordering food at a restaurant")

            selected_model_listen = st.selectbox("Choose AI Model", available_models, key="listen_model") if available_models else None

            if st.form_submit_button("Generate Listening Practice", use_container_width=True):
                if not all([native_language_listen, target_language_listen, selected_model_listen, topic_listen]):
                    st.error("Please fill all fields and select a model.")
                else:
                    with st.spinner("The AI is creating your listening exercise..."):
                        crew = LanguageListeningCrew(selected_model_listen, native_language_listen, target_language_listen, level_listen)
                        st.session_state.listening_material = crew.run(topic_listen)

        if st.session_state.get('listening_material'):
            st.markdown("---")
            st.subheader(f"Your {target_language_listen} ({level_listen}) Listening Practice")
            st.markdown(st.session_state.listening_material)
            st.info("💡 **Pro-Tip:** Copy the transcript text and use the **Text-to-Audio** tool in the **AI Audio Suite** to generate the audio for this exercise!")
            render_download_buttons(st.session_state.listening_material, f"{target_language_listen}_{level_listen}_listening_practice")
def render_street_evangelism_page():
    st.title("✝️ Street Evangelism & Apologetics")
    st.markdown("Equipping you to fulfill the Great Commission (Matthew 28:17-20) with grace and truth.")

    tab1, tab2, tab3 = st.tabs(["**Answering Questions**", "**Beginner's Guide**", "**Top 15 FAQs**"])

    with tab1:
        st.header("Prepare Answers for Common Questions")
        available_models = get_available_models(st.session_state.get('gemini_key'))
        LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")
        RELIGIONS = ["Atheist", "Muslim", "Hindu", "Buddhist", "Agnostic", "Jewish", "Skeptic", "New Age", "Secular Humanist"]

        with st.form("evangelism_form"):
            language = st.selectbox("Language for the Answer:", LANGUAGES)
            religion = st.selectbox("Answering a question from a:", RELIGIONS)
            question = st.text_area("Enter the question you were asked:", placeholder="e.g., How can you prove Jesus is the Son of God?")
            selected_model = st.selectbox("Choose AI Model:", available_models) if available_models else None

            if st.form_submit_button("Generate Answer", use_container_width=True):
                if not all([question, selected_model]):
                    st.error("Please enter a question and select a model.")
                else:
                    with st.spinner("The AI ministry team is preparing a thoughtful answer..."):
                        crew = StreetEvangelismCrew(selected_model, language)
                        st.session_state.evangelism_content = crew.run_apologetics_crew(question, religion)

        if 'evangelism_content' in st.session_state and st.session_state.evangelism_content:
            st.markdown("---")
            st.subheader("Generated Response")
            st.markdown(st.session_state.evangelism_content)
            render_download_buttons(st.session_state.evangelism_content, "apologetics_answer")

    with tab2:
        st.header("Get a Practical Guide for Evangelism")
        available_models = get_available_models(st.session_state.get('gemini_key'))
        LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")

        with st.form("guide_form"):
            language = st.selectbox("Language for the Guide:", LANGUAGES, key="guide_lang")
            topic = st.text_input("Optional: Focus topic for the guide", placeholder="e.g., Salvation, The Holy Spirit, The Trinity")
            selected_model = st.selectbox("Choose AI Model:", available_models, key="guide_model") if available_models else None

            if st.form_submit_button("Generate Beginner's Guide", use_container_width=True):
                if not selected_model:
                    st.error("Please select a model.")
                else:
                    with st.spinner("The AI training team is creating your guide..."):
                        crew = StreetEvangelismCrew(selected_model, language)
                        st.session_state.evangelism_content = crew.run_guide_crew(topic if topic else None)

        if 'evangelism_content' in st.session_state and st.session_state.evangelism_content:
            st.markdown("---")
            st.subheader("Your Evangelism Guide")
            st.markdown(st.session_state.evangelism_content)
            render_download_buttons(st.session_state.evangelism_content, "evangelism_guide")

    with tab3:
        st.header("Top 15 Frequently Asked Questions with Answers")
        available_models = get_available_models(st.session_state.get('gemini_key'))
        LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")

        with st.form("faq_form"):
            language = st.selectbox("Language for FAQs:", LANGUAGES, key="faq_lang")
            selected_model = st.selectbox("Choose AI Model:", available_models, key="faq_model") if available_models else None

            if st.form_submit_button("Generate FAQs with Answers", use_container_width=True):
                if not selected_model:
                    st.error("Please select a model.")
                else:
                    with st.spinner("The AI Apologetics Team is compiling the FAQs..."):
                        crew = StreetEvangelismCrew(selected_model, language)
                        st.session_state.evangelism_content = crew.run_faq_crew()

        if 'evangelism_content' in st.session_state and st.session_state.evangelism_content:
            st.markdown("---")
            st.subheader("Top 15 FAQs for Christians")
            st.markdown(st.session_state.evangelism_content)
            render_download_buttons(st.session_state.evangelism_content, "top_15_faqs")
