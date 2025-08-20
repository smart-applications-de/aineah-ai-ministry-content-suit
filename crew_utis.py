
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
import pandas as pd
from main_v2 import  get_available_models, render_download_buttons, LANGUAGES

def parse_json_from_text(text):
    """Safely extracts a JSON object from a string."""
    match = re.search(r'```json\n({.*?})\n```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            st.error("AI returned an invalid JSON structure. Please try again.")
            return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        st.warning("Could not parse the AI's JSON response. It may not be a clean JSON object.")
        return None
class LanguageAcademyCrew:
    def __init__(self, model_name, native_language, target_language, level):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.native_language = native_language
        self.target_language = target_language
        self.level = level

    def run_guide_crew(self):
        agents = [
            Agent(role='Polyglot Curriculum Designer',
                  goal=f"Outline a 10-12 lesson structure for a comprehensive language guide for a {self.native_language} speaker learning {self.target_language} at the {self.level} level.",
                  backstory="You are an expert linguist who creates structured, effective learning paths that guide students from one proficiency level to the next.",
                  llm=self.llm, verbose=True),
            Agent(role='Language Content Creator',
                  goal=f"Create the detailed content for each lesson in the curriculum. For each lesson, provide a clear grammar explanation, a list of essential vocabulary with translations to {self.native_language}, and example sentences.",
                  backstory=f"You are a language professor specializing in {self.target_language}. You have a gift for explaining complex grammar rules simply and providing vocabulary that is immediately useful for learners.",
                  llm=self.llm, verbose=True),
            Agent(role='Language Coach & Pronunciation Guide',
                  goal=f"Enhance the language guide with practical learning tips, pronunciation guides for difficult sounds, and a 'Takeaway Summary' for each lesson. The tone should be encouraging and motivating.",
                  backstory="You are a popular language coach who helps thousands of students achieve fluency. You know the common pitfalls and provide practical, confidence-boosting advice to make learning stick.",
                  llm=self.llm, verbose=True),
            Agent(role='Senior Editor',
                  goal=f"Compile all the generated content into a single, cohesive, and beautifully formatted language guide in {self.native_language}.",
                  backstory="You are a meticulous editor who ensures the final output is clean, organized, and ready for the user.",
                  llm=self.llm, verbose=True)
        ]
        task_outline = Task(
            description=f"Create a 10-12 lesson curriculum outline for a {self.native_language} speaker learning {self.target_language} at level {self.level}. For A1, the first lesson must cover the Alphabet and Pronunciation. The output should be a list of lesson topics.",
            agent=agents[0], expected_output="A markdown list of 10-12 lesson titles.")
        task_content = Task(
            description="Based on the curriculum outline, write the detailed content for each lesson. Each lesson must include a 'Grammar Focus' section and a 'Vocabulary' table.",
            agent=agents[1], context=[task_outline],
            expected_output=f"A complete document with all lessons, each containing grammar explanations and vocabulary tables with translations into {self.native_language} and English.")
        task_enrich = Task(
            description="Review the lesson content and enrich each lesson. Add a 'Pronunciation Pointer' section, a 'Learning Tip', and a 'Takeaway Summary' at the end of each lesson.",
            agent=agents[2], context=[task_content],
            expected_output="The enriched document with all coaching tips and summaries added to each lesson.")
        task_compile = Task(
            description=f"Compile all the enriched content into a single, final study guide. Format it beautifully in markdown with a main title: '{self.target_language} Language Guide ({self.level})'.",
            agent=agents[3], context=[task_enrich],
            expected_output="The final, comprehensive study guide formatted in clear markdown.",
            output_file="language_guide.md")

        crew = Crew(agents=agents, tasks=[task_outline, task_content, task_enrich, task_compile],
                    process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("language_guide.md", "r", encoding="utf-8") as f:
            return f.read()


class LanguagePracticeCrew:
    def __init__(self, model_name, native_language, target_language, level):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.native_language = native_language
        self.target_language = target_language
        self.level = level

    def run(self, practice_type):
        agents = [
            Agent(role='Goethe-Institut Grammar Examiner',
                  goal=f"Create a set of challenging grammar exercises suitable for a {self.level} learner of {self.target_language}, adhering to Goethe-Institut standards.",
                  backstory=f"You are a strict but fair language professor who designs official Goethe-Zertifikat grammar sections. Your exercises are designed to test a student's true understanding of the material.",
                  llm=self.llm,
                  verbose=True),
            Agent(role='Goethe-Institut Reading Comprehension Specialist',
                  goal=f"Write a short essay in {self.target_language} and create a set of questions to test a {self.level} learner's understanding, following the Goethe-Institut exam format.",
                  backstory="You specialize in creating texts and questions for language certification exams like the Goethe-Zertifikat. Your passages are engaging and your questions are precise.",
                  llm=self.llm, verbose=True),
            Agent(role='Chief Editor and Answer Key Compiler',
                  goal=f"Compile all exercises and essays into a single, cohesive document. Then, create a separate, clear answer key for all questions.",
                  backstory="You are a meticulous editor for a language textbook publisher. You ensure that all materials are perfectly formatted, and that the answer key is clear and easy to follow.",
                  llm=self.llm, verbose=True)
        ]

        if practice_type == "Exercises":
            task_desc_grammar = f"Create 10-15 varied grammar exercises (e.g., fill-in-the-blank, sentence transformation) for a {self.level} learner of {self.target_language}, following Goethe-Institut standards. The instructions should be in {self.native_language}."
            task_desc_comprehension = f"Write one short text (approx. 150 words) in {self.target_language} and create 5 multiple-choice questions to test comprehension, following Goethe-Institut standards. Instructions should be in {self.native_language}."
            output_filename = "language_exercises.md"
        else:  # Final Exam
            task_desc_grammar = f"Create a 'Grammar Section' for a {self.level} final exam, following Goethe-Institut standards. It should contain 20-25 challenging questions covering a wide range of grammar topics suitable for this level. Instructions in {self.native_language}."
            task_desc_comprehension = f"Create a 'Reading Comprehension & Essay' section for a {self.level} final exam, following Goethe-Institut standards. Write one text (approx. 300 words) in {self.target_language}, followed by 5 comprehension questions. Then, add one essay prompt that requires a 150-word response. Instructions in {self.native_language}."
            output_filename = "language_final_exam.md"

        task_grammar = Task(description=task_desc_grammar, agent=agents[0],
                            expected_output="A well-structured markdown section with numbered grammar exercises.")
        task_comprehension = Task(description=task_desc_comprehension, agent=agents[1],
                                  expected_output="A markdown section containing a text passage and numbered comprehension questions.")
        task_compile = Task(
            description=f"Combine the grammar and comprehension sections into a single document titled '{practice_type}'. Then, create a separate section at the end titled 'Answer Key' with clear solutions for all exercises.",
            agent=agents[2], context=[task_grammar, task_comprehension],
            expected_output=f"A complete, beautifully formatted markdown document containing the full {practice_type} and a comprehensive answer key.",
            output_file=output_filename)

        crew = Crew(agents=agents, tasks=[task_grammar, task_comprehension, task_compile], process=Process.sequential,
                    verbose=True)
        crew.kickoff()
        with open(output_filename, "r", encoding="utf-8") as f:
            return f.read()


class LanguageListeningCrew:
    def __init__(self, model_name, native_language, target_language, level):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.native_language = native_language
        self.target_language = target_language
        self.level = level

    def run(self, topic):
        agents = [
            Agent(role='Dialogue Scriptwriter for Language Learning',
                  goal=f"Create a short, natural-sounding dialogue or monologue in {self.target_language} about {topic}, appropriate for a {self.level} learner.",
                  backstory="You are an experienced writer for language learning audio courses. You create content that is both educational and engaging, perfectly matching the specified CEFR level.",
                  llm=self.llm, verbose=True),
            Agent(role='Goethe-Institut Exam Designer',
                  goal=f"Create a set of listening comprehension questions based on a provided transcript that meet Goethe-Institut standards for the {self.level} level.",
                  backstory="You have years of experience designing official Goethe-Zertifikat exams. You know exactly how to formulate questions (e.g., multiple choice, true/false) that accurately test listening skills.",
                  llm=self.llm, verbose=True)
        ]
        task_script = Task(
            description=f"Write a short audio script (approx. 100-150 words for A1/A2, 200-250 for B1/B2, 300+ for C1/C2) in {self.target_language} about {topic}. The language must be natural and appropriate for the {self.level}.",
            agent=agents[0], expected_output="The full text of the audio script in markdown.", output_file="listening_transcript.md")
        task_questions = Task(
            description=f"Based on the provided audio script, create 5-7 listening comprehension questions that meet Goethe-Institut standards for level {self.level}. Include a mix of question types like multiple choice and true/false. Provide a separate answer key. The questions and instructions must be in {self.native_language}.",
            agent=agents[1], context=[task_script],
            expected_output=f"A audio script,  complete set of questions and a separate answer key in markdown in {self.target_language} . Well formatted in 3 separate sections: *** Audio script section, Question Section and Anwers sections.",
            output_file="listening_practice.md")

        crew = Crew(agents=agents, tasks=[task_script, task_questions], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("listening_practice.md", "r", encoding="utf-8") as f:
            return f.read()
class VocabularyCrew:
    def __init__(self, model_name, native_language, target_language, level, scope):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        self.llm = LLM(model=model_name, temperature=0.3, api_key=os.environ["GOOGLE_API_KEY"])
        self.native_language = native_language
        self.target_language = target_language
        self.level = level
        self.scope = scope

    def run(self):
        agents = [
            Agent(role='Expert Lexicographer', goal=f"Generate a list of at least 150  essential vocabulary words in {self.target_language} related to '{self.scope}' for a {self.level} learner.", backstory="You are a lexicographer who specializes in creating CEFR-leveled thematic vocabulary lists for language learners. Your word choices are always practical and relevant.", llm=self.llm, verbose=True),
            Agent(role='Professional Translator', goal=f"Translate the list of {self.target_language} words into {self.native_language} and English.", backstory=f"You are a professional translator fluent in {self.target_language}, {self.native_language}, and English. You ensure accurate and contextually appropriate translations.", llm=self.llm, verbose=True),
            Agent(role='Applied Linguist & Editor', goal=f"For each word, provide a simple explanation or an example sentence in {self.target_language}. Then, compile all information into a structured JSON object.", backstory="You are a linguist who excels at explaining vocabulary in a simple, contextual way for learners. You are also meticulous at structuring data for applications.", llm=self.llm, verbose=True)
        ]
        task_generate_words = Task(description=f"Create a list of at least 150 vocabulary words in {self.target_language} for a {self.level} learner, focusing on the topic of '{self.scope}'.", agent=agents[0], expected_output="A clean list of 100+ words in {self.target_language}.")
        task_translate = Task(description=f"Translate the provided list of {self.target_language} words into two columns: one for {self.native_language} and one for English.", agent=agents[1], context=[task_generate_words], expected_output="A three-column list of words: Target Language, Native Language, English.")
        task_compile = Task(description=f"""For each word in the translated list, add a fourth column with a simple explanation or example sentence in {self.target_language}. Finally, format the entire result into a single JSON object. The JSON must have a key 'vocabulary' which is an array of objects. Each object must have four keys: 'target_word', 'native_translation', 'english_translation', and 'explanation'. The final output MUST be ONLY the raw JSON object, without any markdown formatting.
        You MUST always give full list without ... or etc""", agent=agents[2], context=[task_translate], expected_output="A single JSON object containing the structured vocabulary list.", output_file="vocabulary_list.json")

        crew = Crew(agents=agents, tasks=[task_generate_words, task_translate, task_compile], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("vocabulary_list.json", "r", encoding="utf-8") as f:
            return parse_json_from_text(f.read())
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
        task1 = Task \
            (description=f"Develop a well-reasoned answer to the question '{question}' from a {religion} perspective.", agent=agents[0], expected_output="A structured, well-supported answer in markdown format.")
        task2 = Task \
            (description="Review the apologist's answer. Rewrite it to be warmer in tone, add practical analogies, and conclude with a gentle, encouraging call to consider the message of Jesus.", agent=agents[1], context=[task1], expected_output="The final, polished, and pastoral answer in markdown format.", output_file="evangelism_answer.md")
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
        task2 = Task \
            (description="Review the guide. Add a heartfelt introduction and conclusion. Ensure the tone is encouraging and empowering.", agent=agents[1], context=[task1], expected_output="The final, polished guide, ready for a new evangelist.", output_file="evangelism_guide.md")

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
        task2 = Task \
            (description=f"For each of the 15 questions, provide a thoughtful and biblically referenced answer. The final output should be a well-formatted Q&A document in {self.language}.", agent=agents[1], context=[task1], expected_output="A complete markdown document with each question followed by its answer.", output_file="evangelism_faq.md")

        crew = Crew(agents=agents, tasks=[task1, task2], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("evangelism_faq.md", "r", encoding="utf-8") as f:
            return f.read()


def render_language_academy_page():
    st.title("🗣️ AI Language Academy")
    st.markdown("Your interactive hub for mastering a new language.")
    AVAILABLE_MODELS = get_available_models(st.session_state.get('gemini_key'))
    #LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")
    SCOPE_OPTIONS = ["Home", "Work", "University", "School", "Hospital", "Restaurant", "Travel", "Health","Family","Church","Bible","Salvation",
                     "Jesus","Greetings","Food", "Animals","Universe","Music","News","Politic","Science","Grammar","phrases","proverbs","idoms","Birds","Professions"]


    tab1, tab2, tab3, tab4 = st.tabs([
        "**🎓 Study Guide**",
        "**✍️ Practice & Exams**",
        "**🎧 Listening Practice**",
        "**📚 Vocabulary Builder**"
    ])

    with tab1:
        st.header("Generate a Comprehensive Study Guide")
        if 'language_guide' not in st.session_state: st.session_state.language_guide = None
        available_models = get_available_models(st.session_state.get('gemini_key'))
        #LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")

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
                        try:
                            crew = LanguageAcademyCrew(selected_model, native_language, target_language, level)
                            st.session_state.language_guide = crew.run_guide_crew()
                        except Exception as error:
                            st.error(error)

        if st.session_state.get('language_guide'):
            st.markdown("---")
            st.subheader(f"Your {target_language} ({level}) Study Guide")
            st.markdown(st.session_state.language_guide)
            render_download_buttons(st.session_state.language_guide, f"{target_language}_{level}_guide")
            if st.button(f"🎧 Listen to this AI-Generated Language Guide"):
                st.session_state['text_for_audio'] = st.session_state.language_guide
                st.info("Go to the 'Audio Suite' page to generate the audio.")

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
            if st.button(f"🎧 Listen to this AI-Generated practice_material"):
                st.session_state['text_for_audio'] = st.session_state.practice_material
                st.info("Go to the 'Audio Suite' page to generate the audio.")

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
            st.info \
                ("💡 **Pro-Tip:** Copy the transcript text and use the **Text-to-Audio** tool in the **AI Audio Suite** to generate the audio for this exercise!")
            try:
                with open("listening_transcript.md", "r", encoding="utf-8") as f:
                    st.session_state['transcript'] =f.read()
                if  st.session_state['transcript'] :
                    st.markdown(st.session_state['transcript'])
            except Exception as err:
                st.error(err)

            render_download_buttons(st.session_state.listening_material, f"{target_language_listen}_{level_listen}_listening_practice")


 # ========================= TAB 4: Vocabulary Builder =========================
    with tab4:
        st.header("Build Your Thematic Vocabulary List")
        if 'vocabulary_list' not in st.session_state: st.session_state.vocabulary_list = None

        with st.form("vocabulary_form"):
            col1, col2 = st.columns(2)
            native_language_voc = col1.selectbox("Your Language", LANGUAGES, index=0, key="voc_native")
            target_language_voc = col2.selectbox("Language to Learn", LANGUAGES, index=2, key="voc_target")
            col3, col4 = st.columns(2)
            level_voc = col3.selectbox("Select Level (CEFR)", ["A1", "A2", "B1", "B2", "C1", "C2"], key="voc_level")
            scope_voc = col4.selectbox("Enter scope", SCOPE_OPTIONS, index=4, key="voc_scope")
            selected_model_voc = st.selectbox("Choose AI Model", AVAILABLE_MODELS, key="voc_model")

            if st.form_submit_button("Generate Vocabulary List", use_container_width=True, type="primary"):
                if not all([native_language_voc, target_language_voc, selected_model_voc, level_voc, scope_voc]):
                    st.error("Please fill all fields and select a model.")
                else:
                    with st.spinner(f"The AI linguistics team is compiling your vocabulary list for '{scope_voc}'..."):
                        crew = VocabularyCrew(selected_model_voc, native_language_voc, target_language_voc, level_voc, scope_voc)
                        st.session_state.vocabulary_list = crew.run()

        if st.session_state.get('vocabulary_list'):
            st.markdown("---")
            st.subheader(f"Your {target_language_voc} Vocabulary for '{scope_voc}' ({level_voc})")
            vocab_data = st.session_state.vocabulary_list.get('vocabulary', [])
            
            # --- LOGIC FIX: Check data structure before creating DataFrame ---
            if vocab_data:
                df = pd.DataFrame(vocab_data)
                if f"{native_language_voc}"=='English':
                    df.columns = [f"{target_language_voc} Word", f"{native_language_voc} Translation", "English Translation_1",
                                  f"Explanation in {target_language_voc}"]
                else:
                     df.columns = [f"{target_language_voc} Word", f"{native_language_voc} Translation", "English Translation",
                                  f"Explanation in {target_language_voc}"]

                st.dataframe(df)

                markdown_for_download = df.to_markdown(index=False)
                render_download_buttons(markdown_for_download, f"{target_language_voc}_{scope_voc}_vocabulary")
 
            else:
                st.warning("Could not display the vocabulary list in a table. Please check the raw output.")
                st.json(st.session_state.vocabulary_list)

def render_street_evangelism_page():
    st.title("✝️ Street Evangelism & Apologetics")
    st.markdown("Equipping you to fulfill the Great Commission (Matthew 28:17-20) with grace and truth.")

    tab1, tab2, tab3 = st.tabs(["**Answering Questions**", "**Beginner's Guide**", "**Top 15 FAQs**"])

    with tab1:
        st.header("Prepare Answers for Common Questions")
        available_models = get_available_models(st.session_state.get('gemini_key'))
       # LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")
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
        #LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")

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
       # LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")

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
