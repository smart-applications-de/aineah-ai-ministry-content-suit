# language_academy.py
# A self-contained Streamlit page for the AI Language Academy with all features fully implemented.

import streamlit as st
import os
import re
import json
import pandas as pd
from crewai import Agent, Task, Crew, Process, LLM
import io
from main import get_available_models, render_download_buttons, LANGUAGES, voice_names
import wave
st.session_state['SCOPES']=[
    "Personal Information & Greetings", "Bible", "Christian", "Greetings", "Politics", "Socia Media",
    "Music", "Film", "Jesus", "Religion", "School", "Climate", "Vacation/ Holiday", "Travel & adventure",
    "Restuarant", "Small Talk", "Hospital", "Hotel", "Science", "Art", "Nursing", "Home & Household", "Craftsmann",
    "Family & Friends", "Numbers, Dates, & Time", "Food & Drink",
    "At Home (Rooms & Furniture)", "Technology", "Film & Series", "Sport", "World & Wonders",
    "Daily Routines", "Clothing & Shopping", "Weather & Seasons", "The Body & Health", "Hobbies & Free Time",
    "Basic Travel & Directions",
    "Work & Professions", "Education & University", "Technology & The Internet", "Media & News",
    "Environment & Nature",
    "Culture & Traditions", "Politics & Society", "Feelings & Emotions", "Travel & Tourism (Advanced)",
    "Health & Fitness",
    "Business & Finance", "Science & Research", "Law & Justice", "Arts & Literature", "History & Archaeology",
    "Philosophy & Abstract Concepts"
]
st.session_state['GRAMMAR_TOPICS'] = [
                "Articles (Definite/Indefinite)", "Nouns (Gender/Plurals)", "Present Tense (Regular Verbs & Irregular Verbs)",
                "Basic Sentence Structure (SVO)",
                "Personal Pronouns", "Possessive Adjectives", "Prepositions of Place", "Question Formation",
                "The Verb 'to be'", "The Verb 'to have'",
                "Past Tenses (e.g., Simple Past, Perfect,Imperfect)", "Future Tenses", "Modal Verbs", "Reflexive Verbs",
                "Comparative & Superlative",
                "Relative Clauses", "Conditional Sentences (Type 1 & 2)", "Conjunctions", "The Passive Voice",
                "Subjunctive Mood", "Conditional Sentences (Type 3)", "Advanced Prepositions",
                "Complex Sentence Structures", "Idiomatic Expressions", "Participles as Adjectives"
            ]

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
def pcm_to_wav(pcm_data, channels=1, sample_width=2, sample_rate=24000):
    """Converts raw PCM data to a WAV file in memory."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    buffer.seek(0)
    return buffer.getvalue()


# ==============================================================================
## 2. AI Crew Classes
# ==============================================================================

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
            description="For **every single lesson title** provided in the curriculum outline from the previous step, you must write the detailed content. Each lesson must include a 'Grammar Focus' section and a 'Vocabulary' table.",
            agent=agents[1], context=[task_outline],
            expected_output=f"A complete document containing detailed content for **all 10-12 lessons** outlined previously, with each lesson containing grammar explanations and vocabulary tables with translations into {self.native_language} and English.")
        task_enrich = Task(
            description="Review the complete lesson content and enrich **every single lesson**. Add a 'Pronunciation Pointer' section, a 'Learning Tip', and a 'Takeaway Summary' at the end of each lesson.",
            agent=agents[2], context=[task_content],
            expected_output="The enriched document with coaching tips and summaries added to **all 10-12 lessons**.")
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

    def run(self, practice_type, grammar_topics=None):
        agents = [
            Agent(role='Goethe-Institut Grammar Examiner',
                  goal=f"Create a set of challenging grammar exercises suitable for a {self.level} learner of {self.target_language}, adhering to Goethe-Institut standards.",
                  backstory=f"You are a strict but fair language professor who designs official Goethe-Zertifikat grammar sections. Your exercises are designed to test a student's true understanding of the material.",
                  llm=self.llm, verbose=True),
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
            task_desc_grammar = f"Create 5 varied grammar exercises for EACH of the following topics: {', '.join(grammar_topics)}. The exercises should be suitable for a {self.level} learner of {self.target_language} and follow Goethe-Institut standards. The instructions should be in {self.native_language}."
            task_desc_comprehension = f"Write one short text (approx. 150 words) in {self.target_language} and create 5 multiple-choice questions to test comprehension, following Goethe-Institut standards. Instructions should be in {self.native_language}."
            output_filename = "language_exercises.md"
        else:  # Final Exam
            task_desc_grammar = f"Create a 'Grammar Section' for a {self.level} final exam, following Goethe-Institut standards. It should contain 20-25 challenging questions covering a wide range of grammar topics suitable for this level. Instructions in {self.native_language}."
            task_desc_comprehension = f"Create a 'Reading Comprehension & Essay' section for a {self.level} final exam, following Goethe-Institut standards. Write one text (approx. 300 words) in {self.target_language}, followed by 5 comprehension questions and one essay prompt. Instructions in {self.native_language}."
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
            Agent(role='Dialogue Scriptwriter for Language Learning', goal=f"Create a short, natural-sounding two-speaker dialogue in {self.target_language} about {topic}, appropriate for a {self.level} learner. The speakers should be named Speaker A and Speaker B.", backstory="You are an experienced writer for language learning audio courses. You create content that is both educational and engaging, perfectly matching the specified CEFR level.", llm=self.llm, verbose=True),
            Agent(role='Goethe-Institut Exam Designer', goal=f"Create a set of listening comprehension questions based on a provided transcript that meet Goethe-Institut standards for the {self.level} level.", backstory="You have years of experience designing official Goethe-Zertifikat exams. You know exactly how to formulate questions that accurately test listening skills.", llm=self.llm, verbose=True),
            Agent(role='Content Compiler', goal="Combine the transcript and questions into a single, structured JSON object.", backstory="You are a meticulous content organizer, ensuring data is perfectly structured for application use.", llm=self.llm, verbose=True)
        ]
        task_script = Task(description=f"Write a short audio script (approx. 150-200 words for A1/A2, 250-300 for B1/B2 and and 400+ for C1/C2) in {self.target_language}   about {topic}. It must be a dialogue between 'Speaker A' and 'Speaker B'.", agent=agents[0], expected_output="The full text of the audio script in markdown, with speaker labels.")
        task_questions = Task(description=f"Based on the provided audio script, create 10-12  listening comprehension questions that meet Goethe-Institut standards for level {self.level}. Provide a separate answer key. The questions and instructions must be in {self.target_language}.", agent=agents[1], context=[task_script], expected_output="A complete set of questions and a separate answer key in markdown.")
        task_compile = Task(description="Combine the audio transcript and the questions/answers into a single JSON object. The JSON must have two keys: 'transcript' and 'questions'. The final output MUST be ONLY the raw JSON object.", agent=agents[2], context=[task_script, task_questions], expected_output="A single, clean JSON object with 'transcript' and 'questions' keys.", output_file="listening_practice.json")

        crew = Crew(agents=agents, tasks=[task_script, task_questions, task_compile], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("listening_practice.json", "r", encoding="utf-8") as f:
            return parse_json_from_text(f.read())

    @staticmethod
    def generate_audio(tts_model_name, transcript, speaker_configs):
        try:
            from main import  genai,types, pcm_to_wav
            from google import genai as gen
            from google.genai import types
            client = gen.Client(api_key=st.session_state.get('gemini_key', ''))
            response = client.models.generate_content(
                model=tts_model_name, contents=[transcript],
                config=types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=types.SpeechConfig(multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(speaker_voice_configs=speaker_configs)))
            )
            return response.candidates[0].content.parts[0].inline_data.data
        except Exception as e:
            st.error(f"Audio generation failed: {e}"); return None


class VocabularyCrew:
    def __init__(self, model_name, native_language, target_language, level, scope):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.native_language = native_language
        self.target_language = target_language
        self.level = level
        self.scope = scope

    def run(self):
        agents = [
            Agent(role='Expert Lexicographer',
                  goal=f"Generate a list of at least 100 essential vocabulary words in {self.target_language} related to '{self.scope}' for a {self.level} learner.",
                  backstory="You are a lexicographer who specializes in creating CEFR-leveled thematic vocabulary lists for language learners. Your word choices are always practical and relevant.",
                  llm=self.llm, verbose=True),
            Agent(role='Professional Translator',
                  goal=f"Translate the list of {self.target_language} words into {self.native_language} and English.",
                  backstory=f"You are a professional translator fluent in {self.target_language}, {self.native_language}, and English. You ensure accurate and contextually appropriate translations.",
                  llm=self.llm, verbose=True),
            Agent(role='Applied Linguist & Editor',
                  goal=f"For each word, provide a simple explanation or an example sentence in {self.target_language}. Then, compile all information into a structured JSON object.",
                  backstory="You are a linguist who excels at explaining vocabulary in a simple, contextual way for learners. You are also meticulous at structuring data for applications.",
                  llm=self.llm, verbose=True)
        ]
        task_generate_words = Task(
            description=f"Create a list of at least 100 vocabulary words in {self.target_language} for a {self.level} learner, focusing on the topic of '{self.scope}'.",
            agent=agents[0], expected_output="A clean list of 100+ words in {self.target_language}.")
        task_translate = Task(
            description=f"Translate the provided list of {self.target_language} words into two columns: one for {self.native_language} and one for English.",
            agent=agents[1], context=[task_generate_words],
            expected_output="A three-column list of words: Target Language, Native Language, English.")
        task_compile = Task(
            description=f"For each word in the translated list, add a fourth column with a simple explanation or example sentence in {self.target_language}. Finally, format the entire result into a single JSON object. The JSON must have a key 'vocabulary' which is an array of objects. Each object must have four keys: 'target_word', 'native_translation', 'english_translation', and 'explanation'. The final output MUST be ONLY the raw JSON object, without any markdown formatting.",
            agent=agents[2], context=[task_translate],
            expected_output="A single JSON object containing the structured vocabulary list.",
            output_file="vocabulary_list.json")

        crew = Crew(agents=agents, tasks=[task_generate_words, task_translate, task_compile],
                    process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("vocabulary_list.json", "r", encoding="utf-8") as f:
            return parse_json_from_text(f.read())


class GrammarCrew:
    def __init__(self, model_name, native_language, target_language, level, topics):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.native_language = native_language
        self.target_language = target_language
        self.level = level
        self.topics = topics

    def run(self):
        agents = [
            Agent(role='Linguistics Professor',
                  goal=f"Provide a detailed, technically accurate explanation of the following grammar topics in {self.target_language} for a {self.level} learner: {', '.join(self.topics)}.",
                  backstory=f"You are a university professor of linguistics specializing in the grammar of {self.target_language}. Your explanations are precise, comprehensive, and academically rigorous.",
                  llm=self.llm, verbose=True),
            Agent(role='Expert Language Teacher',
                  goal=f"Simplify the professor's explanation and make it practical for a {self.level} learner. Translate the explanations into {self.native_language} and provide clear examples.",
                  backstory=f"You are a passionate and experienced language teacher. You excel at breaking down complex grammar into simple rules, creating memorable examples, and providing practical takeaways that students can immediately use.",
                  llm=self.llm, verbose=True)
        ]
        task_explain = Task(
            description=f"Generate a detailed academic explanation of these grammar topics: {', '.join(self.topics)} for a {self.level} learner of {self.target_language}.",
            agent=agents[0],
            expected_output="A detailed, technically correct explanation of each grammar topic in markdown format.")
        task_simplify = Task(
            description=f"Take the academic explanation and simplify it. For each topic, provide: 1. A simple rule explanation in {self.native_language}. 2. At least 3 clear example sentences in {self.target_language} with translations to {self.native_language}. 3. A 'Key Takeaway' summary.",
            agent=agents[1], context=[task_explain],
            expected_output="A final, easy-to-understand guide in markdown, with all explanations in the user's native language and examples in the target language.",
            output_file="grammar_guide.md")

        crew = Crew(agents=agents, tasks=[task_explain, task_simplify], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("grammar_guide.md", "r", encoding="utf-8") as f:
            return f.read()


class LanguageComprehensionCrew:
    def __init__(self, model_name, native_language, target_language, level, scope):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.native_language = native_language
        self.target_language = target_language
        self.level = level
        self.scope = scope

    def run(self):
        agents = [
            Agent(
                role='Educational Content Writer',
                goal=f"Write an engaging and level-appropriate text passage in {self.target_language} about '{self.scope}' for a {self.level} learner.",
                backstory=f"You are a skilled writer who creates educational content for language learners. You know how to adjust vocabulary, sentence structure, and complexity to perfectly match CEFR levels.",
                llm=self.llm,
                verbose=True
            ),
            Agent(
                role='Goethe-Institut Exam Designer',
                goal=f"Create a set of reading comprehension questions based on a provided text that meet Goethe-Institut standards for the {self.level} level.",
                backstory="You have years of experience designing official Goethe-Zertifikat exams. You know exactly how to formulate questions that accurately test reading comprehension skills.",
                llm=self.llm,
                verbose=True
            ),
            Agent(
                role='Chief Editor and Answer Key Compiler',
                goal="Compile the text, questions, and a separate, clear answer key into a single, cohesive document.",
                backstory="You are a meticulous editor for a language textbook publisher. You ensure that all materials are perfectly formatted and that the answer key is clear and easy to follow.",
                llm=self.llm,
                verbose=True
            )
        ]

        task_write_text = Task(
            description=f"Write a text passage of appropriate length (A1/A2: ~ 200 words, B1/B2: ~400 words, C1/C2: ~500 words) in {self.target_language} about '{self.scope}' for a {self.level} learner.",
            agent=agents[0],
            expected_output="A well-written text passage in markdown format."
        )
        task_create_questions = Task(
            description=f"Based on the provided text, create 10-12 reading comprehension questions that meet Goethe-Institut standards for level {self.level}. Include a mix of question types. The questions and instructions must be in {self.target_language}.",
            agent=agents[1],
            context=[task_write_text],
            expected_output="A complete set of questions and a separate answer key in markdown."
        )
        task_compile = Task(
            description=f"Combine the text passage and the questions into a single document. Then, create a separate section at the end titled 'Answer Key' with clear solutions for all questions in {self.target_language}.",
            agent=agents[2],
            context=[task_write_text, task_create_questions],
            expected_output=f"A complete, beautifully formatted markdown document containing the full reading comprehension exercise and a comprehensive answer key in {self.target_language}.",
            output_file="reading_comprehension.md"
        )

        crew = Crew(
            agents=agents,
            tasks=[task_write_text, task_create_questions, task_compile],
            process=Process.sequential,
            verbose=True
        )

        crew.kickoff()
        with open("reading_comprehension.md", "r", encoding="utf-8") as f:
            return f.read()


class VerbConjugatorCrew:
    def __init__(self, model_name, native_language, target_language, level):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.native_language = native_language
        self.target_language = target_language
        self.level = level

    def run(self, verb=None):
        agents = [
            Agent(role='Linguistics Professor',
                  goal=f"Provide a detailed, technically accurate breakdown of verb conjugations and tenses in {self.target_language}.",
                  backstory=f"You are a university professor of linguistics specializing in the morphology and syntax of {self.target_language}. Your knowledge is comprehensive and precise.",
                  llm=self.llm, verbose=True),
            Agent(role='Expert Language Teacher',
                  goal=f"Simplify the professor's breakdown and make it practical for a {self.level} learner. Translate explanations into {self.native_language} and provide clear examples.",
                  backstory=f"You are a passionate and experienced language teacher. You excel at making complex verb conjugations understandable with clear examples and practical takeaways.",
                  llm=self.llm, verbose=True)
        ]

        if verb:
            task_desc_prof = f"Provide a complete breakdown of the verb '{verb}' in {self.target_language}. Include its present tense conjugation, past tense (perfect/imperfect), and future tense."
            task_desc_teacher = f"Take the breakdown of '{verb}'. For each tense, provide: 1. A simple explanation of its use in {self.native_language}. 2. Five clear example sentences in {self.target_language} with translations. 3. A 'Key Takeaway' summary."
            output_filename = f"verb_guide_{verb}.md"
        else:
            task_desc_prof = f"Identify and provide a complete breakdown for the 10 most common regular verbs and 10 most common irregular verbs in {self.target_language} for a {self.level} learner. For each verb, include its present, past (perfect/imperfect), and future tense conjugations."
            task_desc_teacher = f"Take the list of 20 verbs. For each verb, provide: 1. A simple explanation of its meaning in {self.native_language}. 2. Two clear example sentences in {self.target_language} with translations."
            output_filename = "common_verbs_guide.md"

        task_analyze = Task(
            description=task_desc_prof,
            agent=agents[0],
            expected_output="A detailed, technically correct breakdown of all requested verbs and their tenses in markdown format."
        )
        task_simplify = Task(
            description=task_desc_teacher,
            agent=agents[1],
            context=[task_analyze],
            expected_output="A final, easy-to-understand guide in markdown, with all explanations in the user's native language and examples in the target language.",
            output_file=output_filename
        )

        crew = Crew(
            agents=agents,
            tasks=[task_analyze, task_simplify],
            process=Process.sequential,
            verbose=True
        )

        crew.kickoff()
        with open(output_filename, "r", encoding="utf-8") as f:
            return f.read()


class DictionaryCrew:
    def __init__(self, model_name, target_language):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.target_language = target_language

    def run(self, word):
        if self.target_language.lower() == 'german':
            backstory = "You are an expert lexicographer from Duden, the authoritative dictionary of the German language. You provide comprehensive and precise definitions."
        elif self.target_language.lower() == 'english':
            backstory = "You are an expert lexicographer from the Oxford English Dictionary. You provide comprehensive and precise definitions with a rich historical context."
        else:
            backstory = f"You are an expert lexicographer for the {self.target_language} language. You provide comprehensive and precise definitions."

        agent = Agent(
            role='Expert Lexicographer',
            goal=f"Provide a complete dictionary entry for the word '{word}' in {self.target_language}.",
            backstory=backstory,
            llm=self.llm,
            verbose=True
        )

        task = Task(
            description=f"Create a comprehensive dictionary entry for the word '{word}' in {self.target_language}. The entry must include: Part of Speech, Phonetic Transcription (if possible), all common meanings/definitions, an example sentence for each meaning, a list of synonyms and antonyms, and the word's etymology (origin).The output MUST be in the language: {self.target_language}. and in English if the {self.target_language} is NOT English ",
            agent=agent,
            expected_output=f"A complete, well-structured markdown document for the dictionary entry in {self.target_language}.",
            output_file="dictionary_entry.md"
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        crew.kickoff()
        with open("dictionary_entry.md", "r", encoding="utf-8") as f:
            return f.read()


# ==============================================================================
## 3. Page Rendering Function
# ==============================================================================




def render_language_academy_page():
    st.title("🗣️ AI Language Academy")
    st.markdown("Your interactive hub for mastering a new language.")

    tab_titles = [
        "**Study Guide**",
        "**Vocabulary Builder**",
        "**Grammar Deep Dive**",
        "**Verb Deep Dive**",
        "**Listening Practice**",
        "**Reading Comprehension**",
        "**Practice & Exams**",
        "**Dictionary**"
    ]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_titles)

    with tab1:
        st.header("Generate a Comprehensive Study Guide")
        if 'language_guide' not in st.session_state: st.session_state.language_guide = None
        try:
            available_models = get_available_models(st.session_state.get('gemini_key'))
            #LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")

            with st.form("guide_form"):
                col1, col2 = st.columns(2)
                native_language = col1.text_input("Your Language", "English", key="guide_native")
                target_language = col2.text_input("Language to Learn", "French", key="guide_target")
                col3, col4 = st.columns(2)
                level = col3.selectbox("Select Your Level (CEFR)", ["A1", "A2", "B1", "B2", "C1", "C2"], key="guide_level")
                selected_model = col4.selectbox("Choose AI Model", available_models,
                                                key="guide_model") if available_models else None

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
        except Exception as err1:
            st.error(err1)

    try:
        with tab2:
            st.header("Build Your Thematic Vocabulary List")
            if 'vocabulary_list' not in st.session_state: st.session_state.vocabulary_list = None
            available_models = get_available_models(st.session_state.get('gemini_key'))
            #LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")



            with st.form("vocab_form"):
                col1, col2 = st.columns(2)
                native_language = col1.text_input("Your Language", "English", key="vocab_native")
                target_language = col2.text_input("Language to Learn", "French", key="vocab_target")

                col3, col4 = st.columns(2)
                level = col3.selectbox("Select Your Level (CEFR)", ["A1", "A2", "B1", "B2", "C1", "C2"], key="vocab_level")
                scope = col4.selectbox("Select a Vocabulary Scope", st.session_state['SCOPES'])

                selected_model = st.selectbox("Choose AI Model", available_models,
                                              key="vocab_model") if available_models else None

                if st.form_submit_button("Generate Vocabulary List", use_container_width=True):
                    if not all([native_language, target_language, selected_model]):
                        st.error("Please fill all fields and select a model.")
                    else:
                        with st.spinner(f"The AI linguistics team is compiling your vocabulary list for '{scope}'..."):
                            crew = VocabularyCrew(selected_model, native_language, target_language, level, scope)
                            st.session_state.vocabulary_list = crew.run()

            if st.session_state.get('vocabulary_list'):
                st.markdown("---")
                st.subheader(f"Your {target_language} Vocabulary for '{scope}' ({level})")

                vocab_data = st.session_state.vocabulary_list.get('vocabulary', [])
                if vocab_data:
                    df = pd.DataFrame(vocab_data)
                    if f"{native_language}"=='English':
                        df.columns = [f"{target_language} Word", f"{native_language} Translation", "English Translation_1",
                                      f"Explanation in {target_language}"]
                    else:
                         df.columns = [f"{target_language} Word", f"{native_language} Translation", "English Translation",
                                      f"Explanation in {target_language}"]

                    st.dataframe(df)

                    markdown_for_download = df.to_markdown(index=False)
                    render_download_buttons(markdown_for_download, f"{target_language}_{scope}_vocabulary")
                else:
                    st.warning("Could not display the vocabulary list in a table. Please check the raw output.")
                    st.json(st.session_state.vocabulary_list)
    except Exception as erro2:
        st.error(erro2)

    try:
        with tab3:
            st.header("Master Specific Grammar Topics")
            if 'grammar_guide' not in st.session_state: st.session_state.grammar_guide = None
            available_models = get_available_models(st.session_state.get('gemini_key'))
            #LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")

            with st.form("grammar_form"):
                col1, col2 = st.columns(2)
                native_language = col1.text_input("Your Language", "English", key="gram_native")
                target_language = col2.text_input("Language to Learn", "German", key="gram_target")

                level = st.selectbox("Select Your Proficiency Level (CEFR)", ["A1", "A2", "B1", "B2", "C1", "C2"],
                                     key="gram_level")
                selected_topics = st.multiselect("Select Grammar Topics to Study", st.session_state['GRAMMAR_TOPICS'],
                                                 default=["Articles (Definite/Indefinite)",
                                                          "Present Tense (Regular Verbs & Irregular Verbs)"])

                selected_model = st.selectbox("Choose AI Model", available_models,
                                              key="gram_model") if available_models else None

                if st.form_submit_button("Generate Grammar Guide", use_container_width=True):
                    if not all([native_language, target_language, selected_model, selected_topics]):
                        st.error("Please fill all fields and select at least one grammar topic.")
                    else:
                        with st.spinner(f"The AI grammar experts are creating your guide..."):
                            crew = GrammarCrew(selected_model, native_language, target_language, level, selected_topics)
                            st.session_state.grammar_guide = crew.run()

            if st.session_state.get('grammar_guide'):
                st.markdown("---")
                st.subheader(f"Your {target_language} Grammar Guide ({level})")
                st.markdown(st.session_state.grammar_guide)
                render_download_buttons(st.session_state.grammar_guide, f"{target_language}_grammar_guide")
    except Exception as error3:
        st.error(error3)

    with tab4:
        try:
            st.header("Master Verb Conjugations and Tenses")
            if 'verb_guide' not in st.session_state: st.session_state.verb_guide = None
            available_models = get_available_models(st.session_state.get('gemini_key'))
            #LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")

            with st.form("verb_form"):
                col1, col2 = st.columns(2)
                native_language = col1.text_input("Your Language", "English", key="verb_native")
                target_language = col2.text_input("Language to Learn", "French", key="verb_target")

                level = st.selectbox("Select Your Proficiency Level (CEFR)", ["A1", "A2", "B1", "B2", "C1", "C2"],
                                     key="verb_level")
                verb_input = st.text_input("Enter a specific verb to analyze (optional)",
                                           placeholder="e.g., finir, machen, to be")

                selected_model = st.selectbox("Choose AI Model", available_models,
                                              key="verb_model") if available_models else None

                if st.form_submit_button("Generate Verb Guide", use_container_width=True):
                    if not all([native_language, target_language, selected_model]):
                        st.error("Please fill all fields and select a model.")
                    else:
                        with st.spinner(f"The AI linguistics team is preparing your verb guide..."):
                            crew = VerbConjugatorCrew(selected_model, native_language, target_language, level)
                            st.session_state.verb_guide = crew.run(verb_input if verb_input else None)

            if st.session_state.get('verb_guide'):
                st.markdown("---")
                st.subheader(f"Your {target_language} Verb Guide")
                st.markdown(st.session_state.verb_guide)
                render_download_buttons(st.session_state.verb_guide, f"{target_language}_verb_guide")
        except Exception as error4:
            st.error(error4)

    with tab5:
        try:
            st.header("Create a Custom Listening Exercise")
            from google import genai as gen
            from google.genai import types
            if 'listening_material' not in st.session_state: st.session_state.listening_material = None
            available_models = get_available_models(st.session_state.get('gemini_key'))
            tts_models = get_available_models(st.session_state.get('gemini_key'), task="text-to-speech")
            voices = voice_names
            SCOPES = [
                "Personal Information & Greetings", "Family & Friends", "Food & Drink", "At Home",
                "Daily Routines", "Shopping", "Weather & Seasons", "Health", "Hobbies & Free Time",
                "Travel & Directions",
                "Work & Professions", "Education & University", "Technology & The Internet", "Media & News"
            ]

            with st.form("listening_form"):
                st.subheader("1. Generate Transcript & Questions")
                col1, col2 = st.columns(2)
                native_language_listen = col1.text_input("Your Language", "English", key="listen_native")
                target_language_listen = col2.text_input("Language to Learn", "French", key="listen_target")

                col3, col4 = st.columns(2)
                level_listen = col3.selectbox("Select Level (CEFR)", ["A1", "A2", "B1", "B2", "C1", "C2"],
                                              key="listen_level")

                use_scope = st.radio("Choose topic source:", ("Select from a list", "Enter a custom topic"),
                                     key="topic_source")
                if use_scope == "Select from a list":
                    topic_listen = st.selectbox("Select a Topic Scope", st.session_state['SCOPES'])
                else:
                    topic_listen = st.text_input("Enter a custom topic for the dialogue", "Ordering food at a restaurant")

                selected_model_listen = st.selectbox("Choose AI Model for Script Writing", available_models,
                                                     key="listen_model") if available_models else None

                if st.form_submit_button("Generate Listening Practice", use_container_width=True):
                    if not all([native_language_listen, target_language_listen, selected_model_listen, topic_listen]):
                        st.error("Please fill all fields and select a model.")
                    else:
                        with st.spinner("The AI is creating your listening exercise..."):
                            crew = LanguageListeningCrew(selected_model_listen, native_language_listen,
                                                         target_language_listen, level_listen)
                            st.session_state.listening_material = crew.run(topic_listen)

            if st.session_state.get('listening_material'):
                st.markdown("---")
                st.subheader(f"Your {target_language_listen} ({level_listen}) Listening Practice")

                transcript = st.session_state.listening_material.get('transcript', 'No transcript generated.')
                questions = st.session_state.listening_material.get('questions', 'No questions generated.')

                st.text_area("Audio Transcript (Copy this to generate audio)", transcript, height=150,
                             key="transcript_text")
                st.json(questions)

                st.markdown("---")
                st.subheader("2. Generate Audio for Transcript")
                with st.form("audio_generation_form"):
                    col1, col2 = st.columns(2)
                    speaker_a_name = col1.text_input("Name for Speaker A", "Speaker A")
                    speaker_a_voice = col2.selectbox(f"Voice for {speaker_a_name}", voices, index=0)

                    col3, col4 = st.columns(2)
                    speaker_b_name = col3.text_input("Name for Speaker B", "Speaker B")
                    speaker_b_voice = col4.selectbox(f"Voice for {speaker_b_name}", voices, index=1)

                    tts_model = st.selectbox("Choose Audio AI Model", tts_models) if tts_models else None

                    if st.form_submit_button("Generate Audio", use_container_width=True):
                        if not tts_model:
                            st.error("Please select an audio model.")
                        else:
                            with st.spinner("🎙️ The AI is recording your audio..."):
                                # Replace generic speaker names in transcript with user-defined names for TTS
                                try:
                                    final_transcript = transcript.replace("Speaker A:", f"{speaker_a_name}:").replace(
                                        "Speaker B:", f"{speaker_b_name}:")

                                    speaker_configs = [
                                        types.SpeakerVoiceConfig(speaker=speaker_a_name, voice_config=types.VoiceConfig(
                                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=speaker_a_voice))),
                                        types.SpeakerVoiceConfig(speaker=speaker_b_name, voice_config=types.VoiceConfig(
                                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=speaker_b_voice)))
                                    ]
                                    st.session_state['audio_data_l']= LanguageListeningCrew.generate_audio(tts_model, final_transcript,
                                                                                      speaker_configs)
                                except Exception as error:
                                    st.error(error)
                    if st.session_state.get('audio_data_l') and  st.session_state.listening_material.get('transcript'):
                        st.success("Audio Generated!")
                        wav_bytes = pcm_to_wav(st.session_state.get('audio_data_l'), channels=1, sample_width=2,
                                               sample_rate=24000)
                        st.audio(wav_bytes, format='audio/wav')

        except Exception as error5:
            st.error(error5)

    with tab6:
        st.header("Improve Your Reading Comprehension")
        if 'comprehension_material' not in st.session_state: st.session_state.comprehension_material = None
        available_models = get_available_models(st.session_state.get('gemini_key'))
        LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")

        try:
            with st.form("comprehension_form"):
                col1, col2 = st.columns(2)
                native_language = col1.text_input("Your Language", "English", key="comp_native")
                target_language = col2.text_input("Language to Learn", "German", key="comp_target")

                col3, col4 = st.columns(2)
                level = col3.selectbox("Select Your Level (CEFR)", ["A1", "A2", "B1", "B2", "C1", "C2"], key="comp_level")
                scope = col4.selectbox("Select a Topic for the Text", st.session_state['SCOPES'])

                selected_model = st.selectbox("Choose AI Model", available_models,
                                              key="comp_model") if available_models else None

                if st.form_submit_button("Generate Reading Exercise", use_container_width=True):
                    if not all([native_language, target_language, selected_model]):
                        st.error("Please fill all fields and select a model.")
                    else:
                        with st.spinner(f"The AI is writing your reading comprehension exercise..."):
                            crew = LanguageComprehensionCrew(selected_model, native_language, target_language, level, scope)
                            st.session_state.comprehension_material = crew.run()

            if st.session_state.get('comprehension_material'):
                st.markdown("---")
                st.subheader(f"Your {target_language} Reading Comprehension Exercise")
                st.markdown(st.session_state.comprehension_material)
                render_download_buttons(st.session_state.comprehension_material,
                                        f"{target_language}_reading_{scope.replace(' ', '_').lower()}")
        except Exception as error6:
            st.error(error6)

    with tab7:
        try:
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

                # REFINED: Conditional UI for grammar topic selection
                grammar_topics_selection = []
                if practice_type == "Exercises":
                    grammar_topics_selection = st.multiselect("Select Grammar Topics for Exercises", st.session_state['GRAMMAR_TOPICS'],
                                                              default=["Present Tense (Regular Verbs & Irregular Verbs)"])

                selected_model_prac = st.selectbox("Choose AI Model", available_models,
                                                   key="prac_model") if available_models else None

                if st.form_submit_button(f"Generate {practice_type}", use_container_width=True):
                    if not all([native_language_prac, target_language_prac, selected_model_prac]):
                        st.error("Please fill all fields and select a model.")
                    elif practice_type == "Exercises" and not grammar_topics_selection:
                        st.error("Please select at least one grammar topic for the exercises.")
                    else:
                        with st.spinner(f"The AI Exam Committee is preparing your {practice_type}..."):
                            crew = LanguagePracticeCrew(selected_model_prac, native_language_prac, target_language_prac,
                                                        level_prac)
                            st.session_state.practice_material = crew.run(practice_type, grammar_topics_selection)

            if st.session_state.get('practice_material'):
                st.markdown("---")
                st.subheader(f"Your {target_language_prac} ({level_prac}) {practice_type}")
                st.markdown(st.session_state.practice_material)
                render_download_buttons(st.session_state.practice_material,
                                        f"{target_language_prac}_{level_prac}_{practice_type}")
        except Exception as error7:
            st.error(error7)

    with tab8:
        st.header("Explore the Dictionary")
        try:
            if 'dictionary_entry' not in st.session_state: st.session_state.dictionary_entry = None
            available_models = get_available_models(st.session_state.get('gemini_key'))

            with st.form("dictionary_form"):
                target_language = st.text_input("Language of the word", "English", key="dict_lang")
                word_to_define = st.text_input("Enter a word to define",
                                               placeholder="e.g., serendipity, serendipia, Zufallsfund")
                selected_model = st.selectbox("Choose AI Model", available_models,
                                              key="dict_model") if available_models else None

                if st.form_submit_button("Define Word", use_container_width=True):
                    if not all([word_to_define, selected_model]):
                        st.error("Please enter a word and select a model.")
                    else:
                        with st.spinner(f"The AI lexicographer is researching '{word_to_define}'..."):
                            crew = DictionaryCrew(selected_model, target_language)
                            st.session_state.dictionary_entry = crew.run(word_to_define)

            if st.session_state.get('dictionary_entry'):
                st.markdown("---")
                st.subheader(f"Dictionary Entry for '{word_to_define}'")
                st.markdown(st.session_state.dictionary_entry)
                render_download_buttons(st.session_state.dictionary_entry, f"dictionary_{word_to_define}")
        except Exception as error8:
            st.error(error8)



