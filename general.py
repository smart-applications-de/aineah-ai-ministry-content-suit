# main.py
# Final, unified, and production-ready application code for the AI Health & Fitness Suite.
import wave

import streamlit as st
import os
import io
import re
import json
import yfinance as yf
import pandas as pd
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from google import genai
import docx
import markdown2
from datetime import datetime
from main_v2 import  get_available_models, render_download_buttons,LANGUAGES
import pypdf
def calculate_bmi(weight_kg, height_cm):
    """Calculates BMI from weight in kg and height in cm."""
    if height_cm == 0:
        return 0
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return bmi

def get_bmi_cluster(bmi):
    """Classifies BMI into a health cluster key."""
    if bmi < 18.5:
        return "underweight"
    elif 18.5 <= bmi < 25:
        return "normal_weight"
    elif 25 <= bmi < 30:
        return "overweight"
    else:
        return "obesity"

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

class SwimmingCrew:
    def __init__(self, model_name, language, level):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        os.environ["SERPER_API_KEY"] = st.session_state.get('serper_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.language = language
        self.level = level

    def run(self):
        agents = [
            Agent(
                role='Kinesiology & Swimming Expert',
                goal=f"Develop the core technical swimming curriculum for a {self.level} swimmer. This includes drills, techniques, and safety protocols.",
                backstory="You are a certified swimming coach and kinesiologist with experience training athletes at all levels, from children learning to float to Olympic hopefuls. Your advice is scientifically sound and practical.",
                llm=self.llm,
                tools=[SerperDevTool()],
                verbose=True
            ),
            Agent(
                role='Health & Wellness Coach',
                goal="Explain the holistic health benefits of swimming, including cardiovascular, muscular, and mental wellness, tailored to the user's experience level.",
                backstory="You are a wellness coach who promotes swimming as a key activity for a healthy lifestyle. You excel at explaining complex health benefits in an inspiring and easy-to-understand manner.",
                llm=self.llm,
                verbose=True
            ),
            Agent(
                role='Content Editor & Translator',
                goal=f"Compile all the technical and wellness information into a single, cohesive, and beautifully formatted swimming guide in {self.language}.",
                backstory="You are a meticulous editor for a major sports and wellness publication. You ensure every guide is well-structured, easy to read, and accurately translated.",
                llm=self.llm,
                verbose=True
            )
        ]
        task_technical = Task(
            description=f"Create the core training content for a {self.level} swimmer. For a Beginner, focus on water safety, breathing, floating, and basic freestyle/backstroke. For Intermediate, focus on stroke refinement for all four strokes, endurance building, and basic turns. For an Expert, focus on advanced drills, interval training, race strategy, and flip turns.",
            agent=agents[0],
            expected_output="A detailed markdown section with clear headings for each technical aspect (e.g., 'Breathing Technique', 'Freestyle Drills')."
        )
        task_benefits = Task(
            description=f"Write a section on the health benefits of swimming, tailored for a {self.level} swimmer. Include cardiovascular, muscular, flexibility, and stress relief benefits. Frame the benefits in a way that is motivating for someone at this level.",
            agent=agents[1],
            expected_output="An engaging markdown section titled 'Holistic Health Benefits of Swimming'."
        )
        task_compile = Task(
            description=f"Combine the technical curriculum and the health benefits sections into a single, final swimming guide. Format it beautifully in markdown with a main title: 'Your Personalized Swimming Guide ({self.level})'. The entire guide must be in {self.language}.",
            agent=agents[2],
            context=[task_technical, task_benefits],
            expected_output="The final, comprehensive swimming guide formatted in clear markdown.",
            output_file="swimming_guide.md"
        )

        crew = Crew(
            agents=agents,
            tasks=[task_technical, task_benefits, task_compile],
            process=Process.sequential,
            verbose=True
        )

        crew.kickoff()
        with open("swimming_guide.md", "r", encoding="utf-8") as f:
            return f.read()


class HomeWorkoutCrew:
    def __init__(self, model_name, language, weight, height, goal, country, bmi, bmi_cluster_translated):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        os.environ["SERPER_API_KEY"] = st.session_state.get('serper_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.language = language
        self.weight = weight
        self.height = height
        self.goal = goal
        self.country = country
        self.bmi = bmi
        self.bmi_cluster_translated = bmi_cluster_translated

    def run(self):
        agents = [
            Agent(
                role='Certified Personal Trainer',
                goal=f"Create a detailed, day-by-day workout plan for a user with the goal of '{self.goal}', considering their BMI is {self.bmi:.1f} ({self.bmi_cluster_translated}).",
                backstory="You are a highly experienced personal trainer who creates safe, effective, and personalized workout plans. You tailor routines based on a user's BMI and specific goals.",
                llm=self.llm,
                verbose=True
            ),
            Agent(
                role='Sports Nutritionist',
                goal=f"Develop a complementary nutrition plan to support the user's fitness goal of '{self.goal}', considering their BMI cluster is '{self.bmi_cluster_translated}'. The plan should consider general dietary habits in {self.country}.",
                backstory="You are a certified nutritionist who specializes in creating meal plans that are scientifically aligned with a user's BMI and fitness objectives.",
                llm=self.llm,
                tools=[SerperDevTool()],
                verbose=True
            ),
            Agent(
                role='Health Advisor',
                goal=f"Provide specific, actionable recommendations for the user to achieve a normal BMI, based on their current status of '{self.bmi_cluster_translated}'.",
                backstory="You are a health advisor who provides clear, encouraging advice on lifestyle changes. You explain the importance of a healthy BMI and provide practical steps to achieve it.",
                llm=self.llm,
                verbose=True
            ),
            Agent(
                role='Fitness Content Editor',
                goal=f"Compile all the workout, nutrition, and health advice into a single, cohesive, and easy-to-read fitness plan in {self.language}.",
                backstory="You are a meticulous editor for a leading fitness magazine. You ensure every plan is well-structured, clearly explained, and motivating for the reader.",
                llm=self.llm,
                verbose=True
            )
        ]
        task_workout = Task(
            description=f"Create a 7-day workout schedule tailored for a user aiming for '{self.goal}', with a BMI of {self.bmi:.1f} ({self.bmi_cluster_translated}). Specify exercises, duration, sets, and reps for each day.",
            agent=agents[0],
            expected_output="A detailed markdown table outlining the 7-day workout plan."
        )
        task_nutrition = Task(
            description=f"Create a nutrition guideline that supports the goal of '{self.goal}' for a person in the '{self.bmi_cluster_translated}' category. Provide examples of breakfast, lunch, dinner, and snacks.",
            agent=agents[1],
            expected_output="A markdown section with dietary recommendations and sample meal ideas."
        )
        task_bmi_advice = Task(
            description=f"Write a concluding section titled 'Path to a Healthy BMI'. If the user is not in the 'Normal weight' category, provide 2-3 clear, actionable steps they can take to move towards a healthier BMI (e.g., for 'Overweight', suggest a slight caloric deficit and increased cardio).",
            agent=agents[2],
            expected_output="A concise, encouraging markdown section with specific recommendations for achieving a normal BMI."
        )
        task_compile = Task(
            description=f"Combine the workout plan, nutrition guide, and BMI advice into a single, final fitness plan. Format it beautifully in markdown with a main title: 'Your Personalized Fitness Plan for {self.goal}'. The entire guide must be in {self.language}.",
            agent=agents[3],
            context=[task_workout, task_nutrition, task_bmi_advice],
            expected_output="The final, comprehensive fitness plan formatted in clear markdown.",
            output_file="fitness_plan.md"
        )

        crew = Crew(
            agents=agents,
            tasks=[task_workout, task_nutrition, task_bmi_advice, task_compile],
            process=Process.sequential,
            verbose=True
        )

        crew.kickoff()
        with open("fitness_plan.md", "r", encoding="utf-8") as f:
            return f.read()


class FitnessStudioCrew:
    def __init__(self, model_name, language, bmi, bmi_cluster, goal):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.language = language
        self.bmi = bmi
        self.bmi_cluster = bmi_cluster
        self.goal = goal

    def run(self):
        agents = [
            Agent(
                role='McFIT Certified Master Trainer',
                goal=f"Create a detailed, 7-day fitness studio workout plan for a user with a BMI of {self.bmi:.1f} ({self.bmi_cluster}) whose goal is '{self.goal}'.",
                backstory="You are a Master Trainer with decades of experience at McFIT, Germany's leading fitness chain. You are an expert in designing workout plans that utilize common gym equipment.",
                llm=self.llm,
                verbose=True
            ),
            Agent(
                role='Exercise Physiologist',
                goal="Provide expert instructions and tips for each exercise in the workout plan.",
                backstory="You are an exercise physiologist who specializes in biomechanics and proper exercise form. You provide clear, concise instructions to maximize effectiveness and prevent injury.",
                llm=self.llm,
                verbose=True
            ),
            Agent(
                role='Health Advisor',
                goal=f"Provide specific, actionable recommendations for the user to achieve a normal BMI, based on their current status of '{self.bmi_cluster}'.",
                backstory="You are a health advisor who provides clear, encouraging advice on lifestyle changes. You explain the importance of a healthy BMI and provide practical steps to achieve it.",
                llm=self.llm,
                verbose=True
            ),
            Agent(
                role='Fitness Content Editor',
                goal=f"Compile all the workout information into a single, cohesive, and easy-to-read fitness plan in {self.language}.",
                backstory="You are a meticulous editor for a leading fitness magazine. You ensure every plan is well-structured, clearly explained, and motivating for the reader.",
                llm=self.llm,
                verbose=True
            )
        ]

        task1 = Task(
            description=f"Create a 7-day workout schedule focused on '{self.goal}'. For each workout day, list 5-6 exercises using standard gym equipment. Specify duration, sets, reps, and rest periods.",
            agent=agents[0],
            expected_output="A detailed markdown table for a 7-day workout plan."
        )
        task2 = Task(
            description="For each exercise in the generated plan, write a short 'How-To' instruction and a 'Pro-Tip' for proper form.",
            agent=agents[1],
            context=[task1],
            expected_output="A list of all exercises from the plan, each with its own 'How-To' and 'Pro-Tip' section."
        )
        task3 = Task(
            description=f"Write a concluding section titled 'Path to a Healthy BMI'. If the user is not in the 'Normal weight' category, provide 2-3 clear, actionable steps they can take to move towards a healthier BMI.",
            agent=agents[2],
            expected_output="A concise, encouraging markdown section with specific recommendations for achieving a normal BMI."
        )
        task4 = Task(
            description=f"Combine the 7-day plan, detailed exercise instructions, and BMI advice into a single, final fitness studio plan. Format it beautifully in markdown with a main title: 'Your Fitness Studio Plan: {self.goal}'. The entire guide must be in {self.language}.",
            agent=agents[3],
            context=[task1, task2, task3],
            expected_output="The final, comprehensive fitness studio plan formatted in clear markdown.",
            output_file="fitness_studio_plan.md"
        )

        crew = Crew(
            agents=agents,
            tasks=[task1, task2, task3, task4],
            process=Process.sequential,
            verbose=True
        )

        crew.kickoff()
        with open("fitness_studio_plan.md", "r", encoding="utf-8") as f:
            return f.read()


class DrivingLicenseCrew:
    def __init__(self, model_name, language, country, license_class, study_level):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        os.environ["SERPER_API_KEY"] = st.session_state.get('serper_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.language = language
        self.country = country
        self.license_class = license_class
        self.study_level = study_level

    def run(self):
        agents = [
            Agent(
                role='Certified Driving Instructor',
                goal=f"Create a comprehensive study guide for the {self.license_class} driving license in {self.country}, focusing on {self.study_level}.",
                backstory=f"You are a highly experienced driving instructor in {self.country}, certified to teach for Class {self.license_class}. You know the local traffic laws, road signs, and common test challenges inside and out.",
                llm=self.llm,
                tools=[SerperDevTool()],
                verbose=True
            ),
            Agent(
                role='Vehicle Safety & Defensive Driving Expert',
                goal="Provide a crucial section on vehicle safety checks and defensive driving principles.",
                backstory="You are a vehicle safety expert who contributes to official driving manuals. Your focus is on creating safe, aware, and responsible drivers.",
                llm=self.llm,
                verbose=True
            ),
            Agent(
                role='Content Editor & Translator',
                goal=f"Compile all the study materials into a single, cohesive, and easy-to-read guide in {self.language}.",
                backstory="You are a meticulous editor for a company that publishes educational materials. You ensure every guide is well-structured, clearly explained, and accurately translated.",
                llm=self.llm,
                verbose=True
            )
        ]

        tasks = []
        if self.study_level in ["Theory", "Both"]:
            tasks.append(Task(
                description=f"Create the 'Traffic Laws & Road Signs' section for the {self.license_class} theory test in {self.country}. Cover key topics like speed limits, right-of-way rules, and common road signs.",
                agent=agents[0],
                expected_output="A detailed markdown section covering the core theoretical knowledge."
            ))
        if self.study_level in ["Practical", "Both"]:
            tasks.append(Task(
                description=f"Create the 'Practical Driving Skills' section for the {self.license_class} practical test in {self.country}. Cover essential maneuvers like parking, lane changes, and navigating intersections.",
                agent=agents[0],
                expected_output="A step-by-step guide in markdown for performing key practical driving skills."
            ))

        tasks.append(Task(
            description="Create a section on 'Vehicle Safety & Defensive Driving'. Include a pre-drive vehicle checklist and 5 key principles of defensive driving.",
            agent=agents[1],
            expected_output="A concise markdown section on safety checks and defensive driving."
        ))

        compile_task = Task(
            description=f"Combine all the generated sections into a single, final study guide. Format it beautifully in markdown with a main title: 'Your Driving License Study Guide: Class {self.license_class} ({self.country})'. The entire guide must be in {self.language}.",
            agent=agents[2],
            context=tasks,
            expected_output="The final, comprehensive study guide formatted in clear markdown.",
            output_file="driving_guide.md"
        )
        tasks.append(compile_task)

        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        crew.kickoff()
        with open("driving_guide.md", "r", encoding="utf-8") as f:
            return f.read()


# ==============================================================================
## 3. Page Rendering Functions
# ==============================================================================

def render_swimming_page():
    st.title("🏊 AI Swimming Coach")
    st.markdown("Get a personalized swimming guideline from an expert AI crew, tailored to your skill level.")

    if 'swimming_guide' not in st.session_state:
        st.session_state.swimming_guide = None

    available_models = get_available_models(st.session_state.get('gemini_key'))
    LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")

    with st.form("swimming_form"):
        st.header("Create Your Swimming Guide")
        col1, col2 = st.columns(2)
        language = col1.selectbox("Choose Language:", LANGUAGES)
        level = col2.selectbox("Select Your Swimming Level:", ["Beginner", "Intermediate", "Expert"])

        selected_model = st.selectbox("Choose AI Model:", available_models) if available_models else None

        if st.form_submit_button("Generate My Guideline", use_container_width=True):
            if not selected_model:
                st.error("Please select a model.")
            else:
                with st.spinner(f"The AI coaching team is developing your {level} guideline..."):
                    crew = SwimmingCrew(selected_model, language, level)
                    st.session_state.swimming_guide = crew.run()

    if st.session_state.get('swimming_guide'):
        st.markdown("---")
        st.header(f"Your Personalized Swimming Guide ({level})")
        st.markdown(st.session_state.swimming_guide)
        render_download_buttons(st.session_state.swimming_guide, f"swimming_guide_{level.lower()}")


def render_fitness_page():
    st.title("🏋️ AI Fitness Trainer")
    st.markdown("Get a personalized workout and nutrition plan from an expert AI crew, tailored to your BMI and goals.")

    st.warning(
        """
        **⚠️ IMPORTANT HEALTH DISCLAIMER**

        This AI provides fitness and nutrition suggestions for informational purposes only. It is **NOT** a substitute for professional medical advice. 

        **Consult with a qualified healthcare professional or a certified personal trainer** before beginning any new fitness program or making changes to your diet, especially if you have pre-existing health conditions.
        """,
        icon="❗"
    )

    tab1, tab2 = st.tabs(["**Home & General Fitness**", "**Fitness Studio Workouts**"])

    with tab1:
        st.header("Create Your General Fitness & Nutrition Plan")
        if 'fitness_plan' not in st.session_state: st.session_state.fitness_plan = None
        available_models = get_available_models(st.session_state.get('gemini_key'))
        BMI_TRANSLATIONS = {
            "English": {"underweight": "Underweight", "normal_weight": "Normal weight", "overweight": "Overweight",
                        "obesity": "Obesity"},
            "German": {"underweight": "Untergewicht", "normal_weight": "Normalgewicht", "overweight": "Übergewicht",
                       "obesity": "Adipositas"},
            "French": {"underweight": "Insuffisance pondérale", "normal_weight": "Poids normal",
                       "overweight": "Surpoids", "obesity": "Obésité"},
            "Spanish": {"underweight": "Bajo peso", "normal_weight": "Peso normal", "overweight": "Sobrepeso",
                        "obesity": "Obesidad"},
            "Italian": {"underweight": "Sottopeso", "normal_weight": "Normopeso", "overweight": "Sovrappeso",
                        "obesity": "Obesità"},
            "Portuguese": {"underweight": "Abaixo do peso", "normal_weight": "Peso normal", "overweight": "Sobrepeso",
                           "obesity": "Obesidade"},
            "Swahili": {"underweight": "Uzito mdogo", "normal_weight": "Uzito wa kawaida",
                        "overweight": "Uzito uliopitiliza", "obesity": "Unene uliokithiri"}
        }
        #LANGUAGES1 = list(BMI_TRANSLATIONS.keys())

        with st.form("home_fitness_form"):
            col1, col2 = st.columns(2)
            weight = col1.number_input("Your Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
            height = col2.number_input("Your Height (cm)", min_value=100.0, max_value=250.0, value=175.0, step=0.5)

            goal = st.selectbox("What is your primary fitness goal?",
                                ["Weight Loss", "Muscle Building", "General Fitness & Endurance"])

            col3, col4 = st.columns(2)
            country = col3.text_input("Your Country", placeholder="e.g., USA, Germany")
            language = col4.selectbox("Language for the Plan:", LANGUAGES)

            selected_model = st.selectbox("Choose AI Model:", available_models) if available_models else None

            if weight > 0 and height > 0:
                bmi = calculate_bmi(weight, height)
                bmi_cluster_key = get_bmi_cluster(bmi)
                bmi_cluster_translated = BMI_TRANSLATIONS.get(language, BMI_TRANSLATIONS["English"]).get(
                    bmi_cluster_key, "N/A")
                st.info(f"**Your BMI is: {bmi:.1f}** (Category: **{bmi_cluster_translated}**)")
            else:
                bmi = 0
                bmi_cluster_translated = "N/A"

            if st.form_submit_button("Generate My Fitness Plan", use_container_width=True):
                if not all([weight, height, goal, country, language, selected_model]):
                    st.error("Please fill all fields and select a model.")
                else:
                    with st.spinner(f"The AI fitness team is creating your plan for {goal}..."):
                        crew = HomeWorkoutCrew(selected_model, language, weight, height, goal, country, bmi,
                                               bmi_cluster_translated)
                        st.session_state.fitness_plan = crew.run()

        if st.session_state.get('fitness_plan'):
            st.markdown("---")
            st.header(f"Your Personalized Fitness Plan")
            st.markdown(st.session_state.fitness_plan)
            render_download_buttons(st.session_state.fitness_plan, f"fitness_plan_{goal.replace(' ', '_').lower()}")

    with tab2:
        st.header("Create Your Fitness Studio Workout Plan")
        if 'studio_plan' not in st.session_state: st.session_state.studio_plan = None
        available_models = get_available_models(st.session_state.get('gemini_key'))

        with st.form("studio_fitness_form"):
            col1, col2 = st.columns(2)
            weight_studio = col1.number_input("Your Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5,
                                              key="studio_weight")
            height_studio = col2.number_input("Your Height (cm)", min_value=100.0, max_value=250.0, value=175.0,
                                              step=0.5, key="studio_height")

            goal_studio = st.selectbox("What is your primary workout goal?",
                                       ["Weight Loss", "Full Body Strength", "Six-Pack Abs", "Chest", "Back", "Legs",
                                        "Biceps", "Triceps", "Shoulders"])

            col3, col4 = st.columns(2)
            language_studio = col3.selectbox("Language for the Plan:", LANGUAGES, key="studio_lang")
            selected_model_studio = col4.selectbox("Choose AI Model:", available_models,
                                                   key="studio_model") if available_models else None

            if weight_studio > 0 and height_studio > 0:
                bmi_studio = calculate_bmi(weight_studio, height_studio)
                bmi_cluster_key_studio = get_bmi_cluster(bmi_studio)
                bmi_cluster_translated_studio = BMI_TRANSLATIONS.get(language_studio, BMI_TRANSLATIONS["English"]).get(
                    bmi_cluster_key_studio, "N/A")
                st.info(f"**Your BMI is: {bmi_studio:.1f}** (Category: **{bmi_cluster_translated_studio}**)")
            else:
                bmi_studio = 0
                bmi_cluster_translated_studio = "N/A"

            if st.form_submit_button("Generate My Studio Plan", use_container_width=True):
                if not all([weight_studio, height_studio, goal_studio, language_studio, selected_model_studio]):
                    st.error("Please fill all fields and select a model.")
                else:
                    with st.spinner(f"The McFIT AI Master Trainer is designing your studio plan..."):
                        crew = FitnessStudioCrew(selected_model_studio, language_studio, bmi_studio,
                                                 bmi_cluster_translated_studio, goal_studio)
                        st.session_state.studio_plan = crew.run()

        if st.session_state.get('studio_plan'):
            st.markdown("---")
            st.header(f"Your Fitness Studio Plan")
            st.markdown(st.session_state.studio_plan)
            render_download_buttons(st.session_state.studio_plan,
                                    f"studio_plan_{goal_studio.replace(' ', '_').lower()}")


def render_driving_license_page():
    st.title("🚗 AI Driving License Guide")
    st.markdown("Get a personalized study guide from an expert AI crew to help you ace your driving test.")

    st.warning(
        """
        **⚠️ IMPORTANT DISCLAIMER: FOR INFORMATIONAL PURPOSES ONLY**

        This AI-generated guide is intended as a supplementary study aid and is **NOT** an official driving manual or a substitute for professional driving instruction. 

        **Always refer to your country's official driving handbook and seek lessons from a certified driving instructor.** Traffic laws and test requirements can change and may vary by region.
        """,
        icon="❗"
    )

    if 'driving_guide' not in st.session_state:
        st.session_state.driving_guide = None

    available_models = get_available_models(st.session_state.get('gemini_key'))

    with st.form("driving_form"):
        st.header("Create Your Study Guide")
        col1, col2 = st.columns(2)
        country = col1.text_input("Country", placeholder="e.g., Germany, USA")
        license_class = col2.text_input("License Class", placeholder="e.g., B, Class 5, Car")

        col3, col4 = st.columns(2)
        language = col3.selectbox("Language for the Guide:", LANGUAGES)
        study_level = col4.selectbox("Area of Study:", ["Theory", "Practical", "Both"])

        selected_model = st.selectbox("Choose AI Model:", available_models) if available_models else None

        if st.form_submit_button("Generate My Study Guide", use_container_width=True):
            if not all([country, license_class, language, selected_model]):
                st.error("Please fill all fields and select a model.")
            else:
                with st.spinner(f"The AI driving instructors are creating your guide for {country}..."):
                    crew = DrivingLicenseCrew(selected_model, language, country, license_class, study_level)
                    st.session_state.driving_guide = crew.run()

    if st.session_state.get('driving_guide'):
        st.markdown("---")
        st.header(f"Your Driving License Study Guide")
        st.markdown(st.session_state.driving_guide)
        render_download_buttons(st.session_state.driving_guide,
                                f"driving_guide_{country.lower()}_{license_class.lower()}")


class TaxAdvisorCrew:
    def __init__(self, model_name, language):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        os.environ["SERPER_API_KEY"] = st.session_state.get('serper_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.language = language

    def run_clarification_crew(self, tax_class, user_question, document_context):
        agents = [
            Agent(role='Tax Document Specialist',
                  goal="Analyze the user-provided tax document context to extract all relevant figures, dates, and key information.",
                  backstory="You are an AI assistant modeled after a meticulous 'Steuerfachangestellte(r)'. You are an expert at reading and interpreting German tax documents.",
                  llm=self.llm, verbose=True),
            Agent(role='German Tax Law Expert',
                  goal=f"Research and provide a detailed analysis of the user's tax situation based on their question, document, and Steuerklasse {tax_class}. Focus on the laws for the current or most recent tax year.",
                  backstory="You are an AI expert modeled after a specialist in German tax law ('Steuerrecht'). You are always up-to-date with the latest changes.",
                  llm=self.llm, tools=[SerperDevTool()], verbose=True),
            Agent(role='Senior Steuerberater (Tax Advisor)',
                  goal=f"Synthesize all the analysis into a comprehensive, easy-to-understand, and actionable advisory report in {self.language}.",
                  backstory="You are an AI assistant with the persona of an experienced German Steuerberater. You provide expert-level guidance and suggest clear next steps.",
                  llm=self.llm, verbose=True)
        ]

        task1 = Task(
            description=f"Analyze the following context from a user's tax document: '{document_context}'. Identify and list all key financial figures and personal data.",
            agent=agents[0],
            expected_output="A structured list of all relevant data points extracted from the document context.")
        task2 = Task(
            description=f"Based on the extracted data, the user's question ('{user_question}'), and their tax class ({tax_class}), research the relevant German tax laws. Provide an expert analysis covering potential deductions and obligations.",
            agent=agents[1], context=[task1],
            expected_output="A detailed analysis of the tax situation with references to relevant regulations.")
        task3 = Task(
            description=f"Compile a final advisory report in {self.language}. Start by directly answering the user's question. Then, summarize the key findings. Conclude with a clear 'Recommended Next Steps' section.",
            agent=agents[2], context=[task2],
            expected_output="The final, comprehensive, and user-friendly advisory report in markdown format.",
            output_file="tax_advice.md")

        crew = Crew(agents=agents, tasks=[task1, task2, task3], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("tax_advice.md", "r", encoding="utf-8") as f:
            return f.read()

    def run_elster_guide_crew(self, tax_form, tax_year):
        agents = [
            Agent(role='German Tax Form Expert',
                  goal=f"Provide a detailed, line-by-line explanation of the German tax form '{tax_form}' for the tax year {tax_year}.",
                  backstory=f"You are an AI expert who has memorized every line of the German tax forms for the last decade, including the official instructions from the Bundesfinanzministerium.",
                  llm=self.llm, tools=[SerperDevTool()], verbose=True),
            Agent(role='ELSTER Software Tutor',
                  goal="Translate the form explanation into a practical, step-by-step guide for filling out the form in the ELSTER software.",
                  backstory="You are an expert tutor for the ELSTER online tax portal. You know how to navigate the software and provide clear, actionable instructions and best-practice tips.",
                  llm=self.llm, verbose=True)
        ]

        task1 = Task(
            description=f"Research and list all the questions and required fields for the German tax form '{tax_form}' for the year {tax_year}. For each field, provide a brief explanation of what information is required.",
            agent=agents[0],
            expected_output="A comprehensive list of all fields in the specified tax form with explanations.")
        task2 = Task(
            description=f"Convert the list of questions and fields into a step-by-step guide for using the ELSTER portal. For each point, explain where to find it in ELSTER and give a 'Best Practice Tip' on how to answer it correctly. The entire guide must be in {self.language}.",
            agent=agents[1], context=[task1],
            expected_output="A complete, step-by-step ELSTER guide in markdown format.", output_file="elster_guide.md")

        crew = Crew(agents=agents, tasks=[task1, task2], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("elster_guide.md", "r", encoding="utf-8") as f:
            return f.read()


class SalaryAnalysisCrew:
    def __init__(self, model_name, language, country, salary, period, **kwargs):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        os.environ["SERPER_API_KEY"] = st.session_state.get('serper_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.language = language
        self.country = country
        self.salary = salary
        self.period = period
        self.details = kwargs

    def run(self):
        current_year = datetime.now().year

        agents = [
            Agent(role='International Payroll Specialist',
                  goal="Calculate the user's annual gross salary and structure the initial data for the tax expert.",
                  backstory="You are an expert in global payroll standards. You understand various pay periods and can accurately annualize salaries.",
                  llm=self.llm, verbose=True),
            Agent(role=f'Tax Law Expert for {self.country} ({current_year})',
                  goal=f"Research and provide a detailed breakdown of all mandatory tax and social security deductions for {self.country} for the current tax year, {current_year}.",
                  backstory=f"You are an expert in tax and social security laws for various countries, with a special focus on {self.country}. You are always up-to-date with the latest {current_year} rates and thresholds.",
                  llm=self.llm, tools=[SerperDevTool()], verbose=True),
            Agent(role='Financial Analyst & Content Editor',
                  goal=f"Calculate the net salary, create a detailed payslip, and explain each deduction clearly in {self.language}, referencing the current year's rules.",
                  backstory="You are a skilled financial analyst who can transform complex calculations into a simple, easy-to-understand payslip. You are also a gifted communicator, able to explain financial concepts to a layperson.",
                  llm=self.llm, verbose=True)
        ]

        context_details = f"Country: {self.country}, Gross Salary: {self.salary} ({self.period}), Language: {self.language}, Tax Year: {current_year}."
        if self.country == "Germany":
            context_details += f" State: {self.details.get('state', 'N/A')}, Tax Class: {self.details.get('tax_class', 'N/A')}, Church Member: {'Yes' if self.details.get('church_tax') else 'No'}."

        task1 = Task(description=f"Based on the user's input ({context_details}), determine the annual gross salary.",
                     agent=agents[0], expected_output="The calculated annual gross salary as a single number.")
        task2 = Task(
            description=f"Based on the user's details ({context_details}), find all applicable tax rates, social security contributions, and their respective income thresholds (Beitragsbemessungsgrenze) for the year {current_year}. For Germany, include Lohnsteuer, Solidaritätszuschlag, Kirchensteuer, Rentenversicherung, Arbeitslosenversicherung, Krankenversicherung, and Pflegeversicherung.",
            agent=agents[1],
            expected_output=f"A structured list of all deductions with their {current_year} percentages and income thresholds.")
        task3 = Task(
            description=f"Using the annual gross salary and the {current_year} deduction rates, create a comprehensive payslip. Calculate each deduction, the total deductions, and the final net monthly salary. Then, add a section explaining each deduction in simple terms, explicitly mentioning that these calculations are based on {current_year} rules. The entire output must be in {self.language}.",
            agent=agents[2], context=[task1, task2],
            expected_output="A complete, beautifully formatted markdown document containing the detailed payslip and explanations.",
            output_file="payslip.md")

        crew = Crew(agents=agents, tasks=[task1, task2, task3], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("payslip.md", "r", encoding="utf-8") as f:
            return f.read()


# ==============================================================================
## 3. Page Rendering Functions
# ==============================================================================

def render_tax_advisor_page():
    st.title(" fiscally.AI 🇩🇪")
    st.markdown("Your AI-powered assistant for German tax clarification and salary calculations.")

    st.error(
        "**⚠️ IMPORTANT DISCLAIMER:** This tool is an AI assistant and does **NOT** provide legally binding tax or financial advice. Always consult with a certified professional.",
        icon="❗")

    if 'tax_advice' not in st.session_state: st.session_state.tax_advice = None
    if 'elster_guide' not in st.session_state: st.session_state.elster_guide = None
    if 'payslip' not in st.session_state: st.session_state.payslip = None

    available_models = get_available_models(st.session_state.get('gemini_key'))
    LANGUAGES = ("English", "German")

    tab1, tab2, tab3 = st.tabs(
        ["**General Tax Clarification**", "**ELSTER Form Guide**", "**Salary & Net Pay Calculator**"])

    with tab1:
        st.header("Analyze a Document or Ask a Question")
        with st.form("tax_form"):
            uploaded_file = st.file_uploader("Upload your tax document (optional)", type=['pdf', 'png', 'jpg', 'jpeg'])
            col1, col2 = st.columns(2)
            tax_class = col1.selectbox("Select Your Steuerklasse (Tax Class):", ["I", "II", "III", "IV", "V", "VI"])
            language = col2.selectbox("Language for the Response:", LANGUAGES)
            user_question = st.text_area("Enter your specific question:", height=100,
                                         placeholder="e.g., Can I deduct my home office expenses?")
            selected_model = st.selectbox("Choose AI Model:", available_models) if available_models else None

            if st.form_submit_button("Get Tax Analysis", use_container_width=True):
                if not user_question and not uploaded_file:
                    st.error("Please enter a question or upload a document.")
                elif not selected_model:
                    st.error("Please select a model.")
                else:
                    document_context = ""
                    if uploaded_file is not None:
                        with st.spinner("Analyzing your document..."):
                            if uploaded_file.type == "application/pdf":
                                pdf_reader = pypdf.PdfReader(uploaded_file)
                                for page in pdf_reader.pages: document_context += page.extract_text()
                            else:
                                document_context = "User has uploaded an image document."

                    with st.spinner(f"The AI tax team is preparing your analysis..."):
                        crew = TaxAdvisorCrew(selected_model, language)
                        st.session_state.tax_advice = crew.run_clarification_crew(tax_class, user_question,
                                                                                  document_context)

        if st.session_state.get('tax_advice'):
            st.markdown("---");
            st.header("Your AI-Generated Tax Report");
            st.markdown(st.session_state.tax_advice)
            render_download_buttons(st.session_state.tax_advice, "tax_advice_report")

    with tab2:
        st.header("Get a Step-by-Step Guide for Your Tax Forms")
        GERMAN_TAX_FORMS = ["Anlage N (Employment Income)", "Anlage V (Rental Income)", "Anlage KAP (Capital Gains)",
                            "Anlage SO (Other Income)", "Anlage G (Business Income)", "Anlage S (Freelance Income)"]

        with st.form("elster_form"):
            tax_form = st.selectbox("Select the Tax Form (Anlage):", GERMAN_TAX_FORMS)
            tax_year = st.number_input("Enter the Tax Year:", min_value=2020, max_value=datetime.now().year,
                                       value=datetime.now().year - 1)
            language = st.selectbox("Language for the Guide:", LANGUAGES, key="elster_lang")
            selected_model = st.selectbox("Choose AI Model:", available_models,
                                          key="elster_model") if available_models else None

            if st.form_submit_button("Generate ELSTER Guide", use_container_width=True):
                if not all([tax_form, tax_year, language, selected_model]):
                    st.error("Please fill all fields and select a model.")
                else:
                    with st.spinner(f"The AI tax experts are creating your guide for {tax_form}..."):
                        crew = TaxAdvisorCrew(selected_model, language)
                        st.session_state.elster_guide = crew.run_elster_guide_crew(tax_form, tax_year)

        if st.session_state.get('elster_guide'):
            st.markdown("---");
            st.header(f"Your ELSTER Guide for {tax_form}")
            st.markdown(st.session_state.elster_guide)
            render_download_buttons(st.session_state.elster_guide, "elster_guide")

    with tab3:
        st.header("Calculate Your Estimated Net Salary")
        with st.form("salary_form"):
            col1, col2 = st.columns(2)
            country = col1.text_input("Country", "Germany")
            language = col2.selectbox("Language for Response:", LANGUAGES, key="salary_lang")

            col3, col4 = st.columns(2)
            salary = col3.number_input("Gross Salary", min_value=0.0, value=50000.0, step=1000.0)
            period = col4.selectbox("Pay Period", ["Annually", "Monthly"])

            german_details = {}
            if country.lower() == 'germany':
                st.subheader("German Tax Details (Angestellte)")
                col5, col6 = st.columns(2)
                german_details['state'] = col5.selectbox("State (Bundesland):",
                                                         ["Baden-Württemberg", "Bavaria (Bayern)", "Berlin",
                                                          "Brandenburg", "Bremen", "Hamburg", "Hesse (Hessen)",
                                                          "Lower Saxony (Niedersachsen)", "Mecklenburg-Vorpommern",
                                                          "North Rhine-Westphalia (NRW)",
                                                          "Rhineland-Palatinate (Rheinland-Pfalz)", "Saarland",
                                                          "Saxony (Sachsen)", "Saxony-Anhalt (Sachsen-Anhalt)",
                                                          "Schleswig-Holstein", "Thuringia (Thüringen)"])
                german_details['tax_class'] = col6.selectbox("Steuerklasse (Tax Class):",
                                                             ["I", "II", "III", "IV", "V", "VI"])
                german_details['church_tax'] = st.checkbox("Are you a member of a church that collects church tax?")

            selected_model = st.selectbox("Choose AI Model:", available_models,
                                          key="salary_model") if available_models else None

            if st.form_submit_button("Calculate Net Salary", use_container_width=True):
                if not all([country, language, salary, selected_model]):
                    st.error("Please fill all required fields and select a model.")
                else:
                    with st.spinner(f"The AI finance team is calculating your net pay..."):
                        crew = SalaryAnalysisCrew(selected_model, language, country, salary, period, **german_details)
                        st.session_state.payslip = crew.run()

        if st.session_state.get('payslip'):
            st.markdown("---");
            st.header(f"Your Estimated Payslip")
            st.markdown(st.session_state.payslip)
            render_download_buttons(st.session_state.payslip, "salary_payslip")


class LessonPlannerCrew:
    def __init__(self, model_name, language, grade_level, subject, topic):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.language = language
        self.grade_level = grade_level
        self.subject = subject
        self.topic = topic

    def run(self):
        agent = Agent(
            role='Lesson Plan Architect',
            goal=f"Create a detailed, engaging, and age-appropriate lesson plan for a {self.grade_level} {self.subject} class on the topic of '{self.topic}'.",
            backstory="You are an expert instructional designer with a PhD in Education. You specialize in creating lesson plans that are not only informative but also highly engaging and tailored to specific grade levels and subjects. You adhere to the highest pedagogical standards.",
            llm=self.llm,
            verbose=True
        )
        task_description = f"""
        Write a complete lesson plan in {self.language} for a {self.grade_level} {self.subject} class of approximately 20-25 students. The topic is '{self.topic}'.
        Your lesson plan MUST adhere to the following structure and principles:
        - **Learning Objectives:** Clearly define what students should know or be able to do.
        - **Materials & Resources:** List all necessary materials.
        - **Lesson Sequence:** Provide a logical, step-by-step sequence.
        - **Time Constraints:** Set realistic time estimates for each part.
        - **Engaging Activities:** Incorporate interactive and tech-friendly activities.
        - **Differentiated Instruction:** Include options to accommodate different learning styles.
        - **Assessment:** Suggest a method for assessing student understanding.
        - **Contingency Plan:** Suggest an alternative plan or extension activity.
        """
        task = Task(
            description=task_description,
            agent=agent,
            expected_output="A complete, well-structured lesson plan in markdown format, including all specified sections.",
            output_file="lesson_plan.md"
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("lesson_plan.md", "r", encoding="utf-8") as f:
            return f.read()


class ExamGeneratorCrew:
    def __init__(self, model_name, language, grade_level, subject, exam_type, country):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        os.environ["SERPER_API_KEY"] = st.session_state.get('serper_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.language = language
        self.grade_level = grade_level
        self.subject = subject
        self.exam_type = exam_type
        self.country = country

    def run(self):
        agents = [
            Agent(
                role='Curriculum Specialist',
                goal=f"Identify the key topics and learning standards for a {self.grade_level} {self.subject} {self.exam_type} in {self.country}.",
                backstory=f"You are an expert in global education curricula, with a deep understanding of the learning objectives for {self.subject} at the {self.grade_level} level in {self.country}.",
                llm=self.llm, tools=[SerperDevTool()], verbose=True
            ),
            Agent(
                role='Exam Designer',
                goal="Create a set of exam questions that accurately assess student knowledge on the provided topics, focusing on grammar and reading comprehension.",
                backstory="You are a professional exam creator for educational institutions. You design fair, challenging, and well-structured questions that cover a range of cognitive skills.",
                llm=self.llm, verbose=True
            ),
            Agent(
                role='Chief Editor and Answer Key Compiler',
                goal="Compile all exam questions into a cohesive document and create a separate, clear answer key.",
                backstory="You are a meticulous editor for an educational testing service. You ensure that exams are perfectly formatted and that the answer key is unambiguous.",
                llm=self.llm, verbose=True
            )
        ]
        task1 = Task(
            description=f"Based on the curriculum for {self.subject} for a {self.grade_level} in {self.country}, outline the key topics that should be covered in a {self.exam_type}.",
            agent=agents[0],
            expected_output="A bulleted list of 3-5 key curriculum topics."
        )
        task2 = Task(
            description=f"Create a set of exam questions based on the key topics. The exam should include a grammar section and a reading comprehension section with a short essay. The instructions must be in {self.language}.",
            agent=agents[1],
            context=[task1],
            expected_output="A well-structured markdown document with numbered questions for each section."
        )
        task3 = Task(
            description=f"Combine the exam questions into a single document titled '{self.exam_type}'. Then, create a separate section at the end titled 'Answer Key' with clear solutions for all questions. The entire output must be in {self.language}.",
            agent=agents[2],
            context=[task2],
            expected_output=f"A complete, beautifully formatted markdown document containing the full {self.exam_type} and a comprehensive answer key.",
            output_file="exam_paper.md"
        )
        crew = Crew(agents=agents, tasks=[task1, task2, task3], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("exam_paper.md", "r", encoding="utf-8") as f:
            return f.read()


# ==============================================================================
## 3. Page Rendering Function
# ==============================================================================

def render_lesson_planner_page():
    st.title("🍎 AI Education Studio")
    st.markdown("Your hub for creating detailed lesson plans and custom exams.")

    tab1, tab2 = st.tabs(["**Lesson Planner**", "**Exam Generator**"])

    with tab1:
        st.header("Create a Custom Lesson Plan")
        if 'lesson_plan' not in st.session_state: st.session_state.lesson_plan = None
        available_models = get_available_models(st.session_state.get('gemini_key'))
        GRADE_LEVELS = [f"{i}{'st' if i == 1 else 'nd' if i == 2 else 'rd' if i == 3 else 'th'} Grade" for i in
                        range(1, 13)] + ["High School (9-12)"]
        SUBJECTS = ["History", "Science", "Mathematics", "Literature", "Art", "Geography", "Music"]

        with st.form("lesson_planner_form"):
            col1, col2 = st.columns(2)
            grade_level = col1.selectbox("Select Grade Level:", GRADE_LEVELS, key="lp_grade")
            subject = col2.selectbox("Select Subject:", SUBJECTS, key="lp_subject")
            topic = st.text_input("Enter the Lesson Topic:",
                                  placeholder="e.g., The History of the Versailles Palace, The Water Cycle")
            col3, col4 = st.columns(2)
            language = col3.selectbox("Language for the Lesson Plan:", LANGUAGES, key="lp_lang")
            selected_model = col4.selectbox("Choose AI Model:", available_models,
                                            key="lp_model") if available_models else None

            if st.form_submit_button("Generate Lesson Plan", use_container_width=True):
                if not all([grade_level, subject, topic, language, selected_model]):
                    st.error("Please fill all fields and select a model.")
                else:
                    with st.spinner(f"The AI Lesson Plan Architect is designing your lesson..."):
                        crew = LessonPlannerCrew(selected_model, language, grade_level, subject, topic)
                        st.session_state.lesson_plan = crew.run()

        if st.session_state.get('lesson_plan'):
            st.markdown("---");
            st.header(f"Your Lesson Plan: {topic}");
            st.markdown(st.session_state.lesson_plan)
            render_download_buttons(st.session_state.lesson_plan, f"lesson_plan_{topic.replace(' ', '_').lower()}")

    with tab2:
        st.header("Create a Custom Exam")
        if 'exam_paper' not in st.session_state: st.session_state.exam_paper = None
        available_models = get_available_models(st.session_state.get('gemini_key'))

        with st.form("exam_form"):
            col1, col2 = st.columns(2)
            country = col1.text_input("Country (for curriculum standards)", "USA", key="exam_country")
            grade_level_exam = col2.selectbox("Select Grade Level:", GRADE_LEVELS, key="exam_grade")

            col3, col4 = st.columns(2)
            subject_exam = col3.selectbox("Select Subject:", SUBJECTS, key="exam_subject")
            exam_type = col4.selectbox("Select Exam Type:", ["Intermediary Exam", "Final Exam"])

            col5, col6 = st.columns(2)
            language_exam = col5.selectbox("Language for the Exam:", LANGUAGES, key="exam_lang")
            selected_model_exam = col6.selectbox("Choose AI Model:", available_models,
                                                 key="exam_model") if available_models else None

            if st.form_submit_button(f"Generate {exam_type}", use_container_width=True):
                if not all([country, grade_level_exam, subject_exam, language_exam, selected_model_exam]):
                    st.error("Please fill all fields and select a model.")
                else:
                    with st.spinner(f"The AI Exam Committee is preparing your {exam_type}..."):
                        crew = ExamGeneratorCrew(selected_model_exam, language_exam, grade_level_exam, subject_exam,
                                                 exam_type, country)
                        st.session_state.exam_paper = crew.run()

        if st.session_state.get('exam_paper'):
            st.markdown("---");
            st.header(f"Your {subject_exam} {exam_type}");
            st.markdown(st.session_state.exam_paper)
            render_download_buttons(st.session_state.exam_paper,
                                    f"{subject_exam.lower()}_{exam_type.lower().replace(' ', '_')}")


# ==============================================================================
## This block allows the file to be run standalone for testing

class TravelCrew:
    def __init__(self, model_name, language,origin,destination,duration):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        os.environ["SERPER_API_KEY"] = st.session_state.get('serper_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.language = language
        self.origin = origin
        self.destination = destination
        self.duration = duration


    def run_planning_crew(self, transport_prefs, accommodation_prefs):
        agents = [
            Agent(role='Transportation Specialist',
                  goal=f"Find the best and cheapest transportation options from {self.origin} to {self.destination} based on the user's preferences: {', '.join(transport_prefs)}.",
                  backstory="You are an expert travel agent who specializes in finding the most efficient and cost-effective travel routes.",
                  llm=self.llm, tools=[SerperDevTool()], verbose=True),
            Agent(role='Accommodation Specialist',
                  goal=f"Find the best and cheapest accommodation options in {self.destination} for a {self.duration}-day stay, based on the user's preferences: {', '.join(accommodation_prefs)}. Provide a budget estimate per person and for a couple.",
                  backstory="You are a travel expert with a knack for finding great deals on places to stay.",
                  llm=self.llm, tools=[SerperDevTool()], verbose=True),
            Agent(role='Chief Itinerary Planner & Editor',
                  goal=f"Compile all the transportation and accommodation information into a single, cohesive travel plan in {self.language}.",
                  backstory="You are a meticulous travel planner who transforms raw data into a beautiful, actionable itinerary.",
                  llm=self.llm, verbose=True)
        ]

        tasks = []
        if transport_prefs:
            tasks.append(Task(
                description=f"Search for the best options for {', '.join(transport_prefs)} from {self.origin} to {self.destination}. Provide a summary including cost, duration, and a booking URL.",
                agent=agents[0],
                expected_output="A markdown section with a summary of the best transportation options."))
        if accommodation_prefs:
            tasks.append(Task(
                description=f"Search for the best options for {', '.join(accommodation_prefs)} in {self.destination} for a {self.duration}-day stay. Provide 2-3 top options with price, description, and a booking URL. Include a budget estimate per person and for a couple.",
                agent=agents[1],
                expected_output="A markdown section with a summary of the best accommodation options."))

        compile_task = Task(
            description=f"Combine all travel information into a single, final travel plan in {self.language}. Include a 'Best Travel Tips' section.",
            agent=agents[2], context=tasks,
            expected_output="The final, comprehensive travel plan formatted in clear markdown.",
            output_file="travel_plan.md")
        tasks.append(compile_task)

        crew = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("travel_plan.md", "r", encoding="utf-8") as f:
            return f.read()

    def run_activities_crew(self, interests):
        agents = [
            Agent(role='Local Tour Guide & Activities Specialist',
                  goal=f"Create a detailed, day-by-day itinerary of activities in {self.destination} for a {self.duration}-day trip, tailored to the user's interests: {interests}. Find price estimates for each activity.",
                  backstory=f"You are an expert local guide for {self.destination}. You know all the best attractions, hidden gems, restaurants, and activities, and can create a perfect itinerary for any interest.",
                  llm=self.llm, tools=[SerperDevTool()], verbose=True),
            Agent(role='Travel Experience Curator',
                  goal=f"Refine the itinerary to ensure a logical flow and a memorable experience. Add practical tips and booking information, highlighting any time-sensitive offers.",
                  backstory="You are a travel experience designer who turns a list of activities into an unforgettable journey. You focus on logistics, pacing, and adding special touches.",
                  llm=self.llm, verbose=True)
        ]

        task1 = Task(
            description=f"Develop a day-by-day itinerary for a {self.duration}-day trip to {self.destination}. If specific interests ('{interests}') are provided, prioritize them. If not, create a balanced itinerary. For each activity, provide a brief description and a price estimate.",
            agent=agents[0], expected_output="A detailed, day-by-day list of activities with price estimates.")
        task2 = Task(
            description=f"Review the itinerary. For any activities that can be booked (like tours, museums, or special restaurants), find and include a direct booking URL. If you find any time-sensitive offers (e.g., 'limited time', 'sells out fast'), highlight them with a warning to book soon. The final output must be in {self.language}.",
            agent=agents[1], context=[task1],
            expected_output="The final, comprehensive itinerary with booking links and tips, formatted in clear markdown.",
            output_file="activities_plan.md")

        crew = Crew(agents=agents, tasks=[task1, task2], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("activities_plan.md", "r", encoding="utf-8") as f:
            return f.read()

    def run_cruise_crew(self, region, departure_port, duration, cruise_line):
        agents = [
            Agent(role='Cruise Deal Specialist',
                  goal=f"Find the best and cheapest cruise deals for a {duration}-day trip in the {region} region, departing from {departure_port}. Prioritize offers from {cruise_line} if specified.",
                  backstory="You are an expert cruise travel agent with access to all major cruise line databases. You are a master at finding the best value, including special offers and package deals.",
                  llm=self.llm, tools=[SerperDevTool()], verbose=True),
            Agent(role='Cruise Travel Critic',
                  goal="Analyze the found cruise options and provide a recommendation. Highlight the best deal and provide practical tips for the chosen cruise.",
                  backstory="You are a seasoned travel critic who has reviewed hundreds of cruises. You know what makes a cruise special and can provide insider tips on how to get the most out of the experience.",
                  llm=self.llm, verbose=True),
            Agent(role='Itinerary Editor',
                  goal=f"Compile all the cruise information into a single, cohesive report in {self.language}.",
                  backstory="You are a meticulous editor who creates beautiful, easy-to-read travel summaries.",
                  llm=self.llm, verbose=True)
        ]

        task1 = Task(
            description=f"Search for the best cruise deals for a {duration}-day trip in the {region} region, departing from {departure_port}. If the user specified a preferred cruise line ({cruise_line}), focus on that. Find 2-3 top options.",
            agent=agents[0],
            expected_output="A list of 2-3 cruise options with details on the ship, itinerary, price, and a booking URL.")
        task2 = Task(
            description="Review the cruise options. Select the best value-for-money deal and write a brief review explaining why it's the top choice. Include 3-5 practical tips for the trip (e.g., what to pack, best excursions).",
            agent=agents[1], context=[task1], expected_output="A detailed recommendation and a list of practical tips.")
        task3 = Task(
            description=f"Combine the cruise options and the expert recommendation into a single, final report. The entire report must be in {self.language}.",
            agent=agents[2], context=[task1, task2],
            expected_output="The final, comprehensive cruise plan formatted in clear markdown.",
            output_file="cruise_plan.md")

        crew = Crew(agents=agents, tasks=[task1, task2, task3], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("cruise_plan.md", "r", encoding="utf-8") as f:
            return f.read()


# ==============================================================================
## 3. Page Rendering Function
# ==============================================================================

def render_travel_page():
    st.title("✈️ AI Travel Planner")
    st.markdown(
        "Your complete travel planning assistant. Get personalized plans for transport, accommodation, and activities.")

    st.warning(
        "**Disclaimer:** This AI provides travel suggestions based on real-time searches, but prices and availability can change rapidly. Always double-check details on the booking websites.",
        icon="❗")

    if 'travel_plan' not in st.session_state: st.session_state.travel_plan = None
    if 'activities_plan' not in st.session_state: st.session_state.activities_plan = None
    if 'cruise_plan' not in st.session_state: st.session_state.cruise_plan = None

    available_models = get_available_models(st.session_state.get('gemini_key'))
   # LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")

    tab1, tab2, tab3 = st.tabs(
        ["**Transport & Accommodation**", "**Activities & Itinerary**", "**Cruise Planner (Kreuzfahrt)**"])

    with tab1:
        st.subheader("Find Your Travel & Lodging")
        with st.form("planning_form"):
            col1, col2 = st.columns(2)
            origin = col1.text_input("From (City, Country)", st.session_state.get('origin', "Berlin, Germany"))
            destination = col2.text_input("To (City, Country)", st.session_state.get('destination', "Paris, France"))
            duration = st.number_input("Duration of Stay (days)", min_value=1, max_value=30,
                                       value=st.session_state.get('duration', 7))
            transport_prefs = st.multiselect("Preferred Transport", ["Flights", "Trains", "Buses"], default=["Flights"])
            accommodation_prefs = st.multiselect("Preferred Accommodation", ["Hotel", "Hostel", "Camping", "Any"],
                                                 default=["Hotel"])
            language = st.selectbox("Language for the Plan:", LANGUAGES, key="plan_lang")
            selected_model = st.selectbox("Choose AI Model:", available_models,
                                          key="plan_model") if available_models else None

            if st.form_submit_button("Generate Travel Plan", use_container_width=True):
                if not all([origin, destination, language, selected_model]):
                    st.error("Please fill all fields and select a model.")
                else:
                    st.session_state.update(origin=origin, destination=destination, duration=duration)
                    with st.spinner(f"The AI travel agency is planning your trip to {destination}..."):
                        crew = TravelCrew(selected_model, language, origin, destination, duration)
                        st.session_state.travel_plan = crew.run_planning_crew(transport_prefs, accommodation_prefs)

        if st.session_state.get('travel_plan'):
            st.markdown("---")
            st.header(f"Your Travel Plan: {st.session_state.origin} to {st.session_state.destination}")
            st.markdown(st.session_state.travel_plan)
            render_download_buttons(st.session_state.travel_plan,
                                    f"travel_plan_{st.session_state.destination.replace(' ', '_').lower()}")

    with tab2:
        st.subheader("Design Your Daily Itinerary")
        with st.form("activities_form"):
            st.info(
                f"Currently planning for a **{st.session_state.get('duration', 7)}-day trip** to **{st.session_state.get('destination', 'N/A')}**.")
            interests = st.text_input("Enter your interests (optional, comma-separated)",
                                      placeholder="e.g., hiking, museums, shopping, restaurants, churches")
            language_act = st.selectbox("Language for the Itinerary:", LANGUAGES, key="act_lang")
            selected_model_act = st.selectbox("Choose AI Model:", available_models,
                                              key="act_model") if available_models else None

            if st.form_submit_button("Generate Activities Plan", use_container_width=True):
                if not all([st.session_state.get('destination'), language_act, selected_model_act]):
                    st.error("Please ensure trip details are set in the first tab and a model is selected.")
                else:
                    with st.spinner(
                            f"The AI tour guide is creating your itinerary for {st.session_state.destination}..."):
                        crew = TravelCrew(selected_model_act, language_act, st.session_state.origin,
                                          st.session_state.destination, st.session_state.duration)
                        st.session_state.activities_plan = crew.run_activities_crew(interests)

        if st.session_state.get('activities_plan'):
            st.markdown("---")
            st.header(f"Your Itinerary for {st.session_state.destination}")
            st.markdown(st.session_state.activities_plan)
            render_download_buttons(st.session_state.activities_plan,
                                    f"itinerary_{st.session_state.destination.replace(' ', '_').lower()}")

    with tab3:
        st.subheader("Find Your Perfect Cruise")
        with st.form("cruise_form"):
            region = st.selectbox("Select Cruise Region:",
                                  ["Caribbean", "Mediterranean", "Alaska", "Norwegian Fjords", "Hawaii",
                                   "Mexican Riviera",
                                   "South America", "Northern Europe & Baltic Sea", "Australia & New Zealand",
                                   "Asia (Japan & Southeast)", "Panama Canal",
                                   "Galapagos Islands"])
            departure_port = st.text_input("Preferred Departure Port (optional)", placeholder="e.g., Miami, Barcelona")
            duration_cruise = st.slider("Cruise Duration (days)", 3, 14, 7)
            cruise_line = st.text_input("Preferred Cruise Line (optional)", placeholder="e.g., Royal Caribbean, MSC")
            language_cruise = st.selectbox("Language for the Plan:", LANGUAGES, key="cruise_lang")
            selected_model_cruise = st.selectbox("Choose AI Model:", available_models,
                                                 key="cruise_model") if available_models else None

            if st.form_submit_button("Find Cruise Deals", use_container_width=True):
                if not selected_model_cruise:
                    st.error("Please select a model.")
                else:
                    with st.spinner(f"The AI cruise specialist is searching for deals in the {region}..."):
                        crew = TravelCrew(selected_model_cruise, language_cruise, "", "",
                                          0)  # Origin/Dest not needed for cruise search
                        st.session_state.cruise_plan = crew.run_cruise_crew(region, departure_port, duration_cruise,
                                                                            cruise_line)

        if st.session_state.get('cruise_plan'):
            st.markdown("---")
            st.header(f"Your Cruise Plan for the {region}")
            st.markdown(st.session_state.cruise_plan)
            render_download_buttons(st.session_state.cruise_plan, f"cruise_plan_{region.replace(' ', '_').lower()}")


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
            task_desc_comprehension = f"Write one short text (approx. 200 words) in {self.target_language} and create 5 multiple-choice questions to test comprehension, following Goethe-Institut standards. Instructions, questions and answers must be   be in {self.target_language}."
            output_filename = "language_exercises.md"
        else:  # Final Exam
            task_desc_grammar = f"Create a 'Grammar Section' for a {self.level} final exam, following Goethe-Institut standards. It should contain 20-25 challenging questions covering a wide range of grammar topics suitable for this level. Instructions in {self.target_language}."
            task_desc_comprehension = f"Create a 'Reading Comprehension & Essay' section for a {self.level} final exam, following Goethe-Institut standards. Write one text (approx. 300 words) in {self.target_language}, followed by 5 comprehension questions and one essay prompt. Instructions  and questions in {self.target_language}."
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
                  goal=f"Generate a list of at least 150 essential vocabulary words in {self.target_language} related to '{self.scope}' for a {self.level} learner.",
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
            description=f"Create a list of at least 150 vocabulary words in {self.target_language} for a {self.level} learner, focusing on the topic related to '{self.scope}'.",
            agent=agents[0], expected_output="A clean list of 150+ words in {self.target_language}.")
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
            description=f"Write a text passage of appropriate length (A1/A2: ~200 words, B1/B2: ~400 words, C1/C2: ~500 words) in {self.target_language} about '{self.scope}' for a {self.level} learner.",
            agent=agents[0],
            expected_output="A well-written text passage in markdown format."
        )
        task_create_questions = Task(
            description=f"Based on the provided text, create 8-10 reading comprehension questions that meet Goethe-Institut standards for level {self.level}. Include a mix of question types. The questions and instructions must be in {self.target_language}.",
            agent=agents[1],
            context=[task_write_text],
            expected_output="A complete set of questions and a separate answer key in markdown."
        )
        task_compile = Task(
            description="Combine the text passage and the questions into a single document. Then, create a separate section at the end titled 'Answer Key' with clear solutions for all questions.",
            agent=agents[2],
            context=[task_write_text, task_create_questions],
            expected_output="A complete, beautifully formatted markdown document containing the full reading comprehension exercise and a comprehensive answer key.",
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


# ==============================================================================
## 3. Page Rendering Function
# ==============================================================================

def render_language_academy_page():
    st.title("🗣️ AI Language Academy")
    st.markdown("Your interactive hub for mastering a new language.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["**Study Guide**", "**Vocabulary Builder**", "**Grammar Deep Dive**", "**Listening Practice**",
         "**Reading Comprehension**", "**Practice & Exams**"])

    with tab1:
        st.header("Generate a Comprehensive Study Guide")
        if 'language_guide' not in st.session_state: st.session_state.language_guide = None
        available_models = get_available_models(st.session_state.get('gemini_key'))
       # LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")

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

    with tab2:
        st.header("Build Your Thematic Vocabulary List")
        if 'vocabulary_list' not in st.session_state: st.session_state.vocabulary_list = None
        available_models = get_available_models(st.session_state.get('gemini_key'))
        #LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")

        SCOPES = [
            "Personal Information & Greetings", "Family & Friends", "Numbers, Dates, & Time", "Food & Drink",
            "At Home (Rooms & Furniture)","Greetings","Hospital","Church","Bible","Holiday","Hobbies","Animals","Grammar",
            "Daily Routines", "Clothing & Shopping", "Weather & Seasons", "The Body & Health", "Hobbies & Free Time",
            "Basic Travel & Directions","Hotel","Holiday", "Politic","Music","Christianity",
            "Work & Professions", "Education & University", "Technology & The Internet", "Media & News",
            "Environment & Nature",
            "Culture & Traditions", "Politics & Society", "Feelings & Emotions", "Travel & Tourism (Advanced)",
            "Health & Fitness",
            "Business & Finance", "Science & Research", "Law & Justice", "Arts & Literature", "History & Archaeology",
            "Philosophy & Abstract Concepts"
        ]

        with st.form("vocab_form"):
            col1, col2 = st.columns(2)
            native_language = col1.text_input("Your Language", "English", key="vocab_native")
            target_language = col2.text_input("Language to Learn", "French", key="vocab_target")

            col3, col4 = st.columns(2)
            level = col3.selectbox("Select Your Level (CEFR)", ["A1", "A2", "B1", "B2", "C1", "C2"], key="vocab_level")
            scope = col4.selectbox("Select a Vocabulary Scope", SCOPES)

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
                if f"{native_language}"=="English":
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

    with tab3:
        st.header("Master Specific Grammar Topics")
        if 'grammar_guide' not in st.session_state: st.session_state.grammar_guide = None
        available_models = get_available_models(st.session_state.get('gemini_key'))
        LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")
        GRAMMAR_TOPICS = [
            "Articles (Definite/Indefinite)", "Nouns (Gender/Plurals)", "Present Tense (10 Regular Verbs)",
            "Present Tense (10 Irregular Verbs)",
            "Basic Sentence Structure (SVO)",
            "Personal Pronouns", "Possessive Adjectives", "Prepositions of Place", "Question Formation",
            "The Verb 'to be'", "The Verb 'to have'",
            "Past Tenses (e.g., Simple Past, Perfect)", "Future Tenses", "Modal Verbs", "Reflexive Verbs",
            "Comparative & Superlative",
            "Relative Clauses", "Conditional Sentences (Type 1 & 2)", "Conjunctions", "The Passive Voice",
            "Subjunctive Mood", "Conditional Sentences (Type 3)", "Advanced Prepositions",
            "Complex Sentence Structures", "Idiomatic Expressions", "Participles as Adjectives"
        ]

        with st.form("grammar_form"):
            col1, col2 = st.columns(2)
            native_language = col1.text_input("Your Language", "English", key="gram_native")
            target_language = col2.text_input("Language to Learn", "German", key="gram_target")

            level = st.selectbox("Select Your Proficiency Level (CEFR)", ["A1", "A2", "B1", "B2", "C1", "C2"],
                                 key="gram_level")
            selected_topics = st.multiselect("Select Grammar Topics to Study", GRAMMAR_TOPICS,
                                             default=["Articles (Definite/Indefinite)",
                                                      "Present Tense (10 Regular Verbs)","Present Tense (10 Irregular Verbs)"])

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

    with tab4:
        st.header("Create a Custom Listening Exercise")
        from google import genai as gen
        from google.genai import types
        if 'listening_material' not in st.session_state: st.session_state.listening_material = None
        available_models = get_available_models(st.session_state.get('gemini_key'))
        tts_models = get_available_models(st.session_state.get('gemini_key'), task="text-to-speech")
        voices = ["Kore", "Puck", "Chipp", "Sadachbia", "Lyra", "Arpy", "Fable", "Onyx"]
        SCOPES = [
            "Personal Information & Greetings", "Family & Friends", "Food & Drink", "At Home",
            "Daily Routines", "Shopping", "Weather & Seasons", "Health", "Hobbies & Free Time", "Travel & Directions",
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
                topic_listen = st.selectbox("Select a Topic Scope", SCOPES)
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


    with tab5:
        st.header("Improve Your Reading Comprehension")
        if 'comprehension_material' not in st.session_state: st.session_state.comprehension_material = None
        available_models = get_available_models(st.session_state.get('gemini_key'))
        LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")
        SCOPES = [
            "Personal Information & Greetings", "Family & Friends", "Numbers, Dates, & Time", "Food & Drink",
            "At Home (Rooms & Furniture)",
            "Daily Routines", "Clothing & Shopping", "Weather & Seasons", "The Body & Health", "Hobbies & Free Time",
            "Basic Travel & Directions",
            "Work & Professions", "Education & University", "Technology & The Internet", "Media & News",
            "Environment & Nature",
            "Culture & Traditions", "Politics & Society", "Feelings & Emotions", "Travel & Tourism (Advanced)",
            "Health & Fitness",
            "Business & Finance", "Science & Research", "Law & Justice", "Arts & Literature", "History & Archaeology",
            "Philosophy & Abstract Concepts"
        ]

        with st.form("comprehension_form"):
            col1, col2 = st.columns(2)
            native_language = col1.text_input("Your Language", "English", key="comp_native")
            target_language = col2.text_input("Language to Learn", "German", key="comp_target")

            col3, col4 = st.columns(2)
            level = col3.selectbox("Select Your Level (CEFR)", ["A1", "A2", "B1", "B2", "C1", "C2"], key="comp_level")
            scope = col4.selectbox("Select a Topic for the Text", SCOPES)

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

    with tab6:
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
                grammar_topics_selection = st.multiselect("Select Grammar Topics for Exercises", GRAMMAR_TOPICS,
                                                          default=["Present Tense (10 Regular Verbs)"])

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

# =================================================























































































































