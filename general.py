# main.py
# Final, unified, and production-ready application code for the AI Health & Fitness Suite.

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
from main_v2 import  get_available_models, render_download_buttons



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


class FitnessCrew:
    def __init__(self, model_name, language, weight, height, goal, country):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        os.environ["SERPER_API_KEY"] = st.session_state.get('serper_key', '')
        self.llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        self.language = language
        self.weight = weight
        self.height = height
        self.goal = goal
        self.country = country

    def run(self):
        agents = [
            Agent(
                role='Certified Personal Trainer',
                goal=f"Create a detailed, day-by-day workout plan for a user with the goal of '{self.goal}'. The user's weight is {self.weight} kg and height is {self.height} cm.",
                backstory="You are a highly experienced personal trainer who creates safe, effective, and personalized workout plans. You break down routines into daily schedules with clear instructions on exercises, sets, reps, and rest periods.",
                llm=self.llm,
                verbose=True
            ),
            Agent(
                role='Sports Nutritionist',
                goal=f"Develop a complementary nutrition plan to support the user's fitness goal of '{self.goal}'. The plan should consider general dietary habits in {self.country}.",
                backstory="You are a certified nutritionist who specializes in creating meal plans for athletes and fitness enthusiasts. You provide practical, easy-to-follow dietary advice that aligns with specific fitness goals.",
                llm=self.llm,
                tools=[SerperDevTool()],
                verbose=True
            ),
            Agent(
                role='Sports Psychologist',
                goal="Provide motivational tips and strategies to help the user stay consistent and overcome mental hurdles.",
                backstory="You are a sports psychologist who helps athletes build mental toughness and maintain motivation. You provide actionable tips on goal setting, consistency, and building a resilient mindset.",
                llm=self.llm,
                verbose=True
            ),
            Agent(
                role='Fitness Content Editor',
                goal=f"Compile all the workout, nutrition, and motivational advice into a single, cohesive, and easy-to-read fitness plan in {self.language}.",
                backstory="You are a meticulous editor for a leading fitness magazine. You ensure every plan is well-structured, clearly explained, and motivating for the reader.",
                llm=self.llm,
                verbose=True
            )
        ]
        task_workout = Task(
            description=f"Create a 7-day workout schedule tailored for a user aiming for '{self.goal}'. Specify exercises, duration, sets, and reps for each day. Include warm-up and cool-down routines.",
            agent=agents[0],
            expected_output="A detailed markdown table or list outlining the 7-day workout plan."
        )
        task_nutrition = Task(
            description=f"Create a nutrition guideline that supports the goal of '{self.goal}'. Provide examples of breakfast, lunch, dinner, and snacks. Consider common foods available in {self.country}.",
            agent=agents[1],
            expected_output="A markdown section with dietary recommendations and sample meal ideas."
        )
        task_motivation = Task(
            description="Write a short section with 3-5 powerful motivational tips on how to stay consistent with the fitness plan.",
            agent=agents[2],
            expected_output="A concise, encouraging markdown section with motivational advice."
        )
        task_compile = Task(
            description=f"Combine the workout plan, nutrition guide, and motivational tips into a single, final fitness plan. Format it beautifully in markdown with a main title: 'Your Personalized Fitness Plan for {self.goal}'. The entire guide must be in {self.language}.",
            agent=agents[3],
            context=[task_workout, task_nutrition, task_motivation],
            expected_output="The final, comprehensive fitness plan formatted in clear markdown.",
            output_file="fitness_plan.md"
        )

        crew = Crew(
            agents=agents,
            tasks=[task_workout, task_nutrition, task_motivation, task_compile],
            process=Process.sequential,
            verbose=True
        )

        crew.kickoff()
        with open("fitness_plan.md", "r", encoding="utf-8") as f:
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
    st.markdown("Get a personalized workout and nutrition plan from an expert AI crew.")

    st.warning(
        """
        **⚠️ IMPORTANT HEALTH DISCLAIMER**

        This AI provides fitness and nutrition suggestions for informational purposes only. It is **NOT** a substitute for professional medical advice. 

        **Consult with a qualified healthcare professional or a certified personal trainer** before beginning any new fitness program or making changes to your diet, especially if you have pre-existing health conditions.
        """,
        icon="❗"
    )

    if 'fitness_plan' not in st.session_state:
        st.session_state.fitness_plan = None

    available_models = get_available_models(st.session_state.get('gemini_key'))
    LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")

    with st.form("fitness_form"):
        st.header("Tell Us About Your Goals")
        col1, col2 = st.columns(2)
        weight = col1.number_input("Your Weight (kg)", min_value=30, max_value=200, value=70)
        height = col2.number_input("Your Height (cm)", min_value=100, max_value=250, value=175)

        goal = st.selectbox("What is your primary fitness goal?",
                            ["Weight Loss", "Muscle Building", "Six-Pack Abs", "General Fitness & Endurance"])

        col3, col4 = st.columns(2)
        country = col3.text_input("Your Country", placeholder="e.g., USA, Germany")
        language = col4.selectbox("Language for the Plan:", LANGUAGES)

        selected_model = st.selectbox("Choose AI Model:", available_models) if available_models else None

        if st.form_submit_button("Generate My Fitness Plan", use_container_width=True):
            if not all([weight, height, goal, country, language, selected_model]):
                st.error("Please fill all fields and select a model.")
            else:
                with st.spinner(f"The AI fitness team is creating your plan for {goal}..."):
                    crew = FitnessCrew(selected_model, language, weight, height, goal, country)
                    st.session_state.fitness_plan = crew.run()

    if st.session_state.get('fitness_plan'):
        st.markdown("---")
        st.header(f"Your Personalized Fitness Plan")
        st.markdown(st.session_state.fitness_plan)
        render_download_buttons(st.session_state.fitness_plan, f"fitness_plan_{goal.replace(' ', '_').lower()}")


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
    LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")

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


# ==============================================================================
## 4. Main Application Router
# ==============================================================================

def main():
    st.set_page_config(page_title="AI Health & Fitness Suite", layout="wide")

    st.sidebar.title("🔐 Central Configuration")
    st.session_state['gemini_key'] = st.sidebar.text_input("Google Gemini API Key", type="password",
                                                           value=st.session_state.get('gemini_key', ''))
    st.session_state['serper_key'] = st.sidebar.text_input("Serper.dev API Key", type="password",
                                                           value=st.session_state.get('serper_key', ''))
    st.sidebar.markdown("---")

    st.sidebar.title("Navigation")
    page_options = {
        "AI Swimming Coach": "🏊",
        "AI Fitness Trainer": "🏋️",
        "AI Driving License Guide": "🚗"
    }
    selection = st.sidebar.radio("Go to", list(page_options.keys()))

    keys_needed = {
        "AI Swimming Coach": ['gemini_key', 'serper_key'],
        "AI Fitness Trainer": ['gemini_key', 'serper_key'],
        "AI Driving License Guide": ['gemini_key', 'serper_key']
    }

    if not all(st.session_state.get(key) for key in keys_needed.get(selection, [])):
        st.warning(f"Please enter the required API Key(s) to use the {selection}.");
        st.stop()

    if selection == "AI Swimming Coach":
        render_swimming_page()
    elif selection == "AI Fitness Trainer":
        render_fitness_page()
    elif selection == "AI Driving License Guide":
        render_driving_license_page()


if __name__ == "__main__":
    main()

#






















































































































