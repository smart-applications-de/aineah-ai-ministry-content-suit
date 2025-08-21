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
from main_v2 import  get_available_models, render_download_buttons,LANGUAGES



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
# ==============================================================================
if __name__ == '__main__':
    st.set_page_config(page_title="AI Education Studio", layout="wide")

    st.sidebar.title("🔐 Central Configuration")
    st.session_state['gemini_key'] = st.sidebar.text_input("Google Gemini API Key", type="password",
                                                           value=st.session_state.get('gemini_key', ''))
    st.session_state['serper_key'] = st.sidebar.text_input("Serper.dev API Key", type="password",
                                                           value=st.session_state.get('serper_key', ''))

    if not all([st.session_state.get(k) for k in ['gemini_key', 'serper_key']]):
        st.warning("Please enter your Gemini and Serper API Keys in the sidebar to use this page.")
        st.stop()

    render_lesson_planner_page()


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























































































































