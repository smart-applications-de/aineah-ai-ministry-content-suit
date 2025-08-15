# stock_analyzer.py
# A self-contained Streamlit page for the AI Stock Analysis Studio with a new News & Insights tab.

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
## 2. AI Crew Class
# ==============================================================================

class StockAnalysisCrew:
    def __init__(self, model_name, ticker):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        os.environ["SERPER_API_KEY"] = st.session_state.get('serper_key', '')
        self.llm = LLM(model=model_name, temperature=0.3, api_key=os.environ["GOOGLE_API_KEY"])
        self.ticker = ticker

    def run_analysis_crew(self):
        agents = [
            Agent(role='Financial Data Analyst', goal=f"Fetch and process historical stock data for {self.ticker}.",
                  backstory="An expert data analyst proficient in using yfinance.", llm=self.llm, verbose=True),
            Agent(role='Fundamental Analyst',
                  goal=f"Analyze the company profile and financial health of {self.ticker}.",
                  backstory="A seasoned fundamental analyst specializing in company valuations.", llm=self.llm,
                  tools=[SerperDevTool()], verbose=True),
            Agent(role='Technical Analyst', goal=f"Analyze price trends and technical indicators for {self.ticker}.",
                  backstory="A sharp technical analyst who uses historical data to forecast price movements.",
                  llm=self.llm, verbose=True),
            Agent(role='Chief Investment Strategist',
                  goal=f"Synthesize all analysis to provide a final investment recommendation for {self.ticker}.",
                  backstory="A respected investment strategist who makes clear, actionable recommendations.",
                  llm=self.llm, verbose=True)
        ]
        task1 = Task(
            description=f"Fetch the last 30 days of historical data for {self.ticker} and calculate the 20-day moving average.",
            agent=agents[0], expected_output="A summary of key data points including latest price and moving average.")
        task2 = Task(description=f"Research and provide a detailed analysis of the company profile for {self.ticker}.",
                     agent=agents[1], expected_output="A concise summary of the company's profile and business model.")
        task3 = Task(description="Analyze the historical data to identify trends.", agent=agents[2], context=[task1],
                     expected_output="A brief technical analysis highlighting key trends.")
        task4 = Task(
            description="Based on all analysis, provide a final investment recommendation (Buy, Hold, or Sell) with a justification.",
            agent=agents[3], context=[task2, task3],
            expected_output="A final recommendation with a clear justification.", output_file="stock_recommendation.md")

        crew = Crew(agents=agents, tasks=[task1, task2, task3, task4], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("stock_recommendation.md", "r", encoding="utf-8") as f:
            return f.read()

    def run_news_crew(self):
        agents = [
            Agent(role='Financial News Analyst',
                  goal=f"Find the latest news, articles, and press releases for {self.ticker}.",
                  backstory="A skilled news analyst who can quickly find the most relevant and impactful financial news.",
                  llm=self.llm, tools=[SerperDevTool()], verbose=True),
            Agent(role='Financial Reporter',
                  goal=f"Synthesize the gathered news into a concise, insightful article for an investor.",
                  backstory="An experienced financial journalist who can turn raw news into a compelling narrative, highlighting key takeaways and providing sources.",
                  llm=self.llm, verbose=True)
        ]
        task1 = Task(
            description=f"Search for the latest news, earnings reports, and press releases for {self.ticker} from the past month.",
            agent=agents[0], expected_output="A list of 3-5 key news items with headlines, summaries, and source URLs.")
        task2 = Task(
            description="Synthesize the news items into a single, insightful article. The article should summarize the key developments and their potential impact on the stock. Conclude with a 'Further Reading' section that lists the source URLs.",
            agent=agents[1], context=[task1],
            expected_output="A well-written article summarizing the latest news, with a list of sources.",
            output_file="stock_news_article.md")

        crew = Crew(agents=agents, tasks=[task1, task2], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("stock_news_article.md", "r", encoding="utf-8") as f:
            return f.read()


# ==============================================================================
## 2. AI Crew Class
# ==============================================================================

class HealthCrew:
    def __init__(self, model_name, language):
        os.environ["GOOGLE_API_KEY"] = st.session_state.get('gemini_key', '')
        os.environ["SERPER_API_KEY"] = st.session_state.get('serper_key', '')
        self.llm = LLM(model=model_name, temperature=0.3, api_key=os.environ["GOOGLE_API_KEY"])
        self.language = language

    def run_medical_crew(self, query, country):
        agents = [
            Agent(role='General Practitioner AI',
                  goal="Analyze a user's medical query to form a preliminary assessment.",
                  backstory="An AI assistant modeled after an experienced General Practitioner.", llm=self.llm,
                  verbose=True),
            Agent(role='Medical Researcher AI',
                  goal=f"Find credible, evidence-based medical information relevant to {country}.",
                  backstory="An AI research assistant specializing in medical literature.", llm=self.llm,
                  tools=[SerperDevTool()], verbose=True),
            Agent(role='Specialist Doctor AI',
                  goal="Synthesize findings to provide a detailed explanation. You MUST NOT give a diagnosis or prescription.",
                  backstory="An AI assistant modeled after a specialist doctor who emphasizes safety.", llm=self.llm,
                  verbose=True),
            Agent(role='Patient Advocate AI',
                  goal=f"Compile all information into a clear, compassionate report in {self.language}, starting with a strong disclaimer.",
                  backstory="An AI assistant designed to be a patient advocate, ensuring clarity and safety.",
                  llm=self.llm, verbose=True)
        ]
        task1 = Task(
            description=f"Analyze the user's query: '{query}'. Identify key symptoms and the primary medical area of concern.",
            agent=agents[0], expected_output="A brief summary of symptoms and the identified medical field.")
        task2 = Task(
            description=f"Based on the preliminary assessment, research potential conditions and general health advice relevant to {country}.",
            agent=agents[1], context=[task1], expected_output="A structured report of findings.")
        task3 = Task(
            description="Synthesize the research to explain potential causes and suggest general wellness tips. Explicitly state this is not a diagnosis.",
            agent=agents[2], context=[task2], expected_output="A detailed medical explanation.")
        task4 = Task(
            description=f"Compile a final report in {self.language}, starting with a mandatory disclaimer and ending with a strong recommendation to see a real doctor.",
            agent=agents[3], context=[task3], expected_output="The final, user-friendly report.",
            output_file="medical_advice.md")

        crew = Crew(agents=agents, tasks=[task1, task2, task3, task4], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("medical_advice.md", "r", encoding="utf-8") as f:
            return f.read()

    def run_mental_health_crew(self, query, focus_area):
        agents = [
            Agent(role='Psychology Advisor AI',
                  goal=f"Analyze the user's query about {focus_area} to provide general, evidence-based wellness strategies.",
                  backstory="An AI assistant modeled after a compassionate psychologist who provides helpful, safe, and general advice.",
                  llm=self.llm, tools=[SerperDevTool()], verbose=True),
            Agent(role='Pastoral Counselor AI',
                  goal=f"Provide biblical wisdom and encouragement for someone struggling with {focus_area}.",
                  backstory="An AI assistant with the heart of a Pentecostal pastor, offering hope found in Jesus.",
                  llm=self.llm, verbose=True),
            Agent(role='Resource Curator AI',
                  goal=f"Compile all advice into a single, compassionate report in {self.language}, starting with a strong safety disclaimer.",
                  backstory="An AI assistant designed to be a patient advocate, emphasizing professional consultation.",
                  llm=self.llm, verbose=True)
        ]
        task1 = Task(
            description=f"Analyze the user's query: '{query}'. Outline 3-4 general wellness strategies for {focus_area}.",
            agent=agents[0], expected_output="A summary of psychological themes and a list of wellness tips.")
        task2 = Task(
            description=f"Based on the user's struggle with {focus_area}, provide pastoral encouragement with 3-5 relevant Bible verses.",
            agent=agents[1], expected_output="A compassionate, faith-based message with scripture references.")
        task3 = Task(
            description=f"Compile a final report in {self.language}, starting with a mandatory disclaimer and ending with a strong recommendation to seek professional help.",
            agent=agents[2], context=[task1, task2],
            expected_output="The final, user-friendly report with actionable next steps.",
            output_file="mental_health_support.md")

        crew = Crew(agents=agents, tasks=[task1, task2, task3], process=Process.sequential, verbose=True)
        crew.kickoff()
        with open("mental_health_support.md", "r", encoding="utf-8") as f:
            return f.read()


# ==============================================================================
## 3. Page Rendering Function
# ==============================================================================

def render_stock_analyzer_page():
    st.title("📈 AI Stock Analysis Studio")
    st.markdown("Get a comprehensive analysis and the latest news for any stock ticker.")

    if 'stock_analysis' not in st.session_state: st.session_state.stock_analysis = None
    if 'stock_news' not in st.session_state: st.session_state.stock_news = None

    available_models = get_available_models(st.session_state.get('gemini_key'))

    ticker = st.text_input("Enter a stock ticker symbol:", placeholder="e.g., GOOGL, MSFT, AAPL").upper()

    tab1, tab2 = st.tabs(["**Analysis & Recommendation**", "**Latest News & Insights**"])

    with tab1:
        st.header("Generate a Full Stock Analysis")
        with st.form("analysis_form"):
            selected_model_analysis = st.selectbox("Choose AI Model for Analysis:", available_models,
                                                   key="analysis_model") if available_models else None
            if st.form_submit_button("Analyze Stock", use_container_width=True):
                if not all([ticker, selected_model_analysis]):
                    st.error("Please enter a ticker and select a model.")
                else:
                    with st.spinner(f"The AI Investment Committee is analyzing {ticker}..."):
                        stock = yf.Ticker(ticker)
                        hist_data = stock.history(period="30d")
                        if not hist_data.empty:
                            hist_data['Mean'] = hist_data['Close'].mean()
                            hist_data['20-Day MA'] = hist_data['Close'].rolling(window=20).mean()
                            st.session_state.stock_data = hist_data.tail(20)
                            info = stock.info
                            st.session_state.company_info = {"name": info.get('longName'), "sector": info.get('sector'),
                                                             "summary": info.get('longBusinessSummary'),
                                                             "website": info.get('website')}
                            crew = StockAnalysisCrew(selected_model_analysis, ticker)
                            st.session_state.stock_analysis = crew.run_analysis_crew()
                        else:
                            st.error(f"Could not fetch data for ticker: {ticker}.")

        if st.session_state.get('stock_analysis'):
            st.markdown("---")
            st.header(f"Analysis for {st.session_state.company_info['name']} ({ticker})")
            st.subheader("Company Profile")
            st.markdown(f"**Sector:** {st.session_state.company_info['sector']}")
            st.markdown(f"[{st.session_state.company_info['website']}]({st.session_state.company_info['website']})")
            with st.expander("Read Business Summary"): st.write(st.session_state.company_info['summary'])
            st.subheader("Investment Recommendation")
            st.info(st.session_state.stock_analysis)
            st.subheader("Recent Historical Data")
            st.dataframe(st.session_state.stock_data)
            render_download_buttons(st.session_state.stock_analysis, f"{ticker}_analysis")

    with tab2:
        st.header("Get the Latest News and Insights")
        with st.form("news_form"):
            selected_model_news = st.selectbox("Choose AI Model for News Summary:", available_models,
                                               key="news_model") if available_models else None
            if st.form_submit_button("Fetch Latest News", use_container_width=True):
                if not all([ticker, selected_model_news]):
                    st.error("Please enter a ticker and select a model.")
                else:
                    with st.spinner(f"The AI Newsroom is gathering intelligence on {ticker}..."):
                        crew = StockAnalysisCrew(selected_model_news, ticker)
                        st.session_state.stock_news = crew.run_news_crew()

        if st.session_state.get('stock_news'):
            st.markdown("---")
            st.header(f"Latest News Summary for {ticker}")
            st.markdown(st.session_state.stock_news)
            render_download_buttons(st.session_state.stock_news, f"{ticker}_news_summary")


# ==============================================================================
## 3. Page Rendering Function
# ==============================================================================

def render_health_support_page():
    st.title("❤️‍🩹 AI Health & Wellness Suite")
    st.markdown("A suite for preliminary health information and pastoral encouragement.")

    st.error(
        """
        **⚠️ IMPORTANT SAFETY & MEDICAL DISCLAIMER: FOR INFORMATIONAL PURPOSES ONLY**

        This tool is powered by AI and is **NOT** a substitute for professional medical or psychological advice, diagnosis, or treatment. 

        **Always seek the advice of your physician or another qualified health provider** with any questions you may have regarding a medical condition. If you are in crisis, please contact a local emergency service immediately.
        """
    )

    tab1, tab2 = st.tabs(["**Medical Consultation (Physical Health)**", "**Pastoral & Mental Health Support**"])

    with tab1:
        st.header("Get Information on Physical Health Symptoms")
        if 'medical_advice' not in st.session_state: st.session_state.medical_advice = None
        available_models = get_available_models(st.session_state.get('gemini_key'))
        LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")

        with st.form("medical_form"):
            query = st.text_area("Please describe your symptoms or medical question in detail:", height=150,
                                 placeholder="e.g., I have had a persistent dry cough and a slight fever for the past three days...")
            col1, col2 = st.columns(2)
            country = col1.text_input("Your Country", placeholder="e.g., Germany, Kenya")
            language = col2.selectbox("Language for the Response:", LANGUAGES, key="med_lang")
            selected_model = st.selectbox("Choose AI Model:", available_models,
                                          key="med_model") if available_models else None

            if st.form_submit_button("Get AI-Powered Information", use_container_width=True):
                if not all([query, country, language, selected_model]):
                    st.error("Please fill all fields and select a model.")
                else:
                    with st.spinner("Your AI medical team is reviewing your query..."):
                        crew = HealthCrew(selected_model, language)
                        st.session_state.medical_advice = crew.run_medical_crew(query, country)

        if st.session_state.get('medical_advice'):
            st.markdown("---")
            st.header("Your AI-Generated Health Information Report")
            st.markdown(st.session_state.medical_advice)
            render_download_buttons(st.session_state.medical_advice, "medical_advice")
            if st.button(f"🎧 Listen to this AI-Generated Health Information Report"):
                st.session_state['text_for_audio'] = st.session_state.medical_advice
                st.info("Go to the 'Audio Suite' page to generate the audio.")

    with tab2:
        st.header("Get Pastoral Encouragement & Wellness Tips")
        if 'mental_health_support' not in st.session_state: st.session_state.mental_health_support = None
        available_models = get_available_models(st.session_state.get('gemini_key'))

        with st.form("mental_health_form"):
            focus_area = st.selectbox("What is the main area of your concern?",
                                      ["Depression", "Anxiety", "Sleep Disorder", "General Stress", "Grief"])
            query = st.text_area("Please describe what you're experiencing or the question you have:", height=150,
                                 placeholder="e.g., I've been feeling overwhelmed with worry lately and can't seem to quiet my thoughts...")
            col1, col2 = st.columns(2)
            language = col1.selectbox("Language for the Response:", LANGUAGES, key="mh_lang")
            selected_model = col2.selectbox("Choose AI Model:", available_models,
                                            key="mh_model") if available_models else None

            if st.form_submit_button("Get Pastoral & Wellness Advice", use_container_width=True):
                if not all([query, language, selected_model]):
                    st.error("Please describe your concern, select a language, and choose a model.")
                else:
                    with st.spinner("Your AI care team is preparing a thoughtful response..."):
                        crew = HealthCrew(selected_model, language)
                        st.session_state.mental_health_support = crew.run_mental_health_crew(query, focus_area)

        if st.session_state.get('mental_health_support'):
            st.markdown("---")
            st.header("A Caring Response for You")
            st.markdown(st.session_state.mental_health_support)
            render_download_buttons(st.session_state.mental_health_support, "mental_health_support")
            if st.button(f"🎧 Listen to this AI-Generated Mental Health Support"):
                st.session_state['text_for_audio'] = st.session_state.mental_health_support
                st.info("Go to the 'Audio Suite' page to generate the audio.")


