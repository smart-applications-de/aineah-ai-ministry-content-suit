__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import io
from crewai import Agent, Task, Crew, Process, LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileReadTool
from datetime import datetime
from docx import Document
from io import BytesIO
import google.generativeai as genai
import time
import asyncio
import wave
import docx
import base64
#from google import gen
from PIL import Image
import pypdf
import markdown2
from google.api_core import exceptions
# --- App Configuration ---
st.set_page_config(
    page_title="AI Ministry & Content Suite",
    page_icon="✨",
    layout="wide"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    /* Button styling */
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #4CAF50;
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: white; 
        color: black;
        border: 1px solid #4CAF50;
    }
    /* Header styling */
    h1, h2, h3 {
        color: #2E4053;
    }
</style>
""", unsafe_allow_html=True)

# --- SHARED COMPONENTS & CONFIGURATION ---

# Using session state to hold credentials and other shared data
if 'gemini_key' not in st.session_state:
    st.session_state['gemini_key'] = ''
if 'serper_key' not in st.session_state:
    st.session_state['serper_key'] = ''

st.sidebar.title("🔐 Central Configuration")
st.sidebar.markdown("Enter your credentials here. They will be used across all pages.")

# Input for Gemini API Key (for all generative tasks)
gemini_key_input = st.sidebar.text_input(
    "Enter Your Google Gemini API Key",
    type="password",
    help="Required for all AI agent text, image, and video generation tasks.",
    value=st.session_state.get('gemini_key', '')
)
st.session_state['gemini_key'] = gemini_key_input

# Input for Serper API Key (for web search tools)
serper_key_input = st.sidebar.text_input(
    "Enter Your Serper.dev API Key",
    type="password",
    help="Required for agents that need web search capabilities.",
    value=st.session_state.get('serper_key', '')
)
st.session_state['serper_key'] = serper_key_input

st.sidebar.markdown("---")
st.sidebar.info(
    "For some services like VEO or Lyria, you may need to authenticate your environment via terminal: `gcloud auth application-default login`")

# --- HELPER FUNCTIONS ---

# Define the list of languages for the entire app
LANGUAGES = ("English", "German", "French", "Swahili", "Italian", "Spanish", "Portuguese")


@st.cache_data
def get_available_models(api_key, task="generateContent"):
    """Fetches and caches the list of available Gemini models for a specific task."""
    if not api_key:
        return []
    try:
        genai.configure(api_key=api_key)
        models = [
            m.name for m in genai.list_models()
            if task in m.supported_generation_methods
        ]

        return sorted([name.replace("models", "gemini") for name in models])
    except Exception as e:
        st.sidebar.error(f"Error fetching models: Invalid API Key or network issue.")
        return []

@st.cache_data
def get_available_modelsAudio(api_key):
    """Fetches and caches the list of available Gemini models for a specific task."""
    if not api_key:
        return []
    try:
        genai.configure(api_key=api_key)
        models = [
            m.name for m in genai.list_models()
            if  "tt"in str(m.name).lower()
        ]

        return sorted([name.replace("models/", "") for name in models])
    except Exception as e:
        st.sidebar.error(f"Error fetching models: Invalid API Key or network issue.")
        return []
@st.cache_data
def get_available_modelsImage(api_key):
    """Fetches and caches the list of available Gemini models for a specific task."""
    if not api_key:
        return []
    try:
        genai.configure(api_key=api_key)
        models = [
            m.name for m in genai.list_models()
            if  "image"in str(m.name).lower()
        ]

        return sorted([name.replace("models/", "") for name in models])
    except Exception as e:
        st.sidebar.error(f"Error fetching models: Invalid API Key or network issue.")
        return []
@st.cache_data
def get_available_modelsVeo(api_key):
    """Fetches and caches the list of available Gemini models for a specific task."""
    if not api_key:
        return []
    try:
        genai.configure(api_key=api_key)
        models = [
            m.name for m in genai.list_models()
            if  "veo"in str(m.name).lower()
        ]

        return sorted([name.replace("models/", "") for name in models])
    except Exception as e:
        st.sidebar.error(f"Error fetching models: Invalid API Key or network issue.")
        return []
@st.cache_data
def get_available_modelstranscript(api_key):
    """Fetches and caches the list of available Gemini models for a specific task."""
    if not api_key:
        return []
    try:
        genai.configure(api_key=api_key)
        models = [
            m.name for m in genai.list_models()
            if  "2.5" in str(m.name).lower() or "flash"  in str(m.name).lower() or "3.0" in str(m.name).lower()
        ]

        return sorted([name.replace("models/", "") for name in models])
    except Exception as e:
        st.sidebar.error(f"Error fetching models: Invalid API Key or network issue.")
        return []
def markdown_to_docx(md_content):
    """Converts markdown content to a DOCX file in memory."""
    doc = Document()
    for line in md_content.split('\n'):
        if line.startswith('### '):
            doc.add_heading(line.replace('### ', ''), level=3)
        elif line.startswith('## '):
            doc.add_heading(line.replace('## ', ''), level=2)
        elif line.startswith('# '):
            doc.add_heading(line.replace('# ', ''), level=1)
        elif line.strip() == '---':
            doc.add_page_break()
        else:
            doc.add_paragraph(line)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()
# def pil_to_bytes(image: Image.Image):
#     """Converts a PIL Image object to a byte stream for downloading."""
#     byte_arr = io.BytesIO()
#     image.save()
#     return byte_arr.getvalue()

def render_download_buttons(content, filename_base):
    """Renders a consistent set of download buttons for text-based content."""
    st.header("Export Your Content")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button("⬇️ DOCX", markdown_to_docx(content), f"{filename_base}.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    with col2:
        st.download_button("⬇️ Markdown", content.encode('utf-8'), f"{filename_base}.md", "text/markdown")
    with col3:
        st.download_button("⬇️ HTML", content.encode('utf-8'), f"{filename_base}.html", "text/html")
    with col4:
        st.download_button("⬇️ Text", content.encode('utf-8'), f"{filename_base}.txt", "text/plain")


# --- PAGE 1: SERMON GENERATOR ---

def render_sermon_page():
    st.title("📖 AI Sermon Generator Crew")
    st.markdown("This application uses a team of AI agents to help you create a detailed, biblically-sound sermon.")
    available_models = get_available_models(st.session_state.get('gemini_key'), task="generateContent")
    sermon_topic = st.text_input("Enter the Sermon Topic:", "The Power of Forgiveness")

    if available_models:
        default_model = "gemini-1.5-pro-latest"
        selected_model = st.selectbox("Choose a Gemini Model:", available_models, index=available_models.index(
            default_model) if default_model in available_models else 0)
    else:
        st.warning("Please enter a valid Gemini API Key in the sidebar to load available models.")
        selected_model = None

    target_language = st.selectbox("Choose Sermon Language:", LANGUAGES)

    if st.button("✨ Generate Sermon"):
        if not st.session_state.get('gemini_key'):
            st.error("❌ Please enter your Gemini API Key in the sidebar to proceed.")
        elif not selected_model:
            st.error("❌ Cannot generate. Please provide a valid API key to load models.")
        elif not sermon_topic:
            st.error("❌ Please provide a Sermon Topic to proceed.")
        else:
            generate_sermon(st.session_state['gemini_key'], sermon_topic, target_language, selected_model)

    if "sermon_content" in st.session_state:
        st.markdown("---")
        st.subheader("Your Generated Sermon")
        st.markdown(st.session_state["sermon_content"])
        st.markdown("---")
        # if st.button("🎧 Listen to this Sermon"):
        #     st.session_state['text_for_audio'] = st.session_state["sermon_content"]
        #     st.info("Go to the 'Audio Suite' page to generate the audio.")
        render_download_buttons(st.session_state["sermon_content"].raw, "sermon")


def generate_sermon(api_key, topic, language, model_name):
    """Initializes and runs the sermon generation crew."""
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        st.info("Lets go")
    except Exception as e:
        st.error(f"Error initializing language model: {e}")
        return

    with st.spinner(f"The AI Sermon Crew is at work using {model_name}..."):
        theologian = Agent(role='Pentecostal Theologian',
                           goal=f'Create a biblically sound outline for a 10-page sermon on "{topic}".',
                           backstory="Experienced theologian skilled in structuring compelling sermons.",
                           llm=llm,
                           verbose=True)
        researcher = Agent(role='Bible Scripture Researcher',
                           goal=f'Find relevant Bible verses for a sermon on "{topic}".',
                           backstory="Meticulous Bible scholar dedicated to scriptural accuracy.", llm=llm,
                           verbose=True)
        historian = Agent(role='Biblical Historian',
                          goal=f'Provide historical and cultural context for scriptures related to "{topic}".',
                          backstory="Expert in ancient history, bringing scriptures to life.", llm=llm, verbose=True)
        curator = Agent(role='Christian Quote Curator',
                        goal=f'Find powerful quotes from famous Christian figures about "{topic}".',
                        backstory="Well-read historian of Christian thought.", llm=llm, verbose=True)
        writer = Agent(role='Gifted Pentecostal Preacher',
                       goal=f'Write a complete, engaging 10-page sermon on "{topic}" in English, which will be the source for translation.',
                       backstory="Seasoned pastor known for powerful storytelling.", llm=llm, verbose=True)
        translator = Agent(role='Expert Theological Translator',
                           goal=f'Translate the final sermon accurately into {language}.',
                           backstory=f"Professional translator specializing in theological texts, native in {language}.",
                           llm=llm, verbose=True)

        task_outline = Task(description=f'Create a comprehensive outline for a 10-page sermon on "{topic}".',
                            agent=theologian, expected_output="A detailed sermon outline.")
        task_research = Task(description='Find relevant Bible verses for each point in the sermon outline.',
                             agent=researcher, context=[task_outline],
                             expected_output="A list of scriptures organized by outline point.")
        task_context = Task(description='Provide historical context for the key scriptures and themes.',
                            agent=historian, context=[task_outline, task_research],
                            expected_output="A summary of historical context.")
        task_quotes = Task(description=f'Find 3-5 powerful quotes about "{topic}".', agent=curator,
                           context=[task_outline], expected_output="A list of relevant quotes.")
        task_write = Task(
            description='Write a complete 10-page sermon in English using the outline, scriptures, context, and quotes.',
            agent=writer, context=[task_outline, task_research, task_context, task_quotes],
            expected_output="A complete 10-page sermon in English.")
        task_translate = Task(
            description=f'Translate the final sermon into {language}. Ensure the translation is natural and culturally appropriate for a {language}-speaking audience.',
            agent=translator, context=[task_write], expected_output=f"The full sermon text translated into {language}.")

        sermon_crew = Crew(agents=[theologian, researcher, historian, curator, writer, translator],
                           tasks=[task_outline, task_research,
                                  task_context, task_quotes, task_write, task_translate],
                           process=Process.sequential, Verbose=True)
        try:
            final_sermon = sermon_crew.kickoff()
            st.session_state['sermon_content'] = final_sermon
        except Exception as e:
            st.error(f"An error occurred during sermon generation: {e}")


# --- PAGE 2: FLYER GENERATOR ---

def render_flyer_page():
    st.title("🚀 AI Flyer Production Studio")
    st.markdown(
        "From idea to share-ready asset in minutes. Your expert AI crew will design a concept, generate flyers, and write the social media copy.")

    available_text_models = get_available_models(st.session_state.get('gemini_key'), task="generateContent")
    available_image_models =get_available_modelsImage(st.session_state.get('gemini_key'))
        #"imagen-3.0-generate-002"
    #st.markdown(get_available_modelsImage(st.session_state.get('gemini_key')))

    st.header("Step 1: Describe Your Flyer")
    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input("**Topic or Theme:**", placeholder="e.g., Street Evangelism, Youth Camp")
        flyer_type = st.selectbox("**Flyer Type:**",
                                  ("Social Media Post (Square)", "Poster (Portrait)", "Banner (Landscape)"))
    with col2:
        text_element = st.text_input("**Key Text (Verse, Slogan, etc.):**",
                                     placeholder='e.g., "John 3:16", "Summer Blast 2025"')
        target_language = st.selectbox("Choose Social Media Copy Language:", LANGUAGES)

    if available_text_models:
        selected_text_model = st.selectbox("Choose a Gemini Model for Concept:", available_text_models,
                                           index=available_text_models.index(
                                               "gemini-1.5-pro-latest") if "gemini-1.5-pro-latest" in available_text_models else 0)
    else:
        st.warning("Please enter a valid Gemini API Key to load text models.")
        selected_text_model = None

    if available_image_models:
        default_image_model = "imagen-3.0-generate-002"
        selected_image_model = st.selectbox("Choose an Imagen Model for Generation:", available_image_models,
                                            index=available_image_models.index(
                                                default_image_model) if default_image_model in available_image_models else 0)
    else:
        st.warning("Please enter a valid Gemini API Key to load image models.")
        selected_image_model = None

    st.header("Step 2: Produce & Distribute")
    if st.button("🚀 Generate Flyer and Social Copy"):
        if not all([st.session_state.get(k) for k in ['gemini_key', 'serper_key']]):
            st.error("❌ Please enter your Gemini and Serper API Keys in the sidebar.")
        elif not selected_text_model:
            st.error("❌ Cannot generate. Please provide a valid API key to load all required models.")
        elif not topic or not text_element:
            st.error("❌ Please provide both a Topic and Key Text to proceed.")
        else:
            create_flyer_and_copy(api_key=st.session_state['gemini_key'], topic=topic, text_element=text_element,
                                  flyer_type=flyer_type, language=target_language, text_model_name=selected_text_model,
                                  image_model_name=selected_image_model)


@st.cache_data
def generate_images_with_langchain(api_key: str, prompt: str, model_name: str):
    """Generates images using the langchain_google_genai library."""
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        from google import genai as gen
        from google.genai import types
        from PIL import Image
        client = gen.Client(api_key=os.environ["GOOGLE_API_KEY"])
        #'imagen-4.0-generate-preview-06-06'
        response = client.models.generate_images(
            #imagen-3.0-generate-002
            model=model_name,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
            )
        )
        st.success("Image generated successfully.")
        image_bytes_list = []
        for i, generated_image in  enumerate(response.generated_images):
            #img_byte_arr = BytesIO()
            generated_image.image.save(f"image{i}.png")
            #initial_image = img_byte_arr
            #st.image(f"image{i}.png", caption=f"Generated Initial Image {i}")
            image_bytes_list.append(f"image{i}.png")
        return image_bytes_list

    except Exception as e:
        st.error(f"An error occurred during image generation: {e}")
        return None


# --- NEW: Image Generation Function ---
def generate_images_from_prompt(api_key:str, prompt:str, num_images:int, aspect_ratio:str, person_gen:str,model_name:str):
    """
    Calls the Google GenAI API to generate images based on user inputs.
    Returns a list of PIL Image objects.
    """
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        from google import genai as gen
        from google.genai import types
        from PIL import Image
        client = gen.Client(api_key=os.environ["GOOGLE_API_KEY"])

        # This client call is based on the new standalone Python SDK.
        # Ensure the model name is current.
        response = client.models.generate_images(
            model=model_name,  # Using a current model name as of late 2024/early 2025
            prompt=prompt,
            # The config maps directly to the user's selections
            config=types.GenerateImagesConfig(
                number_of_images=num_images,
                aspect_ratio=aspect_ratio,
                personGeneration=person_gen

            )
        )
        image_bytes_list = []
        for i, generated_image in  enumerate(response.generated_images):
            #img_byte_arr = BytesIO()
            generated_image.image.save(f"image{i}.png")
            #initial_image = img_byte_arr
            #st.image(f"image{i}.png", caption=f"Generated Initial Image {i}")
            image_bytes_list.append(f"image{i}.png")
        # The new SDK returns a result object with a list of GeneratedImage objects
        return image_bytes_list

    except exceptions.InvalidArgument as e:
        # Catches errors like a prompt being rejected by the safety filter.
        st.error(
            f"⚠️ Your request could not be processed. The prompt may have been rejected by the safety filter. Please try a different prompt. (Error: {e})")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return []

def create_flyer_and_copy(api_key, topic, text_element, flyer_type, language, text_model_name, image_model_name):
    """Initializes and runs the flyer design crew."""
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        llm = LLM(model=text_model_name, temperature=0.8, api_key=os.environ["GOOGLE_API_KEY"])
        search_tool = SerperDevTool(api_key=st.session_state['serper_key'])
    except Exception as e:
        st.error(f"Error initializing language model: {e}")
        return

    with st.spinner("Your AI Design Studio is developing the concept..."):
        brief_specialist = Agent(role='Creative Brief Specialist', goal='Develop a clear creative brief for a flyer.',
                                 backstory="Marketing expert skilled in distilling ideas into actionable briefs.",
                                 llm=llm, tools=[search_tool], verbose=True)
        concept_developer = Agent(role='Visual Concept Developer',
                                  goal='Brainstorm strong visual concepts based on a creative brief.',
                                  backstory="Seasoned Art Director with a modern aesthetic sense.", llm=llm,
                                  tools=[search_tool], verbose=True)
        prompt_crafter = Agent(role='Google Imagen Prompt Engineer',
                               goal='Craft a detailed, effective image generation prompt for Google\'s Imagen model.',
                               backstory="Technical artist who knows how to translate concepts into effective AI prompts.",
                               llm=llm, verbose=True)
        copywriter = Agent(role='Multilingual Social Media Copywriter',
                           goal=f'Write a short, engaging social media post in {language} to accompany the flyer image.',
                           backstory="Viral marketing specialist who crafts words that stop the scroll in multiple languages.",
                           llm=llm, tools=[search_tool], verbose=True)

        briefing = Task(
            description=f"Analyze the request (Topic: '{topic}', Text: '{text_element}', Type: '{flyer_type}') to create a Creative Brief defining Target Audience, Desired Emotion, and Core Message.",
            agent=brief_specialist, expected_output="A concise creative brief.")
        visualizing = Task(
            description="Based on the brief, develop a full visual concept, including a core metaphor, color palette, composition, and artistic style.",
            agent=concept_developer, context=[briefing], expected_output="A detailed visual concept document.")
        crafting_prompt = Task(
            description="Synthesize the brief and concept into a single, masterful image generation prompt for Google Imagen. The prompt must be a detailed paragraph. **Do NOT include the text/slogan in the prompt itself.**",
            agent=prompt_crafter, context=[briefing, visualizing],
            expected_output="A single paragraph: the final image prompt.")
        crafting_copy = Task(
            description=f"Based on the brief and concept, write a compelling social media post in {language}. Incorporate the key text '{text_element}', use 3-5 relevant hashtags, and end with a call-to-action.",
            agent=copywriter, context=[briefing, visualizing],
            expected_output=f"A complete social media post with text and hashtags, written in {language}.")

        flyer_crew = Crew(agents=[brief_specialist, concept_developer, prompt_crafter, copywriter],
                          tasks=[briefing, visualizing, crafting_prompt, crafting_copy], process=Process.sequential,
                          Verbose=True)

        crew_result = None
        try:
            crew_result = flyer_crew.kickoff()
            image_prompt = crew_result.tasks_output[2].raw
            social_copy = crew_result.tasks_output[3].raw
            st.success("Concept approved! Prompt and copy are ready.")
            st.subheader("🎨 Generated Image Prompt")
            st.code(image_prompt, language="text")
            st.session_state['flyer_social_copy'] = social_copy
        except Exception as e:
            st.error(f"An error occurred during AI crew execution: {e}")
            return

    st.markdown("---")

    st.subheader("🎨 Choose Your Image Prompt")

    # Let user choose between the generated prompt or a custom one
    prompt_choice = st.radio(
        "Select the prompt source for flyer generation:",
        ("Use the AI-Generated Prompt", "Enter my own custom prompt"),
        key="prompt_choice"
    )

    if crew_result and prompt_choice == "Use the AI-Generated Prompt":
        with st.spinner("Sending prompt to Google Imagen for final rendering..."):
            image_bytes_list = generate_images_with_langchain(api_key=api_key, prompt=image_prompt,
                                                              model_name=image_model_name)

        if image_bytes_list:
            st.success("Rendering complete! Here are your flyer options:")
            st.subheader("✅ Your Final Flyers")
            st.session_state['generated_flyers'] = image_bytes_list

            cols = st.columns(len(image_bytes_list))
            for i, image_bytes in enumerate(image_bytes_list):
                with cols[i]:
                    st.image(image_bytes, caption=f"Option {i + 1}")
                    st.download_button(label=f"Download Option {i + 1}", data=image_bytes,
                                       file_name=f"flyer_option_{i + 1}.png", mime="image/png")

            st.subheader(f"✍️ Your Social Media Caption in {language}")
            st.text_area("", social_copy, height=150)
            render_download_buttons(social_copy, "social_media_copy")
    if prompt_choice == "Enter my own custom prompt" or not  crew_result:
        final_prompt = st.text_area(
            "Enter your custom prompt here:",
            # Pre-fill with generated prompt as a starting point for editing
            #value=st.session_state.get('custom_prompt', st.session_state['generated_prompt']),
            height=150,
            key='custom_prompt'  # Use a key to save the input
        )
        if final_prompt:
            with st.spinner("Sending prompt to Google Imagen for final rendering..."):
                image_bytes_list = generate_images_with_langchain(api_key=api_key, prompt=final_prompt,
                                                                  model_name=image_model_name)

            if image_bytes_list:
                st.success("Rendering complete! Here are your flyer options:")
                st.subheader("✅ Your Final Flyers")
                st.session_state['generated_flyers'] = image_bytes_list

                cols = st.columns(len(image_bytes_list))
                for i, image_bytes in enumerate(image_bytes_list):
                    with cols[i]:
                        st.image(image_bytes, caption=f"Option {i + 1}")
                        st.download_button(label=f"Download Option {i + 1}", data=image_bytes,
                                           file_name=f"flyer_option_{i + 1}.png", mime="image/png")



# --- PAGE 3: WORSHIP SONG STUDIO ---

def render_music_page():
    st.title("🎶 AI Worship Song Studio")
    st.markdown(
        "Partner with our AI Worship Team to compose a song concept, generate a production-ready prompt for **Google's Lyria AI**, and then generate the audio.")

    available_models = get_available_models(st.session_state.get('gemini_key'))

    st.header("Step 1: Share Your Inspiration")
    col1, col2 = st.columns(2)
    with col1:
        genre = st.selectbox("**Select the Musical Genre:**", (
        "Worship (Hillsong/Bethel style)", "Praise (Elevation/Upbeat style)", "African Gospel Praise"))
        topic = st.text_input("**Core Topic or Theme:**", placeholder="e.g., God's Grace, Faithfulness, Salvation")
    with col2:
        verses = st.text_area("**Enter Bible Verses or Inspirational Text:**", placeholder="e.g., John 3:16, Psalm 23",
                              height=150)
        target_language = st.selectbox("Choose Lyrics Language:", LANGUAGES)

    if available_models:
        selected_model = st.selectbox("Choose a Gemini Model for Songwriting:", available_models,
                                      index=available_models.index(
                                          "gemini-1.5-pro-latest") if "gemini-1.5-pro-latest" in available_models else 0)
    else:
        st.warning("Please enter a valid Gemini API Key in the sidebar to load available models.")
        selected_model = None

    st.header("Step 2: Compose the Song")
    if st.button("Compose & Generate Lyria Prompt"):
        if not all([st.session_state.get(k) for k in ['gemini_key', 'serper_key']]):
            st.error("❌ Please enter your Gemini and Serper API keys in the sidebar.")
        elif not selected_model:
            st.error("❌ Cannot generate. Please provide a valid API key to load models.")
        elif not topic and not verses:
            st.error("🚨 Please provide a Topic or some Bible Verses to inspire the song.")
        else:
            create_and_run_music_crew(api_key=st.session_state['gemini_key'], genre=genre, verses=verses, topic=topic,
                                      language=target_language, model_name=selected_model)

    if "lyria_prompt" in st.session_state:
        st.subheader("✅ Your Final Lyria Prompt")
        st.code(st.session_state["lyria_prompt"], language="text")
        render_download_buttons(st.session_state["lyria_prompt"].raw, "lyria_prompt")

        st.header("Step 3: Generate Your Audio")
        if st.button("🎵 Generate 30-Second Audio Track"):
            generate_and_display_music(st.session_state['gemini_key'],st.session_state["lyria_prompt"])


async def generate_music_async(api_key, prompt, audio_chunks):
    """Async function to handle the real-time music generation."""
    try:
        from google import genai as gen
        from google.genai import types
        os.environ["GOOGLE_API_KEY"]=api_key
        #client = gen.Client(api_key=os.environ["GOOGLE_API_KEY"])
        client = gen.Client(api_key=os.environ["GOOGLE_API_KEY"],http_options={'api_version': 'v1alpha'})

        async def receive_audio(session):
            while True:
                async for message in session.receive():
                    if message.server_content and message.server_content.audio_chunks:
                        audio_chunks.append(message.server_content.audio_chunks[0].data)

        async with client.aio.live.music.connect(model='models/lyria-realtime-exp') as session:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(receive_audio(session))
                await session.set_weighted_prompts(prompts=[gen.types.WeightedPrompt(text=prompt, weight=1.0)])
                await session.set_music_generation_config(
                    config=gen.types.LiveMusicGenerationConfig(bpm=120, temperature=1.0))
                await session.play()
                await asyncio.sleep(30)  # Generate for 30 seconds
                await session.stop()
    except Exception as e:
        st.error(f"Error during async music generation: {e}")


def pcm_to_wav(pcm_data, channels, sample_width, sample_rate):
    """Converts raw PCM data to a WAV file in memory."""
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    buffer.seek(0)
    return buffer.getvalue()


def generate_and_display_music(api_key, prompt):
    """Main function to orchestrate async generation and display in Streamlit."""
    with st.spinner("Connecting to Lyria and generating your audio track... This will take about 45 seconds."):
        audio_chunks = []
        try:
            asyncio.run(generate_music_async(api_key,prompt, audio_chunks))

            if audio_chunks:
                st.success("Audio track generated successfully!")
                raw_audio_data = b''.join(audio_chunks)
                wav_bytes = pcm_to_wav(raw_audio_data, channels=2, sample_width=2, sample_rate=24000)

                st.audio(wav_bytes, format='audio/wav')
                st.download_button(
                    label="⬇️ Download WAV file",
                    data=wav_bytes,
                    file_name="generated_worship_song.wav",
                    mime="audio/wav"
                )
            else:
                st.error("No audio data was generated. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info(
                "This feature is experimental. Please ensure your Google Cloud account has access to the Lyria API.")


def create_and_run_music_crew(api_key, genre, verses, topic, language, model_name):
    """Initializes and runs the music creation crew."""
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        search_tool = SerperDevTool(api_key=st.session_state['serper_key'])
    except Exception as e:
        st.error(f"Error initializing language model: {e}")
        return

    with st.spinner("Your AI Worship Team is gathering... This may take a few minutes."):
        lyricist = Agent(role='Theological Lyricist',
                         goal='Extract core theological truths, emotions, and imagery from user input for a worship song.',
                         backstory="Bridges deep biblical study and heartfelt lyrical expression.", llm=llm,
                         tools=[search_tool], verbose=True)
        songwriter = Agent(role='Worship Songwriter',
                           goal=f'Craft compelling, structured song lyrics in {language} based on the theological concepts.',
                           backstory="Seasoned songwriter skilled in creating poetic and accessible lyrics for worship.",
                           llm=llm, verbose=True)
        arranger = Agent(role='Music Arranger',
                         goal='Define the musical arrangement and atmosphere for a song based on the chosen genre.',
                         backstory="Producer who creates sonic landscapes for any genre, from Bethel to African Gospel.",
                         llm=llm, verbose=True)
        prompt_technician = Agent(role='Lyria Prompt Technician',
                                  goal=f'Synthesize the {language} lyrics and arrangement into a comprehensive prompt for Google\'s Lyria music AI.',
                                  backstory="Specialist in generative AI for music, translating creative vision into precise AI instructions.",
                                  llm=llm, verbose=True)

        lyrical_concept_task = Task(
            description=f"Analyze provided verses ('{verses}') and topic ('{topic}') to create a Lyrical Concept Brief.",
            agent=lyricist, expected_output="A concise Lyrical Concept Brief.")
        arrangement_task = Task(
            description=f"Create a detailed Musical Arrangement Guide for a '{genre}' song about '{topic}'.",
            agent=arranger, expected_output="A detailed Musical Arrangement Guide.")
        song_writing_task = Task(
            description=f"Using the Lyrical Concept Brief, write a complete worship song in {language} with a clear structure (Verse, Chorus, Bridge).",
            agent=songwriter, context=[lyrical_concept_task],
            expected_output="A complete song with clearly labeled sections.")
        prompt_generation_task = Task(
            description=f"Combine the final lyrics (in {language}) and the Musical Arrangement Guide into one single,"
                        f" detailed prompt for Google's Lyria model.",
            agent=prompt_technician, context=[song_writing_task, arrangement_task],
            expected_output="A single, comprehensive text prompt for Lyria.")

        music_crew = Crew(agents=[lyricist, songwriter, arranger, prompt_technician],
                          tasks=[lyrical_concept_task, arrangement_task, song_writing_task, prompt_generation_task],
                          process=Process.sequential, Verbose=True)

        try:
            final_prompt = music_crew.kickoff()
            st.session_state["lyria_prompt"] = final_prompt
        except Exception as e:
            st.error(f"An error occurred during composition: {e}")


# --- PAGE 4: BOOK WRITING STUDIO ---

def render_book_page():
    st.title("📚 AI Book Writing Studio")
    st.markdown(
        "This is your personal AI writer's room. Provide a topic, a detailed prompt, and select your desired language.")

    available_models = get_available_models(st.session_state.get('gemini_key'))

    st.header("Step 1: Define Your Book's Vision")
    language = st.selectbox("**Select the language for your book:**", LANGUAGES)
    topic = st.text_input("**Enter the core topic or theme of your book:**",
                          placeholder="e.g., The history of quantum computing")
    user_prompt = st.text_area("**Provide a detailed description of your book idea:**", height=200,
                               placeholder="Describe your book's focus, target audience, and key themes...")

    if available_models:
        selected_model = st.selectbox("Choose a Gemini Model for Book Writing:", available_models,
                                      index=available_models.index(
                                          "gemini-1.5-pro-latest") if "gemini-1.5-pro-latest" in available_models else 0)
    else:
        st.warning("Please enter a valid Gemini API Key in the sidebar to load available models.")
        selected_model = None

    st.header("Step 2: Assemble Your Crew and Start Writing")
    if st.button(f"Start Writing My Book in {language}"):
        if not all([st.session_state.get(k) for k in ['gemini_key', 'serper_key']]):
            st.error("❌ Please enter your Gemini and Serper API keys in the sidebar.")
        elif not selected_model:
            st.error("❌ Cannot generate. Please provide a valid API key to load models.")
        elif not topic or not user_prompt:
            st.error("🚨 Please provide a Topic and a detailed description to proceed.")
        else:
            create_and_run_book_crew(
                api_key=st.session_state['gemini_key'],
                topic=topic,
                user_prompt=user_prompt,
                language=language,
                model_name=selected_model
            )

    if "book_content" in st.session_state:
        st.markdown("---")
        st.subheader("Your Generated Book Chapters")
        st.markdown(st.session_state["book_content"])
        st.markdown("---")
        render_download_buttons(st.session_state["book_content"], "book_chapters")


def create_and_run_book_crew(api_key, topic, user_prompt, language, model_name):
    """Initializes and runs the book writing crew."""
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        search_tool = SerperDevTool(api_key=st.session_state['serper_key'])
        scrape_tool = ScrapeWebsiteTool()
        file_tool = FileReadTool()
    except Exception as e:
        st.error(f"Error initializing language model or tools: {e}")
        return

    with st.spinner(
            f"Your AI Book Writing Crew is assembling to write in {language}... This may take several minutes."):
        architect = Agent(role='Chief Outline Architect',
                          goal=f'Create a comprehensive, chapter-by-chapter outline for a ~10-page book on "{topic}" in {language}.',
                          backstory='A seasoned developmental editor and bestselling author, you excel at structuring complex ideas into engaging book formats.',
                          llm=llm, tools=[search_tool, scrape_tool], verbose=True)
        researcher = Agent(role='Research Specialist',
                           goal='Gather, verify, and compile detailed information for each point in the book outline.',
                           backstory='A meticulous multilingual researcher with a Ph.D., you can find a needle in a digital haystack.',
                           llm=llm, tools=[search_tool, scrape_tool, file_tool], verbose=True)
        writer = Agent(role='Narrative Crafter',
                       goal=f'Write engaging, well-structured chapters in {language}, based on the provided outline and research.',
                       backstory='A master storyteller and ghostwriter, you bring ideas to life with native-level fluency in several languages.',
                       llm=llm, tools=[search_tool], verbose=True)
        editor = Agent(role='Senior Editor',
                       goal=f'Review, edit, and polish the drafted chapters to ensure stylistic consistency, grammatical correctness, and overall narrative coherence in {language}.',
                       backstory='With a red pen sharpened by years at top publishing houses, you are the final gatekeeper of quality.',
                       llm=llm, verbose=True)
        output_filename_outline = f'book_outline_output_{language.lower()}.md'

        outline_task = Task(
            description=f"Analyze the user's book idea (Topic: '{topic}', Prompt: '{user_prompt}') and develop a comprehensive chapter-by-chapter outline. The entire output MUST be in {language}.",
            agent=architect, expected_output=f"A detailed book outline, written entirely in {language}.",output_file= output_filename_outline)
        research_task = Task(
            description=f"For each chapter in the outline, conduct thorough research. Compile all findings into a structured document, tailored for a writer working in {language}.",
            agent=researcher, context=[outline_task], expected_output="A well-organized research document.")
        writing_task = Task(
            description=f"Using the outline and research, write the full content for the first three chapters of the book. The entire text must be in {language}.",
            agent=writer, context=[research_task],
            expected_output=f"The complete text for the first three chapters, written in {language}.")

        output_filename = f'book_final_output_{language.lower()}.md'
        editing_task = Task(
            description=f"Perform a comprehensive edit of the drafted chapters. Your review and all final edits must be in {language}, ensuring the text sounds like it was written by a native speaker.",
            agent=editor, context=[writing_task],
            expected_output=f"The final, polished text for the written chapters, ready for publication in {language}.",
            output_file=output_filename)

        book_crew = Crew(agents=[architect, researcher, writer, editor],
                         tasks=[outline_task, research_task, writing_task, editing_task], process=Process.sequential,
                         Verbose=True)

        try:
            result = book_crew.kickoff()
            with open(output_filename, 'r', encoding='utf-8') as file:
                final_output = file.read()
            st.session_state['book_content'] = final_output
        except Exception as e:
            st.error(f"An error occurred while running the AI crew: {e}")


# --- PAGE 5: BIBLE STUDY GENERATOR ---

def render_bible_study_page():
    st.title("🌍 Multilingual AI Bible Study Generator")
    st.markdown("Powered by a team of AI Theologians, Pastors, and Historians.")

    # This data is extensive, so it's defined within the page function for clarity.
    ENGLISH_BOOKS = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth", "1 Samuel",
                     "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther",
                     "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah",
                     "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum",
                     "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi", "Matthew", "Mark", "Luke", "John",
                     "Acts", "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians", "Philippians",
                     "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon",
                     "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation"]
    BIBLE_BOOKS_TRANSLATIONS = {
        "English": ENGLISH_BOOKS,
        "German": ["Genesis", "Exodus", "Levitikus", "Numeri", "Deuteronomium", "Josua", "Richter", "Ruth", "1. Samuel",
                   "2. Samuel", "1. Könige", "2. Könige", "1. Chronik", "2. Chronik", "Esra", "Nehemia", "Esther",
                   "Hiob", "Psalmen", "Sprüche", "Prediger", "Hohelied", "Jesaja", "Jeremia", "Klagelieder", "Hesekiel",
                   "Daniel", "Hosea", "Joel", "Amos", "Obadja", "Jona", "Micha", "Nahum", "Habakuk", "Zefanja",
                   "Haggai", "Sacharja", "Maleachi", "Matthäus", "Markus", "Lukas", "Johannes", "Apostelgeschichte",
                   "Römer", "1. Korinther", "2. Korinther", "Galater", "Epheser", "Philipper", "Kolosser",
                   "1. Thessalonicher", "2. Thessalonicher", "1. Timotheus", "2. Timotheus", "Titus", "Philemon",
                   "Hebräer", "Jakobus", "1. Petrus", "2. Petrus", "1. Johannes", "2. Johannes", "3. Johannes", "Judas",
                   "Offenbarung"],
        "French": ["Genèse", "Exode", "Lévitique", "Nombres", "Deutéronome", "Josué", "Juges", "Ruth", "1 Samuel",
                   "2 Samuel", "1 Rois", "2 Rois", "1 Chroniques", "2 Chroniques", "Esdras", "Néhémie", "Esther", "Job",
                   "Psaumes", "Proverbes", "Ecclésiaste", "Cantique des Cantiques", "Ésaïe", "Jérémie", "Lamentations",
                   "Ézéchiel", "Daniel", "Osée", "Joël", "Amos", "Abdias", "Jonas", "Michée", "Nahum", "Habacuc",
                   "Sophonie", "Aggée", "Zacharie", "Malachie", "Matthieu", "Marc", "Luc", "Jean", "Actes", "Romains",
                   "1 Corinthiens", "2 Corinthiens", "Galates", "Éphésiens", "Philippiens", "Colossiens",
                   "1 Thessaloniciens", "2 Thessaloniciens", "1 Timothée", "2 Timothée", "Tite", "Philémon", "Hébreux",
                   "Jacques", "1 Pierre", "2 Pierre", "1 Jean", "2 Jean", "3 Jean", "Jude", "Apocalypse"],
        "Swahili": ["Mwanzo", "Kutoka", "Walawi", "Hesabu", "Kumbukumbu la Torati", "Yoshua", "Waamuzi", "Ruthu",
                    "1 Samweli", "2 Samweli", "1 Wafalme", "2 Wafalme", "1 Mambo ya Nyakati", "2 Mambo ya Nyakati",
                    "Ezra", "Nehemia", "Esta", "Ayubu", "Zaburi", "Methali", "Mhubiri", "Wimbo Ulio Bora", "Isaya",
                    "Yeremia", "Maombolezo", "Ezekieli", "Danieli", "Hosea", "Yoeli", "Amosi", "Obadia", "Yona", "Mika",
                    "Nahumu", "Habakuki", "Sefania", "Hagai", "Zekaria", "Malaki", "Mathayo", "Marko", "Luka", "Yohana",
                    "Matendo", "Warumi", "1 Wakorintho", "2 Wakorintho", "Wagalatia", "Waefeso", "Wafilipi",
                    "Wakolosai", "1 Wathesalonike", "2 Wathesalonike", "1 Timotheo", "2 Timotheo", "Tito", "Filemoni",
                    "Waebrania", "Yakobo", "1 Petro", "2 Petro", "1 Yohana", "2 Yohana", "3 Yohana", "Yuda", "Ufunuo"],
        "Italian": ["Genesi", "Esodo", "Levitico", "Numeri", "Deuteronomio", "Giosuè", "Giudici", "Rut", "1 Samuele",
                    "2 Samuele", "1 Re", "2 Re", "1 Cronache", "2 Cronache", "Esdra", "Neemia", "Ester", "Giobbe",
                    "Salmi", "Proverbi", "Ecclesiaste", "Cantico dei Cantici", "Isaia", "Geremia", "Lamentazioni",
                    "Ezechiele", "Daniele", "Osea", "Gioele", "Amos", "Abdia", "Giona", "Michea", "Naum", "Abacuc",
                    "Sofonia", "Aggeo", "Zaccaria", "Malachia", "Matteo", "Marco", "Luca", "Giovanni", "Atti", "Romani",
                    "1 Corinzi", "2 Corinzi", "Galati", "Efesini", "Filippesi", "Colossesi", "1 Tessalonicesi",
                    "2 Tessalonicesi", "1 Timoteo", "2 Timoteo", "Tito", "Filemone", "Ebrei", "Giacomo", "1 Pietro",
                    "2 Pietro", "1 Giovanni", "2 Giovanni", "3 Giovanni", "Giuda", "Apocalisse"],
        "Spanish": ["Génesis", "Éxodo", "Levítico", "Números", "Deuteronomio", "Josué", "Jueces", "Rut", "1 Samuel",
                    "2 Samuel", "1 Reyes", "2 Reyes", "1 Crónicas", "2 Crónicas", "Esdras", "Nehemías", "Ester", "Job",
                    "Salmos", "Proverbios", "Eclesiastés", "Cantares", "Isaías", "Jeremías", "Lamentaciones",
                    "Ezequiel", "Daniel", "Oseas", "Joel", "Amós", "Abdías", "Jonás", "Miqueas", "Nahúm", "Habacuc",
                    "Sofonías", "Hageo", "Zacarías", "Malaquías", "Mateo", "Marcos", "Lucas", "Juan", "Hechos",
                    "Romanos", "1 Corintios", "2 Corintios", "Gálatas", "Efesios", "Filipenses", "Colosenses",
                    "1 Tesalonicenses", "2 Tesalonicenses", "1 Timoteo", "2 Timoteo", "Tito", "Filemón", "Hebreos",
                    "Santiago", "1 Pedro", "2 Pedro", "1 Juan", "2 Juan", "3 Juan", "Judas", "Apocalipsis"],
        "Portuguese": ["Gênesis", "Êxodo", "Levítico", "Números", "Deuteronômio", "Josué", "Juízes", "Rute", "1 Samuel",
                       "2 Samuel", "1 Reis", "2 Reis", "1 Crônicas", "2 Crônicas", "Esdras", "Neemias", "Ester", "Jó",
                       "Salmos", "Provérbios", "Eclesiastes", "Cânticos", "Isaías", "Jeremias", "Lamentações",
                       "Ezequiel", "Daniel", "Oseias", "Joel", "Amós", "Obadias", "Jonas", "Miqueias", "Naum",
                       "Habacuque", "Sofonias", "Ageu", "Zacarias", "Malaquias", "Mateus", "Marcos", "Lucas", "João",
                       "Atos", "Romanos", "1 Coríntios", "2 Coríntios", "Gálatas", "Efésios", "Filipenses",
                       "Colossenses", "1 Tessalonicenses", "2 Tessalonicenses", "1 Timóteo", "2 Timóteo", "Tito",
                       "Filemom", "Hebreus", "Tiago", "1 Pedro", "2 Pedro", "1 João", "2 João", "3 João", "Judas",
                       "Apocalipse"]
    }

    available_models = get_available_models(st.session_state.get('gemini_key'))

    st.header("1. Select Your Language and Book")
    selected_language = st.selectbox("Choose your language:", list(BIBLE_BOOKS_TRANSLATIONS.keys()), key="bible_lang")
    selected_book_translated = st.selectbox("Choose a book to study:", BIBLE_BOOKS_TRANSLATIONS[selected_language],
                                            key="bible_book")

    if available_models:
        selected_model = st.selectbox("Select Gemini Model", available_models, index=available_models.index(
            "gemini-1.5-pro-latest") if "gemini-1.5-pro-latest" in available_models else 0, key="bible_model")
    else:
        st.warning("Please enter a valid Gemini API Key in the sidebar to load available models.")
        selected_model = None

    st.header("2. Generate Your Study Guide")
    if st.button(f"Create Study Guide for {selected_book_translated}"):
        if not all([st.session_state.get(k) for k in ['gemini_key', 'serper_key']]):
            st.error("🚨 Please enter both your Gemini and Serper API keys in the sidebar to continue.")
        elif not selected_model:
            st.error("❌ Cannot generate. Please provide a valid API key to load models.")
        else:
            if "study_guide_content" in st.session_state:
                del st.session_state["study_guide_content"]

            book_index = BIBLE_BOOKS_TRANSLATIONS[selected_language].index(selected_book_translated)
            english_book_name = ENGLISH_BOOKS[book_index]

            with st.spinner(
                    f"Your AI Bible Study Team is preparing your guide for '{selected_book_translated}' in {selected_language}..."):
                create_and_run_bible_study_crew(
                    english_book_name=english_book_name,
                    language=selected_language,
                    model_name=selected_model,
                    api_key=st.session_state['gemini_key'],
                    serper_api_key=st.session_state['serper_key']
                )

    if "study_guide_content" in st.session_state:
        st.header("3. Your Custom Study Guide")
        st.markdown(st.session_state["study_guide_content"])
        if st.button("🎧 Listen to this Study Guide"):
            st.session_state['text_for_audio'] = st.session_state["study_guide_content"]
            st.info("Go to the 'Audio Suite' page to generate the audio.")
        render_download_buttons(st.session_state["study_guide_content"],
                                f"{selected_book_translated.replace(' ', '_')}_study_guide")


def create_and_run_bible_study_crew(english_book_name, language, model_name, api_key, serper_api_key):
    """Initializes and runs the bible study crew."""
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["SERPER_API_KEY"] = serper_api_key
        llm = LLM(model=model_name, temperature=0.5, api_key= os.environ["GOOGLE_API_KEY"])
        search_tool = SerperDevTool(api_key=os.environ["SERPER_API_KEY"])
    except Exception as e:
        st.error(f"Error initializing services: {e}")
        return

    historian = Agent(role='Biblical Historian & Archaeologist',
                      goal=f'Provide a comprehensive historical, cultural, and literary background for {english_book_name}, in {language}.',
                      backstory="With a PhD from Jerusalem University, you provide the crucial context that makes the biblical text come alive.",
                      llm=llm, tools=[search_tool], verbose=True)
    theologian = Agent(role='Exegetical Theologian',
                       goal=f'Analyze the text of {english_book_name} to uncover its main theological themes, key verses, and structure, presenting findings in {language}.',
                       backstory="As a systematic theologian, you are an expert at exegesis—drawing out the intended meaning of the text.",
                       llm=llm, tools=[search_tool], verbose=True)
    pastor = Agent(role='Pastoral Guide & Counselor',
                   goal=f'Create practical, thought-provoking application questions and prayer points based on the themes of {english_book_name}, written in {language}.',
                   backstory="A seasoned pastor skilled in crafting questions that bridge the gap between ancient text and modern life.",
                   llm=llm, verbose=True)
    editor = Agent(role='Senior Editor for Christian Publishing',
                   goal=f'Compile all sections into a single, cohesive, and beautifully formatted Bible study guide in {language}.',
                   backstory="You work for an international Christian publishing house, ensuring every manuscript is professional and theologically sound.",
                   llm=llm, verbose=True)

    task1 = Task(
        description=f"Create the 'Historical Background' section for a study guide on **{english_book_name}**. Your output MUST be in {language}.",
        agent=historian,
        expected_output=f"A Markdown section on the historical background of {english_book_name}, written entirely in {language}.")
    task2 = Task(
        description=f"Create the 'Theological Themes & Key Verses' section for **{english_book_name}**. Your output MUST be in {language}. Use a well-known {language} Bible translation for quotes.",
        agent=theologian,
        expected_output=f"A detailed Markdown section on theological themes of {english_book_name}, written entirely in {language}.")
    task3 = Task(
        description=f"Create the 'Practical Application & Reflection' section for **{english_book_name}**. Your output MUST be in {language}.",
        agent=pastor,
        expected_output=f"An encouraging Markdown section with discussion questions and prayer points for {english_book_name}, written entirely in {language}.")

    output_filename = f'final_study_guide_{language.lower()}.md'
    task4 = Task(
        description=f"Compile all sections into a single study guide. The main title should be the {language} translation for 'A Study Guide to the Book of {english_book_name}'.",
        agent=editor, context=[task1, task2, task3],
        expected_output=f"A complete, well-formatted Markdown document in {language}.", output_file=output_filename)

    crew = Crew(agents=[historian, theologian, pastor, editor], tasks=[task1, task2, task3, task4],
                process=Process.sequential, Verbose=True)

    try:
        result = crew.kickoff()
        with open(output_filename, 'r', encoding='utf-8') as file:
            st.session_state["study_guide_content"] = file.read()
    except Exception as e:
        st.error(f"An error occurred: {e}")


# --- PAGE 6: NEWSROOM HQ ---

def render_news_page():
    st.title("📰 AI Newsroom Headquarters")
    st.markdown(
        "Welcome, Editor-in-Chief! Commission a complete, up-to-the-minute digital newspaper from your AI journalist crew.")

    available_models = get_available_models(st.session_state.get('gemini_key'))

    st.header("Step 1: Define Your Newspaper's Focus")
    scope_options = ["Global", "National", "Local"]
    scope = st.selectbox("**Select Newspaper Scope:**", scope_options)

    location = ""
    if scope == "Local":
        location = st.text_input("Enter City:", "Bungoma")  # Changed to free text input
    elif scope == "National":
        location = st.text_input("Enter Country:", "Kenya")

    st.markdown("**Select the sections to include in your newspaper:**")
    topic_options = ["NEWS, Top Story", "Business & Stock Market", "Sports", "Technology", "Fashion & Trends"]
    selected_topics = [topic for topic in topic_options if st.checkbox(topic, True, key=f"topic_{topic}")]

    target_language = st.selectbox("Choose Newspaper Language:", LANGUAGES)

    if available_models:
        selected_model = st.selectbox("Choose a Gemini Model for Reporting:", available_models,
                                      index=available_models.index(
                                          "gemini-1.5-pro-latest") if "gemini-1.5-pro-latest" in available_models else 0)
    else:
        st.warning("Please enter a valid Gemini API Key in the sidebar to load available models.")
        selected_model = None

    st.header("Step 2: Go to Print!")
    if st.button("Assemble Today's Newspaper"):
        if not all([st.session_state.get(k) for k in ['gemini_key', 'serper_key']]):
            st.error("🚨 Please enter both your Gemini and Serper API keys in the sidebar to continue.")
        elif not selected_model:
            st.error("❌ Cannot generate. Please provide a valid API key to load models.")
        elif not selected_topics:
            st.error("Please select at least one topic to include in the newspaper.")
        else:
            create_and_run_newspaper_crew(
                api_key=st.session_state['gemini_key'],
                serper_api_key=st.session_state['serper_key'],
                scope=scope,
                location=location,
                topics=selected_topics,
                language=target_language,
                model_name=selected_model
            )

    if "newspaper_content" in st.session_state:
        st.markdown("---")
        st.subheader(f"The {st.session_state.get('newspaper_location', 'Global')} Times")
        st.markdown(st.session_state["newspaper_content"], unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🎧 Listen to this Newspaper"):
            st.session_state['text_for_audio'] = st.session_state["newspaper_content"]
            st.info("Go to the 'Audio Suite' page to generate the audio.")
        render_download_buttons(st.session_state["newspaper_content"], "newspaper")


def create_and_run_newspaper_crew(api_key, serper_api_key, scope, location, topics, language, model_name):
    """Initializes and runs the newspaper creation crew."""
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["SERPER_API_KEY"] = serper_api_key
        llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
        search_tool = SerperDevTool(api_key=os.environ["SERPER_API_KEY"])
    except Exception as e:
        st.error(f"Error initializing services: {e}")
        return

    with st.spinner("Your AI Newsroom is on the story... This will take a few minutes."):
        editor = Agent(role='Managing Editor', goal=f'Oversee the creation of a high-quality newspaper in {language}.',
                       backstory="With decades of experience at major news outlets, you are the final word on journalistic integrity.",
                       llm=llm, allow_delegation=True, verbose=True)
        wire_service = Agent(role='News Wire Service',
                             goal='Continuously scan the web for the latest, most significant news stories.',
                             backstory="You are the digital equivalent of the Associated Press, the first to know about any breaking event.",
                             llm=llm, tools=[search_tool], verbose=True)
        reporters = [Agent(role=f'{topic.title()} Reporter',
                           goal=f'Develop in-depth, accurate, and engaging news articles on {topic} in {language}.',
                           backstory=f"You are a seasoned journalist with a deep specialization in {topic}.", llm=llm,
                           tools=[search_tool], verbose=True) for topic in topics]

        query_location = location if scope in ["Local", "National"] else "world"
        fetch_task = Task(
            description=f"Fetch the most recent and significant news stories for today, {datetime.now().strftime('%Y-%m-%d')}, for a {scope} newspaper focused on {query_location}. Compile a list of headlines, sources, and summaries.",
            agent=wire_service,
            expected_output="A structured list of current news stories, each with a headline, a URL source, and a one-sentence summary.")
        reporting_tasks = [Task(
            description=f"Using the news wire data, write a concise and compelling news article in {language} on your beat: '{topic}'. Your article MUST include a headline, a byline, a 2-3 paragraph body, and the primary source URL.",
            agent=reporter, context=[fetch_task],
            expected_output="A well-formatted news article including a headline, byline, body, and source URL.") for
                           reporter, topic in zip(reporters, topics)]

        output_filename = 'final_newspaper.md'
        editing_task = Task(
            description=f"Review all drafted articles and assemble them into a single, cohesive newspaper format in {language}. For each article, include the source URL at the end in the format: 'Source: [URL]'.",
            agent=editor, context=reporting_tasks,
            expected_output="A single, well-formatted Markdown document containing the complete newspaper with all its articles and their sources.",
            output_file=output_filename)

        crew = Crew(agents=[editor, wire_service] + reporters, tasks=[fetch_task] + reporting_tasks + [editing_task],
                    process=Process.sequential, Verbose=True)

        try:
            result = crew.kickoff()
            st.session_state['newspaper_location'] = location if location else scope
            with open(output_filename, 'r', encoding='utf-8') as file:
                st.session_state["newspaper_content"] = file.read()
        except Exception as e:
            st.error(f"An error occurred while running the AI crew: {e}")


# --- PAGE 7: VIRAL VIDEO STUDIO ---

def render_viral_video_page():
    st.title("📝 AI Viral Video Studio")
    st.markdown(
        "Create a powerful, 45-second video series concept for social media, broken into 5 compelling 8-second clips.")

    available_models = get_available_models(st.session_state.get('gemini_key'))


    st.header("Step 1: What's Your Message?")
    topic_or_verse = st.text_input("**Enter a Christian Topic or Bible Verse:**",
                                   placeholder="e.g., John 3:16, Saved by Grace")
    target_language = st.selectbox("Choose Prompt & Hook Language:", LANGUAGES)

    if available_models:
        selected_model = st.selectbox("Choose a Gemini Model for Video Concepting:", available_models,
                                      index=available_models.index(
                                          "gemini-1.5-pro-latest") if "gemini-1.5-pro-latest" in available_models else 0)
    else:
        st.warning("Please enter a valid Gemini API Key in the sidebar to load available models.")
        selected_model = None

    st.header("Step 2: Create the Vision")
    if st.button("Generate Viral Video Series Concept"):
        if not all([st.session_state.get(k) for k in ['gemini_key', 'serper_key']]):
            st.error("🚨 Please enter both your Gemini and Serper API keys in the sidebar.")
        elif not selected_model:
            st.error("❌ Cannot generate. Please provide a valid API key to load models.")
        elif not topic_or_verse:
            st.error("🚨 Please provide a topic or verse to get started.")
        else:
            create_and_run_viral_video_crew(
                api_key=st.session_state['gemini_key'],
                serper_api_key=st.session_state['serper_key'],
                topic_or_verse=topic_or_verse,
                language=target_language,
                model_name=selected_model
            )

    if "video_series_content" in st.session_state:
        st.markdown("---")
        st.subheader("✅ Your Viral Video Series Concept")
        st.markdown(st.session_state["video_series_content"])
        st.markdown("---")
        render_download_buttons(st.session_state["video_series_content"].raw, "viral_video_series_concept")


def create_and_run_viral_video_crew(api_key, serper_api_key, topic_or_verse, language, model_name):
    """Initializes and runs the viral video creation crew."""
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["SERPER_API_KEY"] = serper_api_key
        llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
    except Exception as e:
        st.error(f"Error initializing services: {e}")
        return

    with st.spinner("Your AI Social Media team is brainstorming a viral series... This might take a moment."):
        strategist = Agent(role='Viral Content Strategist',
                           goal=f'Develop a high-level concept and narrative arc for a 45-second, 5-part video series about "{topic_or_verse}".',
                           backstory='You are a master of digital storytelling, able to create compelling narratives that unfold across multiple short clips.',
                           llm=llm, verbose=True)
        storyboard_artist = Agent(role='Viral Video Storyboard Artist',
                                  goal='Break down a narrative arc into 5 distinct, visually compelling 8-second video concepts.',
                                  backstory='You are a visual thinker, able to translate a story into a sequence of powerful, attention-grabbing shots.',
                                  llm=llm, verbose=True)
        prompt_engineer = Agent(role='Google VEO Prompt Engineer',
                                goal=f'Write 5 unique, detailed VEO prompts in {language}, one for each part of a video storyboard.',
                                backstory='You are an expert in generative video AI, knowing the precise keywords to achieve cinematic quality for a series of related clips.',
                                llm=llm, verbose=True)
        hook_writer = Agent(role='Social Media Hook Writer',
                            goal=f'Write 5 unique, engaging social media hooks in {language}, one for each video clip in a series.',
                            backstory=f'You craft irresistible, scroll-stopping text that makes people want to watch the next part.',
                            llm=llm, verbose=True)
        editor = Agent(role='Content Editor',
                       goal=f'Compile all generated content into a single, well-formatted Markdown document in {language}.',
                       backstory='You are a meticulous editor who ensures the final output is clean, organized, and ready for the user.',
                       llm=llm, verbose=True)

        strategy_task = Task(
            description=f'Based on the topic "{topic_or_verse}", create a cohesive narrative arc for a 45-second video series. Define the overall story and the key message for each of the 5 parts.',
            agent=strategist, expected_output="A high-level summary of the 5-part video series concept.")
        storyboard_task = Task(
            description='Based on the narrative arc, create a detailed storyboard for the 5 video clips. For each clip, describe the scene, subject, camera movement, and emotion.',
            agent=storyboard_artist, context=[strategy_task],
            expected_output="A 5-part storyboard, with a detailed visual description for each 8-second clip.")
        prompting_task = Task(
            description=f'For each of the 5 storyboard parts, write a unique and highly detailed VEO prompt in {language}. Each prompt should be a separate paragraph.',
            agent=prompt_engineer, context=[storyboard_task],
            expected_output=f"Five distinct, detailed VEO prompts, written in {language}.")
        hook_writing_task = Task(
            description=f'For each of the 5 storyboard parts, write a unique and compelling social media hook in {language}. Number them 1 through 5.',
            agent=hook_writer, context=[storyboard_task, strategy_task],
            expected_output=f"Five numbered social media hooks, written in {language}.")
        compiling_task = Task(
            description=f'Compile the overall concept, and for each of the 5 clips, the storyboard description, the VEO prompt, and the social media hook into a single, clean Markdown document. Use clear headings for each section.',
            agent=editor, context=[strategy_task, storyboard_task, prompting_task, hook_writing_task],
            expected_output=f"A final, well-structured Markdown document containing the complete video series plan in {language}.")

        crew = Crew(agents=[strategist, storyboard_artist, prompt_engineer, hook_writer, editor],
                    tasks=[strategy_task, storyboard_task, prompting_task, hook_writing_task, compiling_task],
                    process=Process.sequential, Verbose=True)

        try:
            final_concept = crew.kickoff()
            st.session_state["video_series_content"] = final_concept

        except Exception as e:
            st.error(f"An error occurred while creating the video concept: {e}")


# --- PAGE 8: AUDIO SUITE ---

def render_audio_suite_page():
    st.title("🎧 AI Audio Suite")
    st.markdown("Your one-stop shop for audio generation and transcription.")

    st.header("Text-to-Audio Generator")
    st.markdown("Convert any text into high-quality audio in your chosen language.")

    text_to_convert = st.text_area("Enter text to convert to audio:", value=st.session_state.get('text_for_audio', ''),
                                   height=250)

    # List of available voices for the TTS model
    VOICES = ['Kore', 'Puck', 'Zephyr', 'Charon', 'Fenrir', 'Leda', 'Orus', 'Aoede', 'Callirrhoe', 'Autonoe',
              'Enceladus', 'Iapetus', 'Umbriel', 'Algieba', 'Despina', 'Erinome', 'Algenib', 'Rasalgethi', 'Laomedeia',
              'Achernar', 'Alnilam', 'Schedar', 'Gacrux', 'Pulcherrima', 'Achird', 'Zubenelgenubi', 'Vindemiatrix',
              'Sadachbia', 'Sadaltager', 'Sulafat']
    selected_voice = st.selectbox("Choose a Voice:", VOICES)
    available_audio_models = get_available_modelsAudio(st.session_state['gemini_key'])
    if available_audio_models:
        selected_model_audio = st.selectbox("Choose a Gemini Model for Video Concepting:", available_audio_models,
                                            index=available_audio_models.index(
                                                "gemini-1.5-pro-latest") if "gemini-1.5-pro-latest" in available_audio_models else 0)
    else:
        st.warning("Please enter a valid Gemini API Key in the sidebar to load available models.")
        selected_model_audio = None

    if st.button("🎤 Generate Audio"):
        if not st.session_state.get('gemini_key'):
            st.error("🚨 Please enter your Gemini API key in the sidebar.")
        elif not text_to_convert:
            st.error("🚨 Please enter some text to convert.")
        else:

            generate_and_play_audio(st.session_state['gemini_key'], text_to_convert, selected_voice,selected_model_audio)

    st.markdown("---")

    st.header("Audio-to-Text Transcription & Translation")
    st.markdown("Upload an audio file to transcribe it and translate it into another language.")

    #uploaded_audio = st.file_uploader("Upload an audio file:", type=["wav", "mp3", "m4a"])
    translation_language = st.selectbox("Translate transcript to:", LANGUAGES, key="translate_lang")
    translate = st.text_area("Enter text to Translate:" )


    if st.button("✍️ Transcribe & Translate"):
        if not st.session_state.get('gemini_key'):
            st.error("🚨 Please enter your Gemini API key in the sidebar.")
        else:
            models=get_available_models(st.session_state['gemini_key'])
            model_language = st.selectbox("Translate transcript to:", models, key="translate_model")
            if  translate:
              transcribe_and_translate_audio(st.session_state['gemini_key'], translate, translation_language,model_language)


def generate_and_play_audio(api_key, text, voice_name,model_name):
    """Generates audio from text using the Gemini TTS model and displays it."""
    with st.spinner("Generating audio..."):
        try:
            from google import genai as gen
            from google.genai import types
            os.environ["GOOGLE_API_KEY"] = api_key
            genai.configure(api_key=api_key)
            client = gen.Client(api_key=os.environ["GOOGLE_API_KEY"])

            response = client.models.generate_content(
                model=model_name,
                contents=f"Say this with a clear and engaging tone: {text}",
                config=gen.types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=gen.types.SpeechConfig(
                        voice_config=gen.types.VoiceConfig(
                            prebuilt_voice_config=gen.types.PrebuiltVoiceConfig(
                                voice_name=voice_name,
                            )
                        )
                    ),
                )
            )

            data = response.candidates[0].content.parts[0].inline_data.data
            wav_bytes = pcm_to_wav(data, channels=1, sample_width=2, sample_rate=24000)

            st.audio(wav_bytes, format='audio/wav')
            st.download_button("⬇️ Download Audio", wav_bytes, "generated_audio.mp3", "audio/mpeg")

        except Exception as e:
            st.error(f"An error occurred during audio generation: {e}")


def transcribe_and_translate_audio(api_key, text, language,model_name):
    """Transcribes and translates an audio file."""
    with st.spinner("Uploading and transcribing audio..."):
        try:
            from google import genai as gen
            from google.genai import types
            os.environ["GOOGLE_API_KEY"] = api_key
            client = gen.Client(api_key=os.environ["GOOGLE_API_KEY"])


            # # Upload the file to the Gemini API
            # uploaded_file = client.files.upload(
            #     file=audio_file
            # )
            #
            # # Transcribe



            # Translate
            with st.spinner(f"Translating transcript to {language}..."):
                translate_response = client.models.generate_content(model=model_name,
                contents=[
                    f"Translate the following text to {language}: {text}"
                ])
                translated_text = translate_response.text
                st.subheader(f"Translation ({language}):")
                st.markdown(translated_text)
                render_download_buttons(translated_text, "translated_transcript")

        except Exception as e:
            st.error(f"An error occurred during transcription/translation: {e}")

 #Function to handle the video generation and display
def single_render_viral_video_page():
    """Renders the Viral Video Studio page in Streamlit."""
    st.title("🎬 Single Viral Video Studio")
    st.markdown(
        """
    Unleash your creativity! Use AI to generate a compelling 8-second vertical video from your text prompt.
    """
    )

    # Get the Gemini API key from the user
    if not st.session_state.get('gemini_key'):
        st.error("🚨 Please enter your Gemini API key in the sidebar.")
    # Initialize the client with the provided API key
    try:
        st.markdown(get_available_modelsVeo(st.session_state.get('gemini_key')))
        from google import genai as gen
        from google.genai import types
        os.environ["GOOGLE_API_KEY"]=st.session_state.get('gemini_key')

        #genai.configure(api_key=api_key)
        client = gen.Client(api_key=os.environ["GOOGLE_API_KEY"])
    except Exception as e:
        st.error(f"Failed to configure the Gemini client: {e}")
        return

    # User input for the video prompt
    prompt = st.text_area(
        "Enter your video prompt here:",
        value="A close up of two people staring at a cryptic drawing on a wall, torchlight flickering. A man murmurs, 'This must be it. That's the secret code.' The woman looks at him and whispering excitedly, 'What did you find?'",
        height=150,
    )

    if st.button("Generate VEO Video"):
        if prompt:
            try:
                with st.spinner("Generating video... This may take a few minutes."):
                    # Use the provided code to generate the video
                    operation = client.models.generate_videos(
                        model="veo-3.0-generate-preview",
                        prompt=prompt,
                    )

                    # Poll the operation status until the video is ready.
                    status_placeholder = st.empty()
                    while not operation.done:
                        status_placeholder.info("Waiting for video generation to complete...")
                        time.sleep(10)
                        operation = client.operations.get(operation)

                    status_placeholder.success("Video generation complete!")

                    # Download the generated video.
                    generated_video = operation.response.generated_videos[0]
                    video_bytes_content = client.files.download(file=generated_video.video)
                    generated_video.video.save("viral_veo_video.mp4")
                    st.info("Generated video saved viral_veo_video.mp4")

                # Display the video in Streamlit
                st.subheader("Your Generated Video:")
                st.video(video_bytes_content)

                # Add a download button for the video
                st.download_button(
                    label="⬇️ Download Video",
                    data=video_bytes_content,
                    file_name="viral_veo_video.mp4",
                    mime="video/mp4",
                )

            except Exception as e:
                st.error(f"An error occurred during video generation: {e}")
                st.info("Please ensure you have access to the VEO model with your Gemini API key.")
        else:
            st.warning("Please enter a prompt to generate a video.")


def read_uploaded_file(uploaded_file):
    """Reads content from an uploaded file (txt, pdf, docx)."""
    name = uploaded_file.name
    if name.endswith('.txt'):
        return uploaded_file.getvalue().decode("utf-8")
    elif name.endswith('.pdf'):
        pdf_reader = pypdf.PdfReader(uploaded_file)
        return "\n".join(page.extract_text() for page in pdf_reader.pages)
    elif name.endswith('.docx'):
        doc = docx.Document(uploaded_file)
        return "\n".join(para.text for para in doc.paragraphs)
    return None


def create_downloadable_docx(content):
    """Converts markdown content to a downloadable DOCX file in memory."""
    doc = docx.Document()
    doc.add_paragraph(content)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


def create_downloadable_html(content):
    """Converts markdown content to a downloadable HTML file."""
    html_content = markdown2.markdown(content)
    return html_content.encode("utf-8")


# ==============================================================================
## 2. AI Crew Definitions
# Contains the logic for both the school-level and university-level AI crews.
# ==============================================================================

def run_school_tutor_crew(api_key, country, grade, subject, language, question,model_name):
    """Initializes and runs the AI Tutor Crew for Grades 1-12."""
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])

    # Define Agents
    curriculum_analyst = Agent(role='Curriculum Analyst',
                               backstory="Expert in global K-12 education systems, ensuring explanations are pedagogically sound.",
                               goal=f"Analyze the educational context for a {grade} student in {country} studying {subject}. Determine the appropriate depth and tone.",
                               llm=llm)
    subject_expert = Agent(role=f'{subject.title()} Subject Matter Expert',
                           backstory=f"Renowned teacher in {subject} with a passion for clarity.",
                           goal=f"Accurately solve the student's homework question about {subject}.", llm=llm)
    pedagogy_expert = Agent(role='Pedagogy and Language Expert',
                            backstory="Master educator skilled at adapting complex information for different age groups.",
                            goal=f"Rewrite the expert's solution into an engaging answer in {language} for a {grade} student.",
                            llm=llm)

    # Define Tasks
    task1 = Task(
        description=f"Analyze context: Country {country}, Grade {grade}, Subject {subject}. Plan how to best explain the answer to: '{question}'",
        expected_output="A brief plan on key concepts, depth, and tone.", agent=curriculum_analyst)
    task2 = Task(description=f"Solve this homework question: '{question}'",
                 expected_output="A correct, step-by-step solution.", agent=subject_expert, context=[task1])
    task3 = Task(
        description=f"Take the expert's solution and rewrite it in {language} as a friendly, clear markdown explanation for a student.",
        expected_output="A complete, well-formatted markdown document in {language} with a friendly tone.",
        agent=pedagogy_expert, context=[task2])

    crew = Crew(agents=[curriculum_analyst, subject_expert, pedagogy_expert], tasks=[task1, task2, task3],
                process=Process.sequential)
    return crew.kickoff()


def run_university_tutor_crew(api_key, course, language, question,model_name):
    """Initializes and runs the University AI Professor Crew."""
    os.environ["GOOGLE_API_KEY"] = api_key
    llm =LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])

    # Define Agents
    curriculum_specialist = Agent(role='University Curriculum Specialist',
                                  backstory="Academic advisor with encyclopedic knowledge of university course structures.",
                                  goal=f"Analyze the curriculum for a university course titled '{course}'. Determine the expected academic rigor and prerequisite knowledge.",
                                  llm=llm)
    lead_professor = Agent(role=f'University Professor of {course}',
                           backstory=f"Distinguished professor with a Ph.D. and extensive research experience relevant to {course}.",
                           goal=f"Provide a rigorous, technically correct solution to the student's question.", llm=llm)
    academic_tutor = Agent(role='Senior Academic Tutor',
                           backstory="Award-winning teaching assistant skilled at making complex ideas click.",
                           goal=f"Refine the professor's solution into a comprehensive explanation in {language}, connecting it to broader course concepts.",
                           llm=llm)

    # Define Tasks
    task1 = Task(
        description=f"Analyze the academic framework for the course '{course}' to answer the question: '{question}'.",
        expected_output="An academic plan outlining the theoretical foundations and expected depth.",
        agent=curriculum_specialist)
    task2 = Task(description=f"Provide an expert, in-depth solution to the question: '{question}'.",
                 expected_output="A detailed, technically accurate, step-by-step solution.", agent=lead_professor,
                 context=[task1])
    task3 = Task(
        description=f"Synthesize the solution into a high-quality tutorial explanation in {language}, formatted in professional markdown.",
        expected_output=f"A complete tutorial in {language} linking the solution to core concepts of '{course}'.",
        agent=academic_tutor, context=[task2])

    crew = Crew(agents=[curriculum_specialist, lead_professor, academic_tutor], tasks=[task1, task2, task3],
                process=Process.sequential)
    return crew.kickoff()


# ==============================================================================
## 3. Page Rendering Functions
# Each function defines the UI and logic for a specific page.
# ==============================================================================

def render_tutor_page():
    """Renders the AI Tutor Studio page for Grades 1-12."""
    st.title("🎓 AI Tutor Studio (Grades 1-12)")
    st.markdown("Your personal AI learning assistant. Get step-by-step help with your homework.")

    with st.form("school_tutor_form"):
        st.subheader("Tell us about your homework")
        col1, col2 = st.columns(2)
        country = col1.text_input("Country", "Germany")
        subject = col1.selectbox("Subject", ["Mathematics", "History", "Science", "Literature", "Physics"])
        grade = col2.selectbox("Grade / Class", [f"Grade {i}" for i in range(1, 14)])
        language = col2.selectbox("Language for Explanation", LANGUAGES)
        question = st.text_area("📝 Enter your question here", height=150)
        uploaded_file = st.file_uploader("Or upload a file (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"],
                                         key="school_uploader")
        submitted = st.form_submit_button("Get Help from AI Tutors!", use_container_width=True)
    available_models = get_available_models(st.session_state.get('gemini_key'), task="generateContent")

    if available_models:
        default_model = "gemini-1.5-pro-latest"
        selected_model = st.selectbox("Choose a Gemini Model:", available_models, index=available_models.index(
            default_model) if default_model in available_models else 0)
    else:
        st.warning("Please enter a valid Gemini API Key in the sidebar to load available models.")
        selected_model = None



    if submitted:
        if not question and not uploaded_file:
            st.error("Please enter a question or upload a file.")
        else:
            content = question + (
                f"\n\n--- FROM FILE ---\n{read_uploaded_file(uploaded_file)}" if uploaded_file else "")
            with st.spinner("🚀 Your AI Tutors are working on it..."):
                st.session_state.school_result = run_school_tutor_crew(st.session_state.get('gemini_key'), country, grade, subject, language,
                                                                       content,selected_model)

    if "school_result" in st.session_state and st.session_state.school_result:
        st.markdown("---")
        st.subheader("✨ Here's your explanation:")
        st.markdown(st.session_state.school_result)
        # Add download buttons if needed, similar to the university page


def render_university_tutor_page():
    """Renders the University AI Professor page."""
    st.title("🧑‍🏫 University AI Professor")
    st.markdown("Get expert-level academic help for your university courses.")

    with st.form("uni_tutor_form"):
        st.subheader("Provide your course and question details")
        course = st.text_input("University Course Name", placeholder="e.g., Experimental Physics, Linear Algebra II")
        language = st.selectbox("Language for Explanation",LANGUAGES)
        question = st.text_area("📝 Enter your question or problem here", height=150)
        uploaded_file = st.file_uploader("Or upload a problem set or paper (TXT, PDF, DOCX)",
                                         type=["txt", "pdf", "docx"], key="uni_uploader")
        submitted = st.form_submit_button("Consult the Professor", use_container_width=True)
        available_models = get_available_models(st.session_state.get('gemini_key'), task="generateContent")

        if available_models:
            default_model = "gemini-1.5-pro-latest"
            selected_model = st.selectbox("Choose a Gemini Model:", available_models, index=available_models.index(
                default_model) if default_model in available_models else 0)
        else:
            st.warning("Please enter a valid Gemini API Key in the sidebar to load available models.")
            selected_model = None

    if submitted:
        if not course:
            st.error("Please enter your university course name.")
        elif not question and not uploaded_file:
            st.error("Please enter a question or upload a file.")
        else:
            content = question + (
                f"\n\n--- FROM FILE ---\n{read_uploaded_file(uploaded_file)}" if uploaded_file else "")
            with st.spinner("🚀 Consulting with the AI academic team..."):
                st.session_state.uni_result = run_university_tutor_crew(st.session_state.get('gemini_key'), course, language, content,selected_model)

    if "uni_result" in st.session_state and st.session_state.uni_result:
        st.markdown("---")
        st.subheader("✨ Professor's Explanation:")
        st.markdown(st.session_state.uni_result)
        # Add download buttons



# --- NEW: Chef Crew Definition ---
def run_chef_crew(api_key, country, food_type,model_name):
    """Initializes and runs the AI Chef Crew."""
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])

    # Define the Culinary Agents
    cuisine_specialist = Agent(
        role='World Cuisine & Dietary Specialist',
        goal=f"Generate 3 diverse and exciting meal ideas (Appetizer, Main Course, Dessert) that fit the {food_type} category and are inspired by the cuisine of {country}.",
        backstory="An acclaimed food historian and globetrotter who understands the soul of a country's food and the principles of dietary choices like veganism. Your suggestions are authentic and inspiring.",
        llm=llm,
        verbose=True
    )

    master_chef = Agent(
        role='Executive Chef & Recipe Developer',
        goal="Write clear, concise, and easy-to-follow recipes for the meal ideas provided. Each recipe must include an ingredient list (with metric and imperial measurements), step-by-step instructions, and estimated prep/cook times.",
        backstory="A Michelin-trained chef with a passion for teaching home cooks. You can deconstruct any dish into simple, foolproof steps. Your recipes are reliable and always delicious.",
        llm=llm,
        verbose=True
    )

    food_stylist = Agent(
        role='Food Blogger & Creative Director',
        goal="Format the recipes into a beautiful markdown file. For each of the 3 complete meals, write a tantalizing description and a detailed, effective image generation prompt for Gemini to visualize the final dishes.",
        backstory="A top-tier food blogger and photographer who knows how to make food look irresistible. You are an expert in crafting prompts for AI image generators to create stunning, photorealistic food photography.",
        llm=llm,
        verbose=True
    )

    # Define the Culinary Tasks
    task_brainstorm = Task(
        description=f"Brainstorm 3 complete meal ideas (appetizer, main, dessert) based on {country}'s cuisine for a {food_type} diet.",
        expected_output="A list of 3 distinct meal plans, each containing a name for an appetizer, a main course, and a dessert.",
        agent=cuisine_specialist
    )

    task_develop_recipes = Task(
        description="For each of the 3 meal plans, write a full, detailed recipe for the appetizer, main course, and dessert.",
        expected_output="A complete set of recipes. Each recipe must have a title, ingredient list, and step-by-step instructions.",
        agent=master_chef,
        context=[task_brainstorm]
    )

    task_format_and_present = Task(
        description="Combine all the recipes into a single, beautifully formatted markdown document. For each of the 3 meals, add a mouth-watering description and a specific, copy-paste ready prompt for an AI image generator (like Gemini) to create a picture of the main course.",
        expected_output="A final, user-ready markdown document containing descriptions, recipes for 3 full meals, and 3 distinct, detailed image generation prompts.",
        agent=food_stylist,
        context=[task_develop_recipes]
    )

    chef_crew = Crew(
        agents=[cuisine_specialist, master_chef, food_stylist],
        tasks=[task_brainstorm, task_develop_recipes, task_format_and_present],
        process=Process.sequential
    )

    return chef_crew.kickoff()


def render_chef_page():
    """Renders the AI Chef Studio page."""
    st.title("🍳 AI Chef Studio")
    st.markdown("Your personal guide to culinary discovery. Get complete meal plans and recipes from around the world.")

    with st.form("chef_form"):
        st.subheader("What are you in the mood for?")
        col1, col2 = st.columns(2)
        country = col1.text_input("Enter a Country or Region", placeholder="e.g., Italy, Thailand, Mexico")
        food_type = col2.selectbox("Select a Food Type", ["Any", "Meat", "Vegetarian", "Vegan"])
        available_models = get_available_models(st.session_state.get('gemini_key'), task="generateContent")

        if available_models:
            default_model = "gemini-1.5-pro-latest"
            selected_model = st.selectbox("Choose a Gemini Model:", available_models, index=available_models.index(
                default_model) if default_model in available_models else 0)
        else:
            st.warning("Please enter a valid Gemini API Key in the sidebar to load available models.")
            selected_model = None

        submitted = st.form_submit_button("Generate Meal Ideas", use_container_width=True)


    if submitted:
        if not country:
            st.error("Please enter a country or region.")
        else:
            with st.spinner("👩‍🍳 The AI Chef Crew is crafting your menu... This might take a minute!"):
                try:
                    # Using session_state to store the result
                    st.session_state.chef_result = run_chef_crew(st.session_state.get('gemini_key'), country, food_type, selected_model)
                except Exception as e:
                    st.error(f"An error occurred while communicating with the crew: {e}")
                    st.session_state.chef_result = ""

    if "chef_result" in st.session_state and st.session_state.chef_result:
        st.markdown("---")
        st.subheader("Your Custom Meal & Recipe Plan")
        st.markdown(st.session_state.chef_result)

        # # Add download buttons
        # st.markdown("---")
        # st.subheader("⬇️ Download Your Full Recipe Plan")
        # col1, col2, col3 = st.columns(3)
        # recipe_text = st.session_state.chef_result
        # file_name_base = f"{country}_{food_type}_recipes"
        #
        # col1.download_button(
        #     "As Markdown (.md)", recipe_text.encode('utf-8'), f"{file_name_base}.md", "text/markdown"
        # )
        # col2.download_button(
        #     "As Text File (.txt)", recipe_text.encode('utf-8'), f"{file_name_base}.txt", "text/plain"
        # )
        # col3.download_button(
        #     "As Word Doc (.docx)", create_downloadable_docx(recipe_text), f"{file_name_base}.docx",
        #     "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        # )


# --- NEW: Image Studio Page ---
def render_image_studio_page():
    """Renders the AI Image Studio page."""
    st.title("🎨 AI Image Studio")
    st.markdown("Bring your ideas to life. Generate stunning visuals with a simple text prompt.")
    available_image_models = get_available_modelsImage(st.session_state.get('gemini_key'))
    if available_image_models:
        default_model = "gemini-1.5-pro-latest"
        selected_model = st.selectbox("Choose a Gemini Model:", available_image_models, index=available_image_models.index(
            default_model) if default_model in available_image_models else 0)
    else:
        st.warning("Please enter a valid Gemini API Key in the sidebar to load available models.")
        selected_model = None




    with st.form("image_form"):
        st.subheader("Craft Your Vision")
        prompt = st.text_area("Enter your image prompt", height=100,
                              placeholder="e.g., A majestic lion wearing a crown, sitting on a throne in a futuristic jungle.")

        st.subheader("Configure Your Image")
        col1, col2, col3 = st.columns(3)

        num_images = col1.slider("Number of Images", 1, 4, 1)
        aspect_ratio = col2.selectbox("Aspect Ratio", ["1:1", "3:4", "4:3", "9:16", "16:9"], index=0)
        person_gen_options = {
            "Allow Adults (Default)": "allow_adult",
            "Allow Adults & Children": "allow_all",
            "Don't Allow People": "dont_allow"
        }
        person_gen_display = col3.selectbox("People in Image", options=list(person_gen_options.keys()))
        person_gen_value = person_gen_options[person_gen_display]

        submitted = st.form_submit_button("Generate Images", use_container_width=True)



    if submitted:
        if not prompt:
            st.error("Please enter a prompt to generate an image.")
        else:
            with st.spinner("🎨 The AI is painting your masterpiece..."):
                # Call the backend function to get the images
                generated_images = generate_images_from_prompt(st.session_state.get('gemini_key'), prompt, num_images, aspect_ratio,
                                                               person_gen_value,  selected_model)

                if generated_images:
                    st.session_state.generated_images = generated_images
                else:
                    st.session_state.generated_images = []

    if "generated_images" in st.session_state and st.session_state.generated_images:
        st.markdown("---")
        st.subheader("Your Generated Images")

        # Create a dynamic number of columns for the image gallery
        cols = st.columns(len(st.session_state.generated_images))
        for i, img in enumerate(st.session_state.generated_images):
            with cols[i]:
                # The 'img' object from the SDK is a PIL.Image object
                st.image(img, caption=f"Image {i + 1}", use_column_width=True)

                # Provide a download button for each image
                st.download_button(
                    label="Download",
                    data=img,
                    file_name=f"generated_image_{i + 1}.png",
                    mime="image/png"
                )
# --- NEW: Bible Study Crew ---
def run_bible_study_crew(api_key, serper_api_key, topic, testament, language,model_name):
    """
    Runs a Crew AI team to find scriptures and write a devotional.
    """
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["SERPER_API_KEY"] = serper_api_key
    llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
    search_tool = SerperDevTool(api_key=os.environ["SERPER_API_KEY"])

    # Define the Bible Study Agents
    scholar = Agent(
        role='Bible Scholar and Researcher',
        goal=f"Conduct a comprehensive search of the Bible to find at least 20 relevant verses for the topic: '{topic}'. The search must be limited to the {testament} Testament(s).",
        backstory="You are a meticulous and knowledgeable Bible scholar with a deep understanding of biblical languages and contexts. You use digital tools like Bible Gateway to perform precise and exhaustive scripture searches.",
        tools=[search_tool],
        llm=llm,
        verbose=True
    )

    pastor = Agent(
        role='Pentecostal Pastor and Theologian',
        goal=f"Write a short, inspiring devotional or sermon outline based on the Bible verses provided. The message should be written in {language} and reflect a Pentecostal passion for Jesus and love for people.",
        backstory="You are a seasoned pastor with a gift for teaching. You can take a list of scriptures and weave them into a powerful, practical, and encouraging message that speaks to the heart.",
        llm=llm,
        verbose=True
    )

    # Define the Study Tasks
    task_find_verses = Task(
        description=f"Search for scriptures related to '{topic}' within the {testament} Testament(s). Use search queries like 'bible verses about {topic} in the {testament} testament on Bible Gateway'. Compile a list of the top 10-15 most relevant verses.",
        expected_output=f"A markdown-formatted list of 10-15 bible verses, each with its reference (e.g., John 3:16). The entire list should be in {language}.",
        agent=scholar
    )

    task_write_sermon = Task(
        description="Using the list of scriptures from the scholar, write an inspiring devotional. Start with a compelling introduction, explain the key themes found in the verses, and conclude with a practical application or encouragement for the reader.",
        expected_output=f"A complete devotional message in {language}, approximately 300-500 words long, formatted in markdown with a title, introduction, body, and conclusion.",
        agent=pastor,
        context=[task_find_verses]
    )

    bible_crew = Crew(
        agents=[scholar, pastor],
        tasks=[task_find_verses, task_write_sermon],
        process=Process.sequential
    )
    return bible_crew.kickoff()


# --- NEW: Bible Study Page ---
def render_bible_search():
    """Renders the AI Bible Study Assistant page."""
    st.title("🙏 AI Bible Study Assistant")
    available_models = get_available_models(st.session_state.get('gemini_key'), task="generateContent")
    st.markdown(
        "Your personal theology research partner. Enter a topic to discover relevant scriptures and receive a custom devotional.")
    if available_models:
        default_model = "gemini-1.5-pro-latest"
        selected_model = st.selectbox("Choose a Gemini Model:", available_models, index=available_models.index(
            default_model) if default_model in available_models else 0)
    else:
        st.warning("Please enter a valid Gemini API Key in the sidebar to load available models.")
        selected_model = None

    with st.form("bible_study_form"):
        st.subheader("Start Your Study")
        topic = st.text_input("Enter a Topic, Theme, or Question",
                              placeholder="e.g., Faith, Forgiveness, Who is the Holy Spirit?")

        col1, col2 = st.columns(2)
        language = col1.text_input("Language for Results",  placeholder="e.g German, French, Swahili etc." )
        testament = col2.selectbox("Select Testament(s)", ["All", "Old Testament", "New Testament"])

        submitted = st.form_submit_button("Begin Bible Study", use_container_width=True)

    if submitted:
        if not topic or not language:
            st.error("Please provide a Topic and a Language.")
        else:
            with st.spinner("The AI ministry team is studying God's Word for you..."):
                try:
                    # Using session_state to store the result
                    st.session_state.study_result = run_bible_study_crew(st.session_state.get('gemini_key'),
                                                                         st.session_state['serper_key'] ,
                                                                         topic, testament,
                                                                         language,selected_model)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.session_state.study_result = ""

    if "study_result" in st.session_state and st.session_state.study_result:
        st.markdown("---")
        st.subheader(f"Study Results on '{topic}'")
        st.markdown(st.session_state.study_result)

        st.markdown("---")
        # st.subheader("⬇️ Download Your Study")
        # col1, col2 = st.columns(2)
        #
        # col1.download_button(
        #     "As Markdown (.md)", st.session_state.study_result,
        #     f"bible_study_{topic.replace(' ', '_')}.md"
        # )
        # col2.download_button(
        #     "As Word Doc (.docx)", create_downloadable_docx(st.session_state.study_result),
        #     f"bible_study_{topic.replace(' ', '_')}.docx"
        # )
# --- NEW: Language Learning Crew ---
def run_language_crew(api_key, native_language, target_language, level,model_name):
    """
    Runs a Crew AI team to generate a language learning study guide.
    """
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])
    # Define the Language Learning Agents
    curriculum_designer = Agent(
        role='Polyglot Curriculum Designer',
        goal=f"Outline a 10-lesson study plan for a {native_language} speaker learning {target_language} at the {level} level. The plan should cover key grammar, vocabulary themes, and conversational skills appropriate for this level.",
        backstory="You are an expert linguist and curriculum designer for a world-renowned language school. You create structured, effective learning paths that guide students from one proficiency level to the next.",
        llm=llm,
        verbose=True
    )

    grammar_specialist = Agent(
        role='Grammar & Vocabulary Specialist',
        goal=f"Create the detailed content for each of the 10 lessons. For each lesson, provide a clear grammar explanation, a list of essential vocabulary with translations to {native_language}, and example sentences.",
        backstory=f"You are a language professor specializing in {target_language}. You have a gift for explaining complex grammar rules simply and providing vocabulary that is immediately useful for learners.",
        llm=llm,
        verbose=True
    )

    language_coach = Agent(
        role='Language Coach & Pronunciation Guide',
        goal=f"Enhance the 10-lesson guide with practical learning tips, pronunciation guides for difficult sounds, and a 'Takeaway Summary' for each lesson. The tone should be encouraging and motivating.",
        backstory="You are a popular language coach who helps thousands of students achieve fluency. You know the common pitfalls and provide practical, confidence-boosting advice to make learning stick.",
        llm=llm,
        verbose=True
    )

    # Define the Learning Plan Tasks
    task_design_plan = Task(
        description=f"Create a 10-lesson curriculum outline for a {native_language} speaker learning {target_language} at level {level}. Define a specific topic for each lesson (e.g., 'Lesson 1: Greetings & Basic Introductions').",
        expected_output="A numbered list of 10 lesson titles, each with a brief description of the grammar point and vocabulary theme to be covered.",
        agent=curriculum_designer
    )

    task_create_content = Task(
        description="Based on the 10-lesson curriculum, write the detailed content for each lesson. Each lesson must include a grammar explanation with examples and a vocabulary list with translations.",
        expected_output=f"A complete, 10-lesson document. Each lesson should be clearly marked and contain a 'Grammar Focus' section and a 'Vocabulary' table (word in {target_language}, translation in {native_language}).",
        agent=grammar_specialist,
        context=[task_design_plan]
    )

    task_add_coaching = Task(
        description="Review the 10-lesson document and enrich each lesson. Add a 'Pronunciation Pointer' section for tricky sounds, a 'Learning Tip' with practical advice, and a concise 'Takeaway Summary' at the end of each lesson.",
        expected_output="The final, comprehensive 10-lesson study guide formatted in clear markdown. Each lesson must be fully self-contained with all five sections: Grammar, Vocabulary, Pronunciation, Tips, and Summary.",
        agent=language_coach,
        context=[task_create_content]
    )

    language_crew = Crew(
        agents=[curriculum_designer, grammar_specialist, language_coach],
        tasks=[task_design_plan, task_create_content, task_add_coaching],
        process=Process.sequential
    )
    return language_crew.kickoff()


# --- NEW: Language Academy Page ---
def render_language_academy_page():
    """Renders the AI Language Academy page."""
    st.title("🌍 AI Language Academy")
    available_models = get_available_models(st.session_state.get('gemini_key'), task="generateContent")
    if available_models:
        default_model = "gemini-1.5-pro-latest"
        selected_model = st.selectbox("Choose a Gemini Model:", available_models, index=available_models.index(
            default_model) if default_model in available_models else 0)
    else:
        st.warning("Please enter a valid Gemini API Key in the sidebar to load available models.")
        selected_model = None
    st.markdown(
        "Your personal AI tutor for mastering a new language. Get a complete, 10-lesson study guide tailored to your needs.")
    with st.form("language_form"):
        st.subheader("Set Up Your Learning Path")

        col1, col2, col3 = st.columns(3)
        native_language = col1.text_input("Your Language", placeholder="e.g., German")
        target_language = col2.text_input("Language to Learn", placeholder="e.g., French")
        level = col3.selectbox("Select Your Level (CEFR)", ["A1", "A2", "B1", "B2", "C1", "C2"])
        submitted = st.form_submit_button("Generate My Study Guide", use_container_width=True)

    if submitted:
        if not native_language or not target_language:
            st.error("Please specify both your native language and the language you want to learn.")
        else:
            with st.spinner(
                    f"The AI teaching crew is building your {level} {target_language} curriculum... This may take a few minutes."):
                try:
                    st.session_state.language_result = run_language_crew(st.session_state.get('gemini_key'), native_language, target_language,
                                                                         level,selected_model)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.session_state.language_result = ""

    if "language_result" in st.session_state and st.session_state.language_result:
        st.markdown("---")
        st.subheader(f"Your {target_language} ({level}) Study Guide")
        st.markdown(st.session_state.language_result)


# --- NEW: Podcast Crew & TTS Functions ---

def run_podcast_crew(api_key, country, language, topic, model_name):
    """
    Runs a Crew AI team to generate a podcast script prompt.
    """
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = LLM(model=model_name, temperature=0.7, api_key=os.environ["GOOGLE_API_KEY"])

    # Define the Podcast Production Crew
    researcher = Agent(
        role='Cultural Researcher & Pentecostal Theologian',
        goal=f"Research the topic '{topic}' from a Pentecostal Christian perspective, tailoring the key points and analogies to be highly relevant and impactful for an audience in {country}.",
        backstory="You are a passionate pastor and scholar with a deep love for Jesus and people. You understand how to connect timeless biblical truths to specific cultural contexts, making the message feel personal and alive.",
        llm=llm,
        verbose=True
    )

    scriptwriter = Agent(
        role='Engaging Podcast Scriptwriter',
        goal=f"Write a conversational, two-speaker podcast script based on the researcher's key points. The script should be for a 10-minute episode, written in {language}, and feature a Host and a Guest.",
        backstory="You are a gifted storyteller who writes for a popular Christian podcast with billions of followers. You excel at creating dialogue that is natural, engaging, and spiritually uplifting.",
        llm=llm,
        verbose=True
    )

    producer = Agent(
        role='Podcast Show Producer',
        goal="Format the final script into a perfect, copy-paste ready prompt for a multi-speaker Text-to-Speech model. The prompt must start with 'TTS the following conversation between Host and Guest:' followed by the dialogue.",
        backstory="You are a meticulous producer who knows exactly how to format scripts for AI voice generation. Your work ensures a flawless transition from text to high-quality audio.",
        llm=llm,
        verbose=True
    )

    # Define the Production Tasks
    task_research = Task(
        description=f"Develop 3-4 key talking points for a podcast on '{topic}', ensuring they are spiritually deep and culturally relevant for people in {country}.",
        expected_output=f"A bulleted list of key themes, supporting scriptures, and culturally specific analogies, all in {language}.",
        agent=researcher
    )
    task_script = Task(
        description="Using the key points, write a full podcast script for two speakers (Host, Guest). The tone should be warm, encouraging, and conversational.",
        expected_output="A complete podcast script in {language}, with clear labels for 'Host:' and 'Guest:' for each line of dialogue.",
        agent=scriptwriter,
        context=[task_research]
    )
    task_format = Task(
        description="Take the final script and format it into a single block of text ready for the TTS model. Ensure it starts with the required header.",
        expected_output="A single text block starting with 'TTS the following conversation between Host and Guest:' followed by the entire script.",
        agent=producer,
        context=[task_script]
    )

    podcast_crew = Crew(
        agents=[researcher, scriptwriter, producer],
        tasks=[task_research, task_script, task_format],
        process=Process.sequential
    )
    return podcast_crew.kickoff()


def generate_podcast_audio(api_key, prompt, model_name):
    """
    Calls the Google GenAI API to generate multi-speaker audio.
    Returns the raw PCM audio data.
    """
    try:
        from google import genai as gen
        from google.genai import types
        os.environ["GOOGLE_API_KEY"]=api_key
        #client = gen.Client(api_key=os.environ["GOOGLE_API_KEY"])
        client = gen.Client(api_key=os.environ["GOOGLE_API_KEY"])

        # Define two distinct voices for the podcast
        speaker_configs = [
            types.SpeakerVoiceConfig(
                speaker='Host',
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Kore'))
                # A firm, clear voice for the host
            ),
            types.SpeakerVoiceConfig(
                speaker='Guest',
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Sadachbia'))
                # A lively, knowledgeable voice for the guest
            ),
        ]

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=[
                            types.SpeakerVoiceConfig(
                                speaker='Host',
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name='Kore',
                                    )
                                )
                            ),
                            types.SpeakerVoiceConfig(
                                speaker='Guest',
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name='Puck',
                                    )
                                )
                            ),
                        ]
                    )
                )
            )
        )
        # Extract the audio data
        return response.candidates[0].content.parts[0].inline_data.data
    except Exception as e:
        st.error(f"An error occurred during audio generation: {e}")
        return None


# --- NEW: Podcast Studio Page ---
def render_podcast_studio_page():
    """Renders the AI Podcast Studio page."""
    st.title("🎙️ AI Podcast Studio")
    st.markdown("Create and download a 10-minute, two-speaker podcast on any topic, tailored to your audience.")
    available_models = get_available_models(st.session_state.get('gemini_key'), task="generateContent")
    if available_models:
        default_model = "gemini-1.5-pro-latest"
        selected_model = st.selectbox("Choose a Gemini Model:", available_models, index=available_models.index(
            default_model) if default_model in available_models else 0)
    else:
        st.warning("Please enter a valid Gemini API Key in the sidebar to load available models.")
        selected_model = None

    st.subheader("1. Define Your Podcast Episode")
    col1, col2, col3 = st.columns(3)
    country = col1.text_input("Target Country", placeholder="e.g., Nigeria, Brazil, USA")
    language = col2.text_input("Language", placeholder="e.g., English, Yoruba, Portuguese")
    topic = col3.text_input("Episode Topic", placeholder="e.g., The Power of Forgiveness")

    st.subheader("2. Choose Your Script Source")
    prompt_source = st.radio(
        "How do you want to create the podcast script?",
        ["Let the AI Crew write the script for me", "I will provide my own custom script"],
        key="prompt_choice"
    )

    # Initialize session state for the prompt
    if 'podcast_prompt' not in st.session_state:
        st.session_state.podcast_prompt = ""

    if prompt_source == "Let the AI Crew write the script for me":
        if st.button("Generate Script with AI Crew", use_container_width=True):
            if not all([country, language, topic]):
                st.error("Please fill in Country, Language, and Topic to generate a script.")
            else:
                with st.spinner("The AI Production Crew is writing your podcast..."):
                    st.session_state.podcast_prompt = run_podcast_crew(st.session_state.get('gemini_key'), country, language, topic, selected_model)
    else:
        st.session_state.podcast_prompt = st.text_area(
            "Enter your custom script here",
            height=250,
            placeholder="Start with 'TTS the following conversation between Speaker1 and Speaker2:' followed by the dialogue.",
            value=st.session_state.podcast_prompt
        )

    if st.session_state.podcast_prompt:
        st.markdown("---")
        st.subheader("Generated Script Prompt:")
        st.text_area("Script ready for audio generation:", value=st.session_state.podcast_prompt, height=200,
                     disabled=True)

        st.markdown("---")
        st.subheader("3. Generate and Download Your Podcast")
        available_models_audio = get_available_modelsAudio(st.session_state.get('gemini_key'))
        if available_models_audio:
            default_model = "gemini-1.5-pro-latest"
            selected_model_audio = st.selectbox("Choose a Gemini Model:", available_models_audio, index=available_models_audio.index(
                default_model) if default_model in available_models_audio else 0)
        else:
            st.warning("Please enter a valid Gemini API Key in the sidebar to load available models.")
            selected_model_audio = None
        if st.button("Create Podcast Audio File (.wav)", use_container_width=True):
            with st.spinner("🎙️ The AI is recording your podcast... This may take a few moments."):
                audio_data = generate_podcast_audio(st.session_state.get('gemini_key'), st.session_state.podcast_prompt,selected_model_audio)
                if audio_data:
                    st.success("Podcast audio generated successfully!")

                    # Create a WAV file in memory
                    wav_io = io.BytesIO()
                    with wave.open(wav_io, "wb") as wf:
                        wf.setnchannels(1)  # Mono
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(24000)  # 24kHz sample rate
                        wf.writeframes(audio_data)
                    wav_io.seek(0)

                    st.audio(wav_io, format='audio/wav')

                    st.download_button(
                        label="Download Podcast (.wav)",
                        data=wav_io,
                        file_name=f"{topic.replace(' ', '_')}_podcast.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )


# --- MAIN APP ROUTER ---

def main():
    st.sidebar.title("Navigation")
    page_options = {
        "Home": "🏠",
        "Sermon Generator": "📖",
        "Flyer Production Studio": "🚀",
        "AI Image Studio": "🎨",
        "Worship Song Studio": "🎶",
        "Book Writing Studio": "📚",
        "Bible Study Generator": "🌍",
        "AI Bible Search": "🙏",
        "Newsroom HQ": "�",
        "Viral Video Studio": "📝",
        "Single Video Studio": "🎬",
        "Audio Suite": "🎧",
        "AI Tutor (Grades 1-12)": "🎓",
        "University AI Professor": "🧑‍🏫",
        "AI Chef Studio": "🍳",
        "AI Language Academy": "🌍",
        "AI Podcast Studio": "🎙️",
    }
    selection = st.sidebar.radio("Go to", list(page_options.keys()))

    if selection == "Home":
        st.title("✨ Welcome to the AI Ministry & Content Suite!")
        st.markdown("---")
        st.header("Your AI-Powered Partner in Communication")
        st.markdown("""
        This suite combines powerful tools to help you create and communicate your message effectively:

        - **📖 Sermon Generator:** A collaborative team of AI agents to help you craft deep, biblically-sound, and engaging sermons.
        - **🚀 Flyer Production Studio:** An AI design agency that produces a stunning flyer image and compelling social media copy.
         - **🎨 AI Image Studio:** Bring your ideas to life! Generate stunning, high-quality visuals from a simple text prompt with advanced configuration options.
        - **🎶 Worship Song Studio:** An AI music collective that writes lyrics and creates a production-ready prompt for generative music AI.
        - **📚 Book Writing Studio:** Your personal AI writer's room to outline and draft a book in multiple languages.
        - **🌍 Bible Study Generator:** An AI team that creates in-depth, multilingual study guides for any book of the Bible.
        - **🙏 AI Bible Study Assistant:** Your personal theology research partner. Find key scriptures on any topic and receive a custom devotional.
        - **📰 Newsroom HQ:** Commission a complete digital newspaper with your own team of AI journalists.
        - **🎬 Viral Video Studio:** An AI creative team that concepts a powerful, multi-part vertical video series and generates all the prompts and social hooks.
        - **🎧 Audio Suite:** Convert your generated text to speech or transcribe and translate existing audio files.
         - **🍳 AI Chef Studio:** Discover global cuisines! Get complete meal plans (appetizer, main, dessert), full recipes, and even AI prompts to visualize your food.
        - **🧑‍🏫 University AI Professor:** An academic assistant providing expert-level help for specific university and college courses.
        - **🎓 AI Tutor (Grades 1-12):** An interactive learning assistant for school students, providing clear, curriculum-aware explanations for homework.
        - **🌍 AI Language Academy:** Get a personalized 10-lesson study guide for any language and proficiency level.
        - **🎙️ AI Podcast Studio:** Create a 10-minute, two-speaker podcast on any topic, tailored to your culture and language, and ready for download.

        ### How to Get Started:
        1.  **Configure Credentials:** Enter your **Gemini API Key** and **Serper API Key** in the sidebar. 
        2.  **Navigate:** Use the sidebar navigation to select the tool you want to use.
        3.  **Create:** Follow the instructions on each page to generate your content.
        """)
        st.markdown("---")

    elif selection == "Sermon Generator":
        render_sermon_page()
    elif selection == "Flyer Production Studio":
        render_flyer_page()
    elif selection == "AI Image Studio":
        render_image_studio_page()
    elif selection == "Worship Song Studio":
        render_music_page()
    elif selection == "Book Writing Studio":
        render_book_page()
    elif selection == "Bible Study Generator":
        render_bible_study_page()
    elif selection == "AI Bible Search":
        render_bible_search()
    elif selection == "Newsroom HQ":
        render_news_page()
    elif selection == "Viral Video Studio":
        render_viral_video_page()
    elif selection == "Audio Suite":
        render_audio_suite_page()
    elif selection == "Single Video Studio":
        single_render_viral_video_page()
    elif selection == "AI Tutor (Grades 1-12)":
      render_tutor_page()

    elif selection == "University AI Professor":
        render_university_tutor_page()
    elif selection == "AI Chef Studio":
        render_chef_page()
    elif selection == "AI Language Academy":
        render_language_academy_page()
    elif selection == "AI Podcast Studio":
        render_podcast_studio_page()



if __name__ == "__main__":
    main()
