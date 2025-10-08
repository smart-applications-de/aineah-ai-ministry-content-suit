import streamlit as st
import os
import tempfile
import base64
import io
from io import BytesIO
import wave  # NEW: Required to handle raw PCM audio data from TTS model

# Import for custom component functionality
import streamlit.components.v1 as components

# Imports for Gemini (Using the new SDK structure from user's prompt)
from google import genai
from google.genai import types


# --- HELPER FUNCTION FOR TTS AUDIO CONVERSION ---
# The TTS API returns raw PCM audio data, which needs to be wrapped in a WAV header
# to be played by standard media players (like Streamlit's st.audio).
def pcm_to_wav_bytes(pcm_data, channels=1, sample_width=2, rate=24000):
    """Converts raw PCM audio data into a WAV file format in memory."""
    buffer = BytesIO()
    try:
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm_data)
        buffer.seek(0)
        return buffer
    except Exception as e:
        # Handle potential errors during WAV creation
        print(f"Error creating WAV file: {e}")
        return None


# --- CUSTOM COMPONENT DEFINITION (No Change Needed Here) ---

def gencomponent(name, template="", script=""):
    """Generates a Streamlit custom component from HTML/CSS/JS strings."""

    def html():
        # Enhanced CSS for aesthetics, centralization, and visual feedback
        return f"""
            <!DOCTYPE html>
            <html lang="en">
                <head>
                    <meta charset="UTF-8" />
                    <title>{name}</title>
                    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css" crossorigin="anonymous" referrerpolicy="no-referrer" />
                    <style>
                        body {{
                            background-color: transparent;
                            margin: 0;
                            padding: 20px 0;
                            display: flex;
                            justify-content: center; /* Center the button horizontally */
                            align-items: center;
                        }}
                        #toggleBtn {{
                            width: 80px;
                            height: 80px;
                            border-radius: 50%; /* Circular button */
                            border: 4px solid #F0F2F6;
                            cursor: pointer;
                            color: white;
                            font-size: 28px; /* Larger icon */
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            background: linear-gradient(145deg, #1f78b4, #005a96); /* Blue gradient */
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
                            transition: all 0.2s ease-in-out;
                        }}
                        #toggleBtn:hover {{
                            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15), 0 3px 6px rgba(0, 0, 0, 0.1);
                        }}
                        #toggleBtn.recording {{
                            background: linear-gradient(145deg, #FF6B6B, #E53935); /* Red gradient for recording */
                            transform: scale(1.05);
                            animation: pulse 1.5s infinite;
                        }}
                        @keyframes pulse {{
                            0% {{ box-shadow: 0 0 0 0 rgba(229, 57, 53, 0.7); }}
                            70% {{ box-shadow: 0 0 0 10px rgba(229, 57, 53, 0); }}
                            100% {{ box-shadow: 0 0 0 0 rgba(229, 57, 53, 0); }}
                        }}
                    </style>
                    <script>
                        function sendMessageToStreamlitClient(type, data) {{
                            const outData = Object.assign({{
                                isStreamlitMessage: true,
                                type: type,
                            }}, data);
                            window.parent.postMessage(outData, "*");
                        }}

                        const Streamlit = {{
                            setComponentReady: function() {{
                                sendMessageToStreamlitClient("streamlit:componentReady", {{apiVersion: 1}});
                            }},
                            setFrameHeight: function(height) {{
                                sendMessageToStreamlitClient("streamlit:setFrameHeight", {{height: height}});
                            }},
                            setComponentValue: function(value) {{
                                sendMessageToStreamlitClient("streamlit:setComponentValue", {{value: value}});
                            }},
                            RENDER_EVENT: "streamlit:render",
                            events: {{
                                addEventListener: function(type, callback) {{
                                    window.addEventListener("message", function(event) {{
                                        if (event.data.type === type) {{
                                            event.detail = event.data
                                            callback(event);
                                        }}
                                    }});
                                }}
                            }}
                        }}
                    </script>
                </head>
                <body>
                    {template}
                </body>
                <script src="https://unpkg.com/hark@1.2.0/hark.bundle.js"></script>
                <script>
                    {script}
                </script>
            </html>
        """

    dir = f"{tempfile.gettempdir()}/{name}"
    if not os.path.isdir(dir): os.mkdir(dir)
    fname = f'{dir}/index.html'
    f = open(fname, 'w')
    f.write(html())
    f.close()
    func = components.declare_component(name, path=str(dir))

    def f(**params):
        component_value = func(**params)
        return component_value

    return f


template = """
    <button id="toggleBtn"><i class="fa-solid fa-microphone fa-lg" ></i></button>
"""

script = """
    let mediaStream = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let speechEvents = null;
    let silenceTimeout = null;
    let isRecording = false;
    const toggleBtn = document.getElementById('toggleBtn');
    const microphoneIcon = toggleBtn.querySelector('i');

    Streamlit.setComponentReady();
    Streamlit.setFrameHeight(120); // Increased height to accommodate centered button and padding

    function blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                // Extracts the Base64 part after the comma
                const base64String = reader.result.split(',')[1]; 
                resolve(base64String);
            };
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    }

    async function handleRecordingStopped() {
        // Output audio format is 'audio/webm' as provided by MediaRecorder
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' }); 
        const base64Data = await blobToBase64(audioBlob);

        Streamlit.setComponentValue({
            audioData: base64Data,
            status: 'stopped'
        });
    }

    function onRender(event) {
        const args = event.detail.args;
        window.harkConfig = {
            interval: args.interval || 50,
            threshold: args.threshold || -60,
            play: args.play !== undefined ? args.play : false,
            silenceTimeout: args.silenceTimeout || 1500
        };
    }

    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);

    toggleBtn.addEventListener('click', async () => {
        if (!isRecording) {
            // --- START RECORDING ---
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(mediaStream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = () => {
                    // This handles stopping either manually (button click) or automatically (silence)
                    handleRecordingStopped().catch(err => {
                        console.error('Error handling recording:', err);
                        Streamlit.setComponentValue({
                            error: 'Failed to process recording'
                        });
                    });
                };

                // Initialize Hark for silence detection
                speechEvents = hark(mediaStream, {
                    interval: window.harkConfig.interval,
                    threshold: window.harkConfig.threshold,
                    play: window.harkConfig.play
                });

                // Simplified silence detection: stop the recording on prolonged silence
                speechEvents.on('stopped_speaking', () => {
                    // Set timeout to stop recording after silenceTimeout duration
                    silenceTimeout = setTimeout(() => {
                        if (mediaRecorder && mediaRecorder.state === 'recording') {
                            // Automatically stop recording after silence
                            mediaRecorder.stop();
                        }
                    }, window.harkConfig.silenceTimeout);
                });

                speechEvents.on('speaking', () => {
                    // Clear the timeout if speech resumes
                    if (silenceTimeout) {
                        clearTimeout(silenceTimeout);
                        silenceTimeout = null;
                    }
                });

                mediaRecorder.start();
                isRecording = true;
                toggleBtn.classList.add('recording');
                microphoneIcon.classList.remove('fa-microphone');
                microphoneIcon.classList.add('fa-stop'); // Change icon to stop

                Streamlit.setComponentValue({ status: 'recording' }); // Inform Streamlit that recording started

            } catch (err) {
                console.error('Error accessing microphone:', err);
                Streamlit.setComponentValue({
                    error: err.message
                });
            }
        } else {
            // --- STOP RECORDING ---
            isRecording = false;
            toggleBtn.classList.remove('recording');
            microphoneIcon.classList.remove('fa-stop');
            microphoneIcon.classList.add('fa-microphone'); // Change icon back to microphone

            if (speechEvents) {
                speechEvents.stop();
                speechEvents = null;
            }

            if (silenceTimeout) {
                clearTimeout(silenceTimeout);
                silenceTimeout = null;
            }

            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop(); // This triggers mediaRecorder.onstop
            }

            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }
        }
    });
"""


def audio_recorder_with_silence(interval=50, threshold=-60, play=False, silenceTimeout=1500):
    """
    Custom Streamlit component for recording audio with silence detection.
    It returns a dictionary containing Base64 encoded WebM audio data upon stop.
    """
    component_func = gencomponent('configurable_audio_recorder', template=template, script=script)
    # The component function returns the value sent back by JS via setComponentValue
    return component_func(interval=interval, threshold=threshold, play=play, silenceTimeout=silenceTimeout)


# --- MAIN STREAMLIT APPLICATION LOGIC ---
def render_Live_Audio():
    # --- APPLICATION STATE ---
    if 'transcription' not in st.session_state:
        st.session_state.transcription = None
    if 'translation' not in st.session_state:  # New state for translation
        st.session_state.translation = None
    if 'tts_audio_buffer' not in st.session_state:  # NEW: State to hold the translated speech audio buffer
        st.session_state.tts_audio_buffer = None
    if 'audio_buffer' not in st.session_state:
        st.session_state.audio_buffer = None
    if 'recording_status' not in st.session_state:
        st.session_state.recording_status = 'ready'  # New state to track if component is waiting for input

    # --- STREAMLIT UI SETUP ---
    st.set_page_config(layout="wide", page_title="Audio Recorder & Transcriber")

    # --- SIDEBAR FOR API KEY and LANGUAGE SELECTION ---
    TARGET_LANGUAGES = ["English", "Spanish", "French", "German", "Japanese", "Chinese", "Hindi", "Russian","Romanian","Swahili","Italian","Indonesian","Turkish","Portuguese"]

    with st.sidebar:
        st.header("Configuration")
        st.markdown("You need a Google AI API key to use the transcription feature.")
        api_key = st.text_input(
            "Enter your Google AI API Key",
            type="password",
            help="Get your key from https://aistudio.google.com/app/apikey"
        )

        st.markdown("---")
        st.header("Translation Settings")
        selected_target_lang = st.selectbox(
            "Select Target Language for Translation",
            TARGET_LANGUAGES,
            index=0  # Default to English
        )

    st.title("🎙️ Live Audio Recorder, Transcriber & Translator")
    st.markdown(
        "Record audio in **any language**. The audio is transcribed in its source language and then automatically translated to your selected target language."
    )
    st.markdown("---")

    # --- 1. RECORD AUDIO ---
    st.subheader("1. Record Audio (Silence Detection)")
    st.markdown("Click the microphone to start. Click again to manually stop, or stop automatically after a pause.")

    # Call the custom component
    recorded_data = audio_recorder_with_silence(silenceTimeout=1500)

    # --- NEW: Check if recording has started to clear old state (for overwrite) ---
    if recorded_data and recorded_data.get('status') == 'recording':
        # Clear previous recording and transcription/translation data
        if st.session_state.audio_buffer is not None:
            st.session_state.audio_buffer = None
            st.session_state.transcription = None
            st.session_state.translation = None  # Clear translation state
            st.session_state.tts_audio_buffer = None  # NEW: Clear TTS audio buffer
            st.session_state.recording_status = 'recording'
            st.rerun()  # Rerun to clear the display/playback sections

    # --- 2. PROCESS THE RECORDING WHEN AVAILABLE (stopped) ---
    if recorded_data and recorded_data.get('status') == 'stopped':
        # This block executes when the custom component returns data
        if st.session_state.audio_buffer is None:
            with st.spinner("Processing WebM audio... Please wait."):
                base64_data = recorded_data.get('audioData')
                error = recorded_data.get('error')

                if error:
                    st.error(f"Recording Error: {error}")
                    st.session_state.recording_status = 'error'
                    st.rerun()

                elif base64_data:
                    try:
                        # Decode the Base64 WebM audio data
                        webm_bytes = base64.b64decode(base64_data)

                        # Store bytes in an in-memory buffer
                        buffer = BytesIO(webm_bytes)

                        st.session_state.audio_buffer = buffer
                        st.session_state.recording_status = 'finished'
                        st.rerun()  # Rerun to display the audio player and transcription button

                    except Exception as e:
                        st.error(f"Error processing audio: {e}")
                        st.session_state.recording_status = 'error'

                else:
                    st.warning("No audio was recorded.")
                    st.session_state.recording_status = 'ready'

    # --- 3. AUDIO PLAYBACK AND TRANSCRIPTION/TRANSLATION ---
    if st.session_state.audio_buffer:
        st.subheader("2. Review and Process Audio")

        st.info(
            "The audio is recorded in the browser-native **WebM** format. "
            "The file will be transcribed in the source language, translated, and then converted "
            f"to speech in **{selected_target_lang}**."
        )

        col1, col2 = st.columns(2)

        with col1:
           # st.markdown("##### Source Audio Playback")
            # Display the recorded audio (WebM format)
            st.session_state.audio_buffer.seek(0)
           # st.audio(st.session_state.audio_buffer, format='audio/webm')

        # with col2:
        #     st.markdown("##### Download")
        #     # Download button (Download as WebM for compatibility)
        #     st.session_state.audio_buffer.seek(0)
        #     st.download_button(
        #         label="⬇️ Download Recording (WebM)",
        #         data=st.session_state.audio_buffer.getvalue(),
        #         file_name="user_recording.webm",
        #         mime="audio/webm",
        #         use_container_width=True,
        #     )
        #
        st.markdown("---")

        # Transcription & Translation Button
        if st.button(f"Transcribe, Translate & Convert to Audio (to {selected_target_lang})", use_container_width=True,
                     type="primary"):
            if not api_key:
                st.error("🚨 Please enter your Google AI API Key in the sidebar to process the audio.")
            else:
                try:
                    client = genai.Client(api_key=api_key)

                    # --- STEP 1: TRANSCRIPTION (Source Language) ---
                    with st.spinner("Step 1/3: Transcribing audio in its original language..."):
                        st.session_state.audio_buffer.seek(0)
                        audio_bytes = st.session_state.audio_buffer.getvalue()

                        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type='audio/webm')

                        # Instruct the model to transcribe in whatever language it hears
                        transcribe_prompt = 'You are an expert audio-to-text model. Transcribe this WebM audio clip exactly as spoken, noting the source language.'

                        transcription_response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=[transcribe_prompt, audio_part]
                        )

                        st.session_state.transcription = transcription_response.text

                    # --- STEP 2: TRANSLATION (Target Language) ---
                    with st.spinner(f"Step 2/3: Translating to {selected_target_lang}..."):

                        translation_prompt = (
                            f"Translate the following transcribed text into the **{selected_target_lang}** language. "
                            f"Only output the translated text. Original text: \n\n{st.session_state.transcription}"
                        )

                        translation_response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=[translation_prompt]
                        )

                        st.session_state.translation = translation_response.text

                    # --- STEP 3: TEXT-TO-SPEECH (TTS) for Translation ---
                    with st.spinner(f"Step 3/3: Converting translated text to speech..."):

                        # Prompt focuses on the TTS delivery
                        tts_prompt = f"Say the following translated text in a clear, standard voice: {st.session_state.translation}"

                        tts_response = client.models.generate_content(
                            model="gemini-2.5-flash-preview-tts",
                            contents=tts_prompt,
                            config=types.GenerateContentConfig(
                                response_modalities=["AUDIO"],
                                speech_config=types.SpeechConfig(
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            # Using a standard, clear voice
                                            voice_name='Kore',
                                        )
                                    )
                                ),
                            )
                        )

                        # Extract the raw PCM audio data (base64 decoded)
                        tts_data_part = tts_response.candidates[0].content.parts[0].inline_data
                        #pcm_data = base64.b64decode(tts_data_part.data)

                        # Convert PCM to WAV bytes in memory using the helper function
                        wav_buffer = pcm_to_wav_bytes(tts_data_part.data,channels=1, sample_width=2, rate=24000)
                        st.session_state.tts_audio_buffer = wav_buffer

                        st.rerun()  # Rerun to display the final results

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    st.session_state.transcription = None
                    st.session_state.translation = None
                    st.session_state.tts_audio_buffer = None  # Clear new state on error

    # --- 4. TRANSCRIPTION & TRANSLATION RESULT ---
    if st.session_state.transcription:
        st.subheader("3. Processing Results")

        # Display Transcription
       # st.markdown("##### Original Transcription (Source Audio)")
       # st.text_area("Transcription", st.session_state.transcription, height=150, label_visibility="collapsed")

        # Display Translation
        if st.session_state.translation:
            #st.markdown(f"##### Translated Text (Target Language: {selected_target_lang})")
            #st.text_area("Translation", st.session_state.translation, height=150, label_visibility="collapsed")

            # NEW: Display TTS Audio Playback
            if st.session_state.tts_audio_buffer:
                st.markdown(f"##### Audio Playback (Translated)")
                wav_bytes = (st.session_state.get('tts_audio_buffer'))
                st.audio(wav_bytes, format='audio/wav',autoplay=True)
                #
                #st.audio(st.session_state.tts_audio_buffer, format='audio/wav')

    st.markdown("---")
    st.markdown("Powered by **Streamlit Custom Components**, **hark.js**, and **Google Gemini**.")
