import streamlit as st
import pvrecorder
import wave
import struct
import threading

# --- APPLICATION STATE ---
# We use Streamlit's session state to keep track of the app's state across reruns.
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_frames' not in st.session_state:
    st.session_state.audio_frames = []
if 'recorder_thread' not in st.session_state:
    st.session_state.recorder_thread = None
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = None
if 'audio_file_path' not in st.session_state:
    st.session_state.audio_file_path = None


# --- RECORDING LOGIC ---
def record_audio(device_index, frames_list, stop_event):
    """
    This function runs in a separate thread to perform the audio recording.
    It continuously reads audio frames from the selected device until the
    stop_event is set.
    """
    recorder = None
    try:
        # Initialize the recorder
        recorder = pvrecorder.PvRecorder(device_index=device_index, frame_length=512)
        recorder.start()

        # Recording loop
        while not stop_event.is_set():
            frame = recorder.read()
            frames_list.extend(frame)

    except Exception as e:
        # We can't use st.error here because this isn't the main Streamlit thread.
        # Instead, we print the error to the console for debugging.
        print(f"Error during recording: {e}")
    finally:
        if recorder:
            recorder.stop()
            recorder.delete()

def render_Live_Audio():
    # --- STREAMLIT UI ---
    st.set_page_config(layout="wide", page_title="Audio Recorder")

    st.title("🎙️ Live Audio Recorder")
    st.markdown(
        "Select your microphone, click **Start Recording**, and when you're done, "
        "click **Stop Recording**. The captured audio will appear below and play automatically."
    )
    st.markdown("---")

    # --- DEVICE SELECTION ---
    try:
        # Get a list of available audio input devices
        devices = pvrecorder.PvRecorder.get_available_devices()
        device_options = {name: i for i, name in enumerate(devices)}

        st.subheader("1. Select your microphone")
        selected_device_name = st.selectbox(
            "Available microphones:",
            options=device_options.keys(),
            label_visibility="collapsed"
        )
        selected_device_index = device_options[selected_device_name]
    except Exception as e:
        st.error("Could not get audio devices. Please ensure you have a microphone connected and have granted permissions.")
        st.error(f"Details: {e}")
        st.stop()

    # --- AMPLIFICATION CONTROL ---
    st.subheader("2. Post-Recording Volume Gain")
    gain = st.slider(
        "Select an amplification factor. This will be applied to the audio after recording.",
        min_value=1.0,
        max_value=5.0,
        value=1.0,
        step=0.1
    )

    # --- CONTROL BUTTONS ---
    st.subheader("3. Control Recording")
    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        start_button = st.button("Start Recording", key="start", type="primary", disabled=st.session_state.is_recording,
                                 use_container_width=True)

    with col2:
        stop_button = st.button("Stop Recording", key="stop", disabled=not st.session_state.is_recording,
                                use_container_width=True)

    if st.session_state.is_recording:
        st.info("🔴 Recording in progress... Click 'Stop Recording' to finish.")

    # --- BUTTON ACTIONS ---
    if start_button:
        # Set the recording state
        st.session_state.is_recording = True
        st.session_state.audio_frames = []
        st.session_state.audio_file_path = None
        st.session_state.stop_event = threading.Event()

        # Create and start the recording thread
        thread = threading.Thread(
            target=record_audio,
            args=(selected_device_index, st.session_state.audio_frames, st.session_state.stop_event)
        )
        st.session_state.recorder_thread = thread
        thread.start()

        st.toast("Recording started!", icon="🎤")
        st.rerun()

    if stop_button:
        # Signal the recording thread to stop
        st.session_state.stop_event.set()
        # Wait for the thread to complete its execution
        st.session_state.recorder_thread.join()
        st.session_state.is_recording = False

        st.toast("Recording stopped!", icon="✅")

        # --- AUDIO PROCESSING ---
        if st.session_state.audio_frames:
            with st.spinner("Amplifying and saving audio..."):
                audio_path = 'user_recording.wav'

                # Amplify audio frames using the selected gain.
                # We must clamp the values to the 16-bit integer range [-32768, 32767]
                # to prevent audio clipping and distortion.
                amplified_frames = [
                    int(min(max(frame * gain, -32768), 32767))
                    for frame in st.session_state.audio_frames
                ]

                # Pack the recorded frames into a wave file
                try:
                    audio_data = struct.pack('h' * len(amplified_frames), *amplified_frames)

                    with wave.open(audio_path, 'w') as f:
                        f.setnchannels(1)  # Mono audio
                        f.setsampwidth(2)  # 16-bit PCM
                        f.setframerate(16000)  # 16kHz sample rate
                        f.writeframes(audio_data)

                    st.session_state.audio_file_path = audio_path
                except Exception as e:
                    st.error(f"Failed to save audio: {e}")
        else:
            st.warning("No audio was recorded.")

        st.rerun()

    # --- AUDIO PLAYBACK ---
    if st.session_state.audio_file_path:
        st.subheader("4. Play Your Recording")
        st.audio(st.session_state.audio_file_path, format='audio/wav', autoplay=True)

    st.markdown("---")
    st.markdown("Powered by **pvrecorder** and **Streamlit**.")

