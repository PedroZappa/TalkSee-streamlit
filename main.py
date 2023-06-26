from dotenv import load_dotenv
import os
import time
import io
from io import BytesIO
import tempfile
import threading

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from stqdm import stqdm
import torch
import whisper


# Load env variables from .env file
load_dotenv()
# Setup Model Storage
models_path = os.environ.get("MODELS_PATH")
# enable write permission on models_path
os.chmod(models_path, 0o775)

# Init vars
model_file = ''
whisper_file = ''
audio_file = None


def main():
    global audio_file
    
    # Initialize Session State        
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None
        
    if 'whisper_loaded' not in st.session_state:
        st.session_state.whisper_loaded = False
        
    if 'model' not in st.session_state:
        st.session_state.model = None
        
    if 'transcribe_flag' not in st.session_state:
        st.session_state.transcribe_flag = False
        
    # DEBUG Session State
    # print("⚠️ Session State:", st.session_state)
    
    # Check if CUDA is available
    torch.cuda.is_available()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup Audio Stream
    # p, stream = create_pyaudio_stream(FORMAT, CHANNELS, RATE, FRAMES_PER_BUFFER)    
    
    # Streamlit UI: Title
    st.title("🗣 ⇢ TalkSee ⇢ 👀")
    # UI Columns
    col1, col2 = st.columns(2)
    
    # Select WhisperAI model
    model = None
    with col1:
        st.header("Select Model")
        whisper_select = st.selectbox(
            'Available Multilingual Models',
            ('tiny', 'base', 'small', 'medium', 'large', 'large-v2'),
            help="""
                |  Size  | Parameters | Multilingual model | Required VRAM | Relative speed |
                |:------:|:----------:|:------------------:|:-------------:|:--------------:|
                |  tiny  |    39 M    |       `tiny`       |     ~1 GB     |      ~32x      |
                |  base  |    74 M    |       `base`       |     ~1 GB     |      ~16x      |
                | small  |   244 M    |      `small`       |     ~2 GB     |      ~6x       |
                | medium |   769 M    |      `medium`      |     ~5 GB     |      ~2x       |
                | large  |   1550 M   |      `large`       |    ~10 GB     |       1x       |
            """,
            label_visibility='visible'
        )
    ## Get models path
    whisper_file = os.path.join(models_path, f"{whisper_select}.pt")
    whisper_selected = None
    
    # Get model (if not already loaded)
    if st.session_state.model is None or st.session_state.model != whisper_select:
        st.session_state.model, whisper_selected = model_exists(whisper_select, DEVICE, models_path, col1, col2)
        
    with col1:
        st.text(f"✅ Torch Status: {DEVICE}")
        alert = st.text(f"✅ Model Loaded: {whisper_selected}")
    
    with col1:
        st.divider()
    
    # Get user input
    ## Select Input Mode
    with col2:
        st.header("Select Input Mode")
        input_type = st.radio(
            'Select Input Mode',
            ('Mic', 'File'),
            label_visibility='collapsed',
            horizontal=True
        )     
        st.empty() 
            
    # Get User Input
    with col2:
        ## MIC or FILE
        if input_type == 'Mic':
            #  Render UI 🎙️
            # st.header("Record Audio")
            #  Setup User Mic Input
            audio_data = setup_mic(col1, col2) 
            
        else:
            #  Render UI
            # st.header("📂 Upload Audio")
            #  Setup User File Input
            audio_data = setup_file(col1, col2)
            

    # Transcribe audio file
    if audio_data is not None and st.session_state.transcribe_flag:
        
        # Init transcription thread
        transcription_thread = threading.Thread(
            target=transcribe,
            args=(audio_data, st.session_state.model, col1, col2)
        )
        transcription_thread.start()
        
        # Use stqdm progress bar to show transcription progress
        with stqdm(total=100, desc="Transcribing", bar_format="{l_bar}{bar} [ETA: {remaining}]", ncols=100) as progress_bar:
            for percent in range(100):
                if not transcription_thread.is_alive():
                    # If thread is running
                    progress_bar.update(100 - progress_bar.n)  
                    # Complete the progress bar if the model is downloaded
                    break
                progress_bar.update(1)
                time.sleep(0.1)
                
        transcription_thread.join()
        transcription = transcribe(audio_data, st.session_state.model, col1, col2)
        
        # Render UI
        st.header("✍️ Transcription")
        st.markdown(transcription["text"])
        
        st.session_state.transcribe_flag = False # Reset the flag

    # Session State DEBUGGER
    with st.expander("Session State", expanded=False):
        st.session_state
    
    # main() end # 
    ##############


def model_exists(whisper_selected, device, models_path, col1, col2):
    if not whisper_selected:
        st.warning(f"Select a model! ⏫", icon="🚨")     
    
    else:
        ## Check if select model exists in models directory
        if not os.path.exists(whisper_file):

            with col1:
                download_info = st.info(f"Downloading Whisper {whisper_selected} model...")
                
                if whisper_selected:
                    # Init thread for download model
                    download_thread = threading.Thread(
                        target=download_model, 
                        args=(whisper_selected, device, models_path)
                    )
                    download_thread.start()
                    
                    # Use stqdm progress bar to show download  progress
                    with stqdm(total=100, desc="Downloading", bar_format="{l_bar}{bar} [ETA: {remaining}]", ncols=100) as progress_bar:
                        for percent in range(100):
                            if not download_thread.is_alive():
                                # If thread is running
                                progress_bar.update(100 - progress_bar.n)  
                                # Complete the progress bar if the model is downloaded
                                break
                            progress_bar.update(1)
                            time.sleep(0.1)

                    # Wait for the download thread to finish and get the model
                    download_thread.join()
                    model = download_model(whisper_selected, device, models_path)
                        
                    # show loaded model if selected
                    if model:
                        # Update Session State
                        st.session_state.whisper_loaded = True
                
                # Render UI
                download_info.empty()
                
        else:
            # If model exists setup model object
            model = download_model(whisper_selected, device, models_path)
            print("Model Already DLd: ", model)
    
    return model, whisper_selected


def download_model(whisper_selected, device, models_path):
    model = whisper.load_model(
        whisper_selected,
        device=device,
        download_root=models_path
    )
    return model


# Handle User Input 
def setup_mic(col1, col2):
    global audio_file
    audio_data = None
        
    # Init Streamlit Audio Recorder
    audio_bytes = audio_recorder(
        text='',
        recording_color="#a34bff",
        neutral_color="#000",
        icon_name="microphone-lines",
        icon_size='7x',
        pause_threshold=2.0, 
        sample_rate=41_000
    )
    
    # if Recorder is clicked
    if audio_bytes:  
        frames = []

        # Render UI
        print("Recording...")
        rec_feedback = st.info("Recording...", icon="🔴")
    
        st.spinner('Wait for it...')
            # time.sleep(5)
            
        # Render UI
        rec_feedback.empty()
        # progress.empty()
        # Clear the progress bar when it's full
        # progress_placeholder.empty()
        
        
        # Read the binary content of the file
        with open('output.wav', 'rb') as f:
            file_content = f.read()
            
        # Open file from streamlit recorder
        with open("output.wav", "wb") as f:
            f.write(audio_bytes)
        # st.audio(audio_bytes, format="audio/wav")

        # # Create a BytesIO object
        uploaded_file = BytesIO(audio_bytes)
        uploaded_file.name = 'output.wav'
        uploaded_file.type = 'audio/wav'
        uploaded_file.id = len(uploaded_file.getvalue()) if st.session_state.audio_file is not None else 0
        uploaded_file.size = len(audio_bytes)

        # # Convert BytesIO object to a file-like object
        file_like_object = io.BytesIO(uploaded_file.getvalue())

        # # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file.flush()

        #     # Load Recorded file to memory
            audio_data = whisper.load_audio(temp_file.name)
            audio_data = whisper.pad_or_trim(audio_data) 

        # # Clean up temporary file
        os.unlink(temp_file.name)
        
        # # Update Session_State
        st.session_state.audio_file = uploaded_file
        # Signal for transcription
        st.session_state.transcribe_flag = True
        
        if audio_data.size > 0:
            # Render Playback Audio File
            st.header("🎧 Recorded File")
            st.audio(uploaded_file)

    return st.session_state.audio_file if st.session_state.audio_file else None
    
    
def setup_file(col1, col2):
    global audio_file
    
    with col2:
        ## Upload Pre-Recorded Audio file
        audio_file = st.file_uploader(
            "Upload Audio File", 
            key="audio_file",
            # Supported file types
            type=["wav", "mp3", "m4a"],
            label_visibility='collapsed'
        )
        
        # Signal for transcription
        st.session_state.transcribe_flag = True
        
        if audio_file:
            # Render Playback Audio File
            st.header("🎧 Uploaded File")
            st.audio(audio_file)
                
    return audio_file


# Transcribe Audio
def transcribe(audio_file, model, col1, col2):
    transcription = {}
    
    # if st.button("Transcribe!"):
    if audio_file is not None:
        #  Render UI
        # feedback = st.info("Transcribing...")
        # audio_file.name == filePath
        transcription = model.transcribe(audio_file.name)
        print("audio_file id: ", audio_file.id)
        
        with col1:
            st.success(
                    "Transcription Complete!",
                    icon="🤩"
                )
        print("Transcribed!:", transcription["text"])
    else:
        st.error("Please input a valid audio file!")

    return transcription
    

# Run
if __name__ == "__main__":
    main()