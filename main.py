from dotenv import load_dotenv
import sys
import os
import time
import io
from io import BytesIO
import tempfile

import streamlit as st
from streamlit.web import cli as stcli
from streamlit import runtime
from audio_recorder_streamlit import audio_recorder
import torch
import whisper


# Check if running on Streamlit Cloud or locally
if os.getenv('STREAMLIT_SHARING_MODE') == 's4a':
    print("Running on Streamlit Cloud")
    # Check if the directory exists
    if not os.path.exists(models_path):
        # Create directory in the current working directory
        dir_path = os.path.join(os.getcwd(), 'models')
        os.makedirs(dir_path, exist_ok=True)      
else:
    print("Running locally")
    # Load env variables from .env file
    load_dotenv()
    # Setup Model Storage
    models_path = os.environ.get("MODELS_PATH")
    # enable write permission on models_path
    os.chmod(models_path, 0o775)
    
# DEBUG
# Check for streamlit sharing mode
print(os.getenv('STREAMLIT_SHARING_MODE'))

# Init vars
model_file = ''
whisper_file = '' 
audio_file = None

# Initialize Session State        
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
    
if 'whisper_selected' not in st.session_state:
    st.session_state.whisper_selected = False
    
if 'whisper_loaded' not in st.session_state:
    st.session_state.whisper_loaded = False
    
if 'model' not in st.session_state:
    st.session_state.model = None
    

def main():
    audio_data = None
    transcription = dict()
    
    # Session State DEBUGGER
    with st.expander("Session State", expanded=False):
        st.session_state
    
    # Check if CUDA is available
    torch.cuda.is_available()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    if whisper_select != st.session_state.whisper_selected or st.session_state.whisper_loaded != True:
        st.session_state.model, st.session_state.whisper_selected = model_exists(whisper_select, DEVICE, models_path, col1, col2)
        
    with col1:
        st.text(f"✅ Torch Status: {DEVICE}")
        alert = st.text(f"✅ Model Loaded: {st.session_state.whisper_selected}")
        ui_success = st.empty()
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

    transcription_placeholder = st.empty()
    
    with col1:
        if audio_data and st.button('Transcribe', use_container_width=True):
            feedback_transcribing = st.info("✍️ Transcribing...")
            
            transcription = transcribe(audio_data, st.session_state.model, col1, col2)
            print("Transcribed!:", transcription["text"])
            
            # Render UI
            feedback_transcribing.empty()
            st.header("✍️ Transcription")
            transcription_placeholder.markdown(transcription["text"])
            ui_success.success(
                    "Transcription Complete!",
                    icon="🤩"
                )
            time.sleep(3.5)
            ui_success.empty()

    
    # main() end # 
    ##############


def model_exists(whisper_selected, device, models_path, col1, col2):
    if not whisper_selected:
        st.warning(f"Select a model! ⏫", icon="🚨")     
    ## Check if select model exists in models directory
    else:
        if not os.path.exists(whisper_file):

            with col1:
                download_info = st.spinner(f"Loading Whisper {whisper_selected} model...")
                
                if whisper_selected:
                    model = whisper.load_model(
                        whisper_selected,
                        device=device,
                        download_root=models_path
                    )
                        
                    # show loaded model if selected
                    if model:
                        # Update Session State
                        st.session_state.whisper_loaded = True
                
                    # Render UI
                    # download_info.empty()
    
    return model, whisper_selected


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

        # # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file.flush() 

            # Load Recorded file to memory
            audio_data = whisper.load_audio(temp_file.name)
            audio_data = whisper.pad_or_trim(audio_data) 

        # Clean up temporary file
        os.unlink(temp_file.name)
        
        # # Update Session_State
        st.session_state.audio_file = uploaded_file
        print("setup_mic() session_state.audio_file:", st.session_state.audio_file)
        
        if audio_data.size > 0:
            # Render Playback Audio File
            st.header("🎧 Recorded File")
            st.audio(st.session_state.audio_file)
        
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
        print("Loading file...")
        
        if audio_file:
            # Render Playback Audio File
            st.header("🎧 Uploaded File")
            st.audio(audio_file)
                
    return audio_file


def transcribe(audio_file, model, col1, col2):
    transcription = {}
    print("Transcribing...")
    
    if audio_file is not None:
        transcription = model.transcribe(audio_file.name)
        print("audio_file id: ", audio_file.id)
    
    return transcription


# Run
if __name__ == "__main__":
    main()