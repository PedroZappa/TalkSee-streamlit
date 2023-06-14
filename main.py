from dotenv import load_dotenv
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import whisper 
import time
import os
import torch
import pyaudio
import wave
from tqdm.auto import tqdm
import numpy as np
from io import BytesIO
import streamlit.components.v1 as components
from st_custom_components import st_audiorec


# Load env variables from .env file
load_dotenv()

# Setup Model Storage
models_path = os.environ.get("MODELS_PATH")

# enable write permission on models_path
os.chmod(models_path, 0o775)

# AUDIO CONSTANTS
FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Init vars
model_file = ''
whisper_file = ''
audio_file = None

# Initialize Session State
if 'stop_rec' not in st.session_state:
    st.session_state.stop_rec = False
    
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
    
if 'whisper_loaded' not in st.session_state:
    st.session_state.whisper_loaded = False

# Change Session State
def stop_rec():
    # On button clicked
    st.session_state.stop_rec = True

# DEBUG Session State
print("Session State: ", st.session_state)
print("stop_rec: ", st.session_state.stop_rec)
print("audio_file: ", st.session_state.audio_file)


def main():
    global audio_file
    
    
    # Check if CUDA is available
    torch.cuda.is_available()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup Audio Stream
    p, stream = create_pyaudio_stream(FORMAT, CHANNELS, RATE, FRAMES_PER_BUFFER)    
    
    # Streamlit UI: Title
    st.sidebar.title("🗣 ⇢  👀: a speech-to-text web app")
    st.title("🗣 ⇢ TalkSee ⇢ 👀")
    # st.info("a speech-to-text web app")

    
    # Select WhisperAI model
    model = None
    whisper_selected = st.sidebar.selectbox(
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
        """
    )
    ## Get models path
    whisper_file = os.path.join(models_path, f"{whisper_selected}.pt")
    print(whisper_file)
    
    ## Check if selected model exists
    model = model_exists(whisper_selected, DEVICE, models_path)
    
    
    # Get user input
    ## Select Input Mode
    st.sidebar.header("Select Input Mode")
    input_type = st.sidebar.radio(
        'Select Input Mode',
        ('Mic', 'File'),
        label_visibility='collapsed',
        horizontal=True
    )          
    ## MIC or FILE
    if input_type == 'Mic':
        #  Render UI
        st.sidebar.header("Record Audio")
        #  Setup User Mic Input
        audio_file = setup_mic(p, stream, RATE, CHANNELS, FORMAT, FRAMES_PER_BUFFER)     
    else:
        #  Setup User File Input
        audio_file = setup_file()

    # Render UI       
    st.sidebar.header("Record Audio")

    # Transcribe audio file
    transcription = transcribe(audio_file, model)
    

    # Generate Image ( Extra GOAL )

    
    # Session State DEBUGGER
    "st.sesion_state obj:", st.session_state
    
    # Render Torch Status
    st.sidebar.text(f"Torch Status: {DEVICE}")
    
    # main() end #
    ##############


# Setup Audio Stream 
def create_pyaudio_stream(format, channels, rate, frames_per_buffer):
    ## Create PyAudio
    p = pyaudio.PyAudio()
    ## Init Stream
    stream = p.open(
        format=format, 
        channels=channels, 
        rate=rate, 
        input=True, 
        frames_per_buffer=frames_per_buffer
    )
    return p, stream


def model_exists(whisper_selected, device, models_path):
    if not whisper_selected:
        st.sidebar.warning(f"Select a model! ⏫", icon="🚨")     
    
    else:
        whisper_select = st.sidebar.info(f"Selected Whisper Model: {whisper_selected}", icon="👆")
        
        ## Check if select model exists in models directory
        if not os.path.exists(whisper_file):

            download_info = st.info(f"Downloading Whisper {whisper_selected} model...")
            # whisper_progress = st.progress(0, text=progress_text)
            
            # Load Model
            model = load_whisper(whisper_selected, device, models_path, whisper_select)
            
            # Render UI
            download_info.empty()
                
            # Progress Update
            # for percent in tqdm():
            #     time.sleep(0.1)
        # time.sleep(3) # Wait for 3 seconds
        # alert.empty() # Clear the alert
    
    # model = load_whisper(whisper_selected, device, models_path, whisper_select)
    
    return model


# Load Whisper Models
# @st.cache_resource() 
def load_whisper(whisper_selected, device, models_path, whisper_select):
    ## Load user selected model
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
            alert = st.text(f"✅ Loaded Model: {whisper_selected}")
            whisper_select.empty() 
        
        return model


# Handle User Input 
## if MIC
def setup_mic(p, stream, rate, channels, format, frames_per_buffer):
    global audio_file
    # sEtup streamlit_audio_recorder stream
    # wav_audio_data = st_audiorec()
    
    # # Start Audio Recording 
    # if wav_audio_data is not None:
    #     # display audio data as received on the backend
    #     audio_file = st.audio(wav_audio_data, format='audio/wav')
        
    # if button clicked
    if st.sidebar.button("Record", key='record_btn'):        
        rec_frames = record_audio(stream, rate, frames_per_buffer) 
        # Save Recording to a file
        audio_file = save_audio(p, channels, format, rate, rec_frames) 
        
        if audio_file.size > 0:
            # Playback Audio File
            st.sidebar.header("Play Recorded Audio File")
            st.sidebar.audio(
                audio_file,
                format="audio/wav",
                sample_rate=RATE,
            )  
            
    print(audio_file)
            
    return audio_file 
    
## if FILE
def setup_file():
    global audio_file
    
    ## Upload Pre-Recorded Audio file
    audio_file = st.file_uploader(
        "Upload Audio File", 
        key="upload_file",
        # Supported file types
        type=["wav", "mp3", "m4a"]
    )
    if audio_file:
        # Render Playback Audio File
        st.sidebar.header("Play Uploaded Audio File")
        st.sidebar.audio(audio_file)
        
    print(audio_file)
        
    return audio_file


def record_audio(stream, rate, frames_per_buffer):
    # Time to record
    seconds = 6
    # Audio frames buffer
    frames = []
    print("Recording...")
    # Render UI
    feedback = st.info("Recording...")
    
    # Record Audio input
    for i in range(int(rate / frames_per_buffer * seconds)):
        data = stream.read(
            frames_per_buffer, 
            exception_on_overflow=False
        )
        frames.append(data)  
        
        # Check if the "Stop" button has been clicked
        if st.session_state.stop_rec:
            break
        
    # Reset stop_rec for future recordings
    st.session_state.stop_rec = False
        
    # Check if recording is done
    if not st.session_state.stop_rec:
        print("Recording Finished!")
    else:
        print("Recording Stopped")
        
    feedback.empty()
    
    return frames 


def save_audio(p, channels, format, rate, frames):
    # Save Recorded Audio to file
    with wave.open("output.wav", "wb") as file:
        file.setnchannels(channels)
        file.setsampwidth(p.get_sample_size(format))
        file.setframerate(rate)
        # combine all elements in frames list into a binary string
        frames_bytes = b"".join(frames)
        file.writeframes(frames_bytes)
        
    # Convert frames to NumPy array
    audio_arr = np.frombuffer(frames_bytes, dtype=np.int16)
        
    print(f"Inside save_audio: {file}" )
    
    # Store audio_arr in session_state
    st.session_state.audio_file = audio_arr
    print("audio_file: ", st.session_state.audio_file)
    
    
    return audio_arr
    

def load_audio_file(input) :
    # Open audio file
    with wave.open('output.wav', 'rb') as file:
        # Audio Length
        audio_data = file.getnframes() / file.getframerate()
        print(f"Loaded audio file length: {audio_data}")
        
        # Read Audio Frames
        frames = file.readframes(-1)
        print("Types of frames object:", type(frames), type(frames[0]))
        
        # Convert frames to NumPy array
        audio_array = np.frombuffer(frames, dtype=np.int16)
        
    return audio_array


# Transcribe Audio
def transcribe(audio_file, model):
    transcription = {}
    if st.sidebar.button("Transcribe!"):
        if audio_file is not None:
            #  Render UI
            feedback = st.sidebar.info("Transcribing...")
            # audio_file.name == filePath
            transcription = model.transcribe(audio_file.name)
            st.sidebar.success(
                "Transcription Complete!",
                icon="🤩"
            )
            # Render UI
            st.header("✍️ Transcription 📃")
            st.markdown(transcription["text"])
            feedback.empty()
        else:
            st.sidebar.error("Please input a valid audio file!")
    
    # print("Transcription:", transcription['text'])
            
    return transcription
    

# Run
if __name__ == "__main__":
    main()