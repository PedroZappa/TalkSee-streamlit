from dotenv import load_dotenv
import streamlit as st
import time
import os
import torch
import pyaudio
import whisper 
import wave
from tqdm.auto import tqdm
from io import BytesIO
import io
import tempfile


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

class UploadedFile:
    def __init__(self, id, name, type, size):
        self.id = id
        self.name = name
        self.type = type
        self.size = size
        

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
print("⚠️  Session State:", st.session_state)


def main():
    global audio_file
    
    # Check if CUDA is available
    torch.cuda.is_available()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Session State DEBUGGER
    st.session_state
    
    # Setup Audio Stream
    p, stream = create_pyaudio_stream(FORMAT, CHANNELS, RATE, FRAMES_PER_BUFFER)    
    
    # Streamlit UI: Title
    st.sidebar.title("🗣 ⇢  👀 : a speech-2-text web app")
    st.sidebar.text(f"Torch Status: {DEVICE}")
    st.title("🗣 ⇢ TalkSee ⇢ 👀")
    
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
    st.sidebar.header("Select I/O Mode")
    input_type = st.sidebar.radio(
        'Select Input Mode',
        ('Mic', 'File'),
        label_visibility='collapsed',
        horizontal=True
    )          
    
    ## MIC or FILE
    if input_type == 'Mic':
        #  Render UI
        st.sidebar.header("🎙️ Record Audio")
        #  Setup User Mic Input
        audio_data = setup_mic(p, stream, RATE, CHANNELS, FORMAT, FRAMES_PER_BUFFER)   
        
        # DEBUG
        print("🎙️ setup_mic: ", audio_data)
        print("🎙️ audio_data type: ", type(audio_data))
        print()
        
    else:
        #  Setup User File Input
        audio_data = setup_file()
        
        # DEBUG
        print("📂 setup_file:", audio_data)
        print("📂 audio_data type", type(audio_data))


    # Render UI       
    st.sidebar.header("✍️ Transcribe Audio")


    # Transcribe audio file
    transcription = transcribe(audio_data, model)
    

    # Generate Image ( Extra GOAL )

    
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
    audio_data = None
        
    # if button clicked
    if st.sidebar.button("Record", key='record_btn'):  
        seconds = 6
        frames = []

        # Render UI
        print("Recording...")
        rec_feedback = st.sidebar.info(
            "Recording...", 
            icon="🔴"
        )
        
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
            
            
        # Render UI
        rec_feedback.empty()

        # Save Recorded Audio to file
        output_file_path = "output.wav"
        with wave.open(output_file_path, "wb") as file:
            file.setnchannels(channels)
            file.setsampwidth(p.get_sample_size(format))
            file.setframerate(rate)
            # combine all elements in frames list into a binary string
            frames_bytes = b"".join(frames)
            file.writeframes(frames_bytes)

        # Store file in session_state
        # st.session_state.audio_file = output_file_path
        # print("Session State audio_file: ", st.session_state.audio_file)
        
        # print("audio_file_path inside setup_mic():", audio_file_path)
        
        # Read the binary content of the file
        with open('output.wav', 'rb') as f:
            file_content = f.read()

        # Create a BytesIO object
        uploaded_file = BytesIO(file_content)
        uploaded_file.name = 'output.wav'
        uploaded_file.type = 'audio/wav'
        
        # Convert BytesIO object to a file-like object
        file_like_object = io.BytesIO(uploaded_file.getvalue())

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file.flush()

            # Load Recorded file to memory
            audio_data = whisper.load_audio(temp_file.name)
            audio_data = whisper.pad_or_trim(audio_data) 

        # Clean up temporary file
        os.unlink(temp_file.name)
        
        st.session_state.audio_file = uploaded_file
        print("audio loaded: ", audio_data)

    return st.session_state.audio_file if st.session_state.audio_file else None
    
    
## if FILE
def setup_file():
    global audio_file
    
    ## Upload Pre-Recorded Audio file
    audio_file = st.file_uploader(
        "Upload Audio File", 
        key="audio_file",
        # Supported file types
        type=["wav", "mp3", "m4a"]
    )
        
    if audio_file:
        # Render Playback Audio File
        st.sidebar.header("🎧 Play Uploaded Audio File")
        st.sidebar.audio(audio_file)
                
    return audio_file


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
    # mel = whisper.log_mel_spectrogram(audio).to(model.device)

            
    return transcription
    

# Run
if __name__ == "__main__":
    main()