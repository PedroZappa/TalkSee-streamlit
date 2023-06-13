from dotenv import load_dotenv
import streamlit as st
import whisper 
import time
import os
import torch
from tqdm.auto import tqdm

# Load environment variables from .env file
load_dotenv()

# Setup Model Storage
# models_path = os.getenv("MODELS_DIR")
models_path = os.environ.get("MODELS_PATH")
# models_path = "models/"
os.chmod(models_path, 0o775)

model_file = ''

def main():
    # Streamlit UI: Title
    st.title("🗣 ⇢ TalkSee ⇢ 👀")
    # st.sidebar.title("TalkSee")
    st.sidebar.title("🗣 ⇢ 👀")

    # Check if CUDA is available
    torch.cuda.is_available()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.text(f"Torch Status: {DEVICE}")
    
    # Get user input
    ## Select Input Mode
    st.sidebar.header("Select Input Mode")
    input_type = st.sidebar.radio(
        'Select Input Mode',
        ('Mic', 'File'),
        label_visibility='collapsed',
        horizontal=True
    )
    
    if input_type == 'Mic':
        ## Record Live Audio
        st.write('Input Mode: Mic')
        #
        # Implement MIC input
        #
    else:
        ## Upload Audio file w/ Streamlit
        audio_file = st.file_uploader(
            "Upload Audio File", 
            # Supported file types
            type=["wav", "mp3", "m4a"]
        )
        if audio_file:
            # Playback Audio File
            st.sidebar.header("Play Uploaded Audio File")
            st.sidebar.audio(audio_file)
    
    # Load WhisperAI model
    ## Select model
    whisper_selected = st.sidebar.selectbox(
        'Available Multilingual Models',
        ('', 'tiny', 'base', 'small', 'medium', 'large', 'large-v2'),
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
    whisper_file = os.path.join(models_path, f"{whisper_selected}.pt")
    print(f"Whisper file:", whisper_file)
    
    # Check if selected model exists
    if whisper_selected:
        st.sidebar.success(f"Whisper Selected: {whisper_selected}", icon="✅")
        
        ## Check if select model exists in models directory
        print(f"Selected model: {model_file}")
        if not os.path.exists(whisper_file):
            st.warning(
                f"Model {whisper_selected} not found in {models_path}.",
                icon="🚨"
            )
            # progress_text = f"Downloading Whisper {whisper_selected} model..."
            # whisper_progress = st.progress(0, text=progress_text)
            
            # Load Model
            load_whisper(whisper_selected, DEVICE)
            
            # Progress Update
            # for percent in tqdm():
            #     time.sleep(0.1)
                
    else:
        st.sidebar.warning(f"Select a model! ⏫", icon="🚨")
    
    load_whisper(whisper_selected, DEVICE)
    
    # Transcribe audio file
    if st.sidebar.button("Transcribe!"):
        if audio_file is not None:
            st.sidebar.write("Transcribing...")
            # audio_file.name == filePath
            transcription = model.transcribe(audio_file.name)
            st.sidebar.success(
                "Transcription Complete!",
                icon="🤩"
            )
            st.markdown(transcription["text"])
        else:
            st.sidebar.error("Please input a valid audio file!")
    
    # Print results
    ...


def load_whisper(whisper_selected, DEVICE):
    ## Load user selected model
    if whisper_selected:
        model = whisper.load_model(
            whisper_selected,
            device=DEVICE,
            download_root=models_path
        )
           
        # show loaded model if selected
        if model_file:
            st.sidebar.text(f"Whisper {whisper_selected} model loaded")

# Run
if __name__ == "__main__":
    main()