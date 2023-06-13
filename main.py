from dotenv import load_dotenv
import streamlit as st
import whisper 
import time
import os
import torch

# Load environment variables from .env file
load_dotenv()

# Setup Model Storage
models_dir = os.getenv("MODELS_DIR")


def main():
    # Streamlit UI: Title
    st.title("TalkSee : 🗣 ⇢ 👀")

    # Check if CUDA is available
    torch.cuda.is_available()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.text(f"Torch running on {DEVICE}")
    
    # Get user input
    ## Select Input Mode
    input_type = st.sidebar.radio(
        'Select Input Mode',
        ('Mic', 'File')
    )
    
    if input_type == 'Mic':
        ## Record Live Audio
        st.write('Input Mode: Mic')
        ...
    else:
        ## Upload Audio file w/ Streamlit
        audio_file = st.file_uploader(
            "Upload Audio File", 
            # Supported file types
            type=["wav", "mp3", "m4a"]
        )
    
    
    # Load WhisperAI model
    ## Select model size
    whisper_selected = st.sidebar.selectbox(
        'Available Multilingual Models',
        ('tiny', 'base', 'small', 'medium', 'large', 'large-v2'),
        help="""
            |  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
            |:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
            |  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
            |  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
            | small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~6x       |
            | medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
            | large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |
        """ 
    ) 
    
    ## Check if select model exists in models directory
    # model_file = os.path.join(models_dir, whisper_selected")
    # if not os.path.exists(model_file):
    #     st.warning(f"Model {whisper_selected} not found in {models_dir}. Downloading...")
    
    ## Load user selected model
    # model = whisper.load_model(
    #     whisper_selected,
    #     download_root=models_dir
    # )
    # st.text(f"Whisper {whisper_selected} model loaded")
    
    
    # Transcribe audio file
    
    # Print results
    ...


# Run
if __name__ == "__main__":
    main()
    