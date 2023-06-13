import streamlit as st
import whisper 


def main():
    # Streamlit UI: Title
    st.title("TalkSee : 🗣 ⇢ 👀")

    
    # Get user input
    ## Record Live Audio
    
    ## Upload Audio file w/ Streamlit
    audio_file = st.file_uploader(
        "Upload Audio File", 
        type=["wav", "mp3", "m4a"]
    )
    
    
    # Load WhisperAI model
    ## Select model size
    whisper_selected = st.sidebar.selectbox(
        'Available Multilingual Models',
        ('tiny', 'base', 'small', 'medium', 'large', 'tiny.en', 'base.en', 'small.en', 'medium.en'),
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
    ## 
    
    
    # Transcribe audio file
    
    # Print results
    
    ...


# Run
if __name__ == "__main__":
    main()
    