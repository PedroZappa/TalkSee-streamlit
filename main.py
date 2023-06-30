import logging
import logging.handlers
import sys, os, time, io, queue
from io import BytesIO
from pathlib import Path

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer, ClientSettings
from twilio.rest import Client
import pydub
import torch
import whisper

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

# Setup models storage path
models_path = st.secrets["MODELS_PATH"]

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
        # Render UI
        st.text(f"✅ Torch Status: {DEVICE}")
        alert = st.text(f"✅ Model Loaded: {st.session_state.whisper_selected}")
        feedback_transcribing = st.empty()
        transcription_success = st.empty()
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
                #  Setup User Mic Input
                audio_data = setup_mic()
    
            else:
                #  Setup User File Input
                audio_data = setup_file(col2)
        

    # Setup UI
    transcription_placeholder = st.empty()
    
    with col1:
        if audio_data is not None and st.button('Transcribe', use_container_width=True):
            feedback_transcribing.info("✍️ Transcribing...")
            
            transcription = transcribe(audio_data, st.session_state.model)
            print("Transcribed!:", transcription["text"])
            
            # Render UI
            feedback_transcribing.empty()
            st.header("✍️ Transcription")
            transcription_placeholder.markdown(transcription["text"])
            transcription_success.success(
                    "Transcription Complete!",
                    icon="🤩"
                )
            time.sleep(3.5)
            transcription_success.empty()

    # Session State DEBUGGER
    with st.expander("Session State", expanded=False):
        st.session_state
        
    # main() end # 
    ##############


@st.cache_resource
def model_exists(whisper_selected, device, models_path, _col1, _col2):
    if not whisper_selected:
        st.warning(f"Select a model! ⏫", icon="🚨")     
    ## Check if select model exists in models directory
    else:
        if not os.path.exists(whisper_file):

            with _col1:
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


def setup_mic():
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
        # Open file from streamlit recorder
        with open("output.wav", "wb") as f:
            f.write(audio_bytes)

        # Create a BytesIO object
        recorded_file = BytesIO(audio_bytes)
        recorded_file.name = 'output.wav'
        recorded_file.type = 'audio/wav'
        recorded_file.id = len(recorded_file.getvalue()) if st.session_state.audio_file is not None else 0
        recorded_file.size = len(audio_bytes)
    
        # Update Session_State
        st.session_state.audio_file = recorded_file
        print("setup_mic() session_state.audio_file:", st.session_state.audio_file)
        
        if recorded_file:
            # Render Playback Audio File
            st.header("🎧 Recorded File")
            st.audio(st.session_state.audio_file)
        
    return st.session_state.audio_file if st.session_state.audio_file else None
  
  
# def setup_mic():
#     webrtc_ctx = webrtc_streamer(
#         key="sendonly-audio",
#         mode=WebRtcMode.SENDONLY,
#         audio_receiver_size=256,
#         rtc_configuration={"iceServers": get_ice_servers()},
#         media_stream_constraints={
#             "video": False, 
#             "audio": True
#         },
#     )
    
#     status_indicator = st.empty()
    
#     if not webrtc_ctx.state.playing:
#         return
    
#     status_indicator.write("Loading...")
#     text_output = st.empty()
#     # stream = None
    
#     # session_state audio_buffer
#     if "audio_buffer" not in st.session_state:
#         st.session_state["audio_buffer"] = pydub.AudioSegment.empty()
    
#     # Record Audio
#     while True:
#         sound_chunk = pydub.AudioSegment.empty()
        
#         if webrtc_ctx.audio_receiver:
#             try:
#                 audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
#             except queue.Empty:
#                 time.sleep(0.1)
#                 status_indicator.write("No frame arrived.")
#                 continue
            
#             status_indicator.write("Running... Talk and See!")
            
#             for audio_frame in audio_frames:
#                 sound = pydub.AudioSegment(
#                     data=audio_frame.to_ndarray().tobytes(),
#                     sample_width=audio_frame.format.bytes,
#                     frame_rate=audio_frame.sample_rate,
#                     channels=len(audio_frame.layout.channels),
#                 )
#                 sound_chunk += sound
    
#         else:
#             status_indicator.write("AudioReceiver is not set. Abort.")
#             break
        
#     audio_buffer = st.session_state["audio_buffer"]
#     print("audio_buffer", audio_buffer)
    
#     if not webrtc_ctx.state.playing and len(audio_buffer) > 0:
#         st.info("Writing wav to disk")
#         # Open file from streamlit recorder
#         with open("output.wav", "wb") as f:
#             f.write(audio_buffer)
#         # audio_buffer.export("temp.wav", format="wav")
#         # Reset
#         st.session_state["audio_buffer"] = pydub.AudioSegment.empty()
    
#     # Create a BytesIO object
#     recorded_file = BytesIO(audio_buffer)
#     recorded_file.name = 'output.wav'
#     recorded_file.type = 'audio/wav'
#     recorded_file.id = len(recorded_file.getvalue()) if st.session_state.audio_file is not None else 0
#     recorded_file.size = len(audio_buffer)

#     # Update Session_State
#     st.session_state.audio_file = recorded_file
#     print("setup_mic() session_state.audio_file:", st.session_state.audio_file)

#     if recorded_file:
#         # Render Playback Audio File
#         st.header("🎧 Recorded File")
#         st.audio(st.session_state.audio_file)

#     return st.session_state.audio_file if st.session_state.audio_file else None
  

def setup_file(col2):
    with col2:
        ## Upload Pre-Recorded Audio file
        uploaded_file = st.file_uploader(
            "Upload Audio File", 
            key="uploaded_audio_file",
            # Supported file types
            type=["wav", "mp3", "m4a"],
            label_visibility='collapsed'
        )
        print("Loading file...", uploaded_file)
        
        if uploaded_file:
            # Write the uploaded file to disk
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Update Session_State
            st.session_state.audio_file = uploaded_file
            print("setup_file() session_state.audio_file:", st.session_state.audio_file)
            
            # Render Playback Audio File
            st.header("🎧 Uploaded File")
            st.audio(st.session_state.audio_file)
                
    return st.session_state.audio_file if st.session_state.audio_file else None


@st.cache_data
def transcribe(audio_file, _model):
    transcription = {}
    print("Transcribing...", audio_file)
    
    if audio_file is not None:
        
            
        # Transcribe audio file
        transcription = _model.transcribe(audio_file.name)
        print("audio_file id: ", audio_file.id)
    else:
        print("Upload a valid audio file")
    
    return transcription



# This code is based on https://github.com/whitphx/streamlit-webrtc/blob/c1fe3c783c9e8042ce0c95d789e833233fd82e74/sample_utils/turn.py
@st.cache_data  # type: ignore
def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
    We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
    but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
    See https://github.com/whitphx/streamlit-webrtc/issues/1213
    """

    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token.ice_servers

# Run
if __name__ == "__main__":
    main()