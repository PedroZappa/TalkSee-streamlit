# _`TalkSee`_ : Software Development Documentation

Video Demo: <URL HERE> ...

___

## _`Table o'Contents`_

- [Description](#description)
- [Architecture](#architecture)
- [Dependencies](#dependencies)
- [Goals](#goals)
- [Components](#components)
  - [Front-end Web Interface](#front-end-web-interface)
  - [Streamlit Web Server](#streamlit-web-server)
  - [Speech-2-Text Model](#speech-2-text-model)
- [Functionality](#functionality)

___

## [Description](#table-ocontents)

`TalkSee` 🗣👀 : a `speech-2-text` web app.

___

## [Architecture](#table-ocontents)

The architecture of `TalkSee` will consist of:

- A `Streamlit front-end web interface` that allows users to `record audio`.

- A `Streamlit web server` to handle incoming audio files;

- The `OpenAI Whisper ASR`  ( _`Automatic Speech Recognition`_ ) model to transcribe the audio files to text.

** ( `Extra Feature`: Upgrade to `speech-2-image` by implementing a diffusion model )

___

## [Dependencies](#table-ocontents)

_`TalkSee`_ relies on:

- [Streamlit](https://streamlit.io/) `open-source framework` for building the `web app`;

- [`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc) package for `handling real-time audio stream` over the network;

- [Whisper](https://github.com/openai/whisper) for converting `speech` to `text`;

___

## [Goals](#table-ocontents)

- Build a simple responsive interface with `Streamlit`;

- Structure the app with proper separation of concerns;

- Learn about `speech recognition`;

___

## [Components](#table-ocontents)

### [_`Front-end Web Interface`_](#table-ocontents)

- The front-end web interface will be built using `Streamlit`;

- Will get `live` or `pre-recorded audio` from the user using the `streamlit-webrtc` package ;

- The text will be rendered in the body of the app.

### [_`Streamlit Web Server`_](#table-ocontents)

- The `Streamlit web server` will handle `incoming audio files`;

- Process them using the `Whisper ASR model`;

- The server will be set up with the following route:
  - `POST /transcribe`:
    - Receives the `recorded audio file` from the front-end and `converts it to text` using the `Whisper ASR model`.

### [Speech-2-Text Model](#table-ocontents)

- The `OpenAI Whisper ASR model` will be used to convert the recorded audio files to text.
  - It is a powerful and `open-source model` that can perform `multilingual speech recognition`, `speech translation`, and `language identification`

___

## [Functionality](#table-ocontents)

The main functionality of `TalkSee` is to `receive live audio input` of human speech, `transcribe` it, and `output text`.

- `TalkSee` will be able to:
  - Record audio using the user's microphone.
  - Transcribe the recorded audio to text using the `OpenAI Whisper ASR model`.
  - Display the transcribed text in the front-end web interface.

___