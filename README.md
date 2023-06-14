# 🗣 ⇢ _`TalkSee`_  ⇢ 👀 : a `speech-2-text` web app

## Software Design Document (SDD)

Video Demo: <URL HERE> ...

___

## _`Table o'Contents`_

- [Introduction](#introduction)
- [System Overview](#system-overview)
- [System Architecture](#system-architecture)
- [Data Flow](#data-flow)
- [User Interaction](#user-interaction)
- [Dependencies](#dependencies)
- [Development Environment](#development-environment)
- [Future Enhancements](#future-enhancements)
- [TalkSee](#talksee)

___

## [Introduction](#table-ocontents)

The 🗣 ⇢ _`TalkSee`_  ⇢ 👀 `speech-to-text` web app is designed to `transcribe audio files` or `live audio input` into text.

It utilizes the `WhisperAI` model for `speech recognition` and provides a `user-friendly interface` using `Streamlit`.

___

## [System Overview](#table-ocontents)

The 🗣 ⇢ _`TalkSee`_  ⇢ 👀 web app consists of the following components:

### User Interface

> Provides a graphical interface for users to interact with the web app.

### Audio Input

> Supports two modes of audio input: microphone input and file upload.

### Speech Recognition

> Uses the WhisperAI model to transcribe the audio input into text.

### Text Output

> Displays the transcribed text to the user.

___

## [System Architecture](#table-ocontents)

The system architecture of the 🗣 ⇢ _`TalkSee`_  ⇢ 👀 web app can be divided into the following modules:

### [User Interface Module](#table-ocontents)

#### Dependencies

- `Streamlit`, `pyaudio`

#### Responsibilities

- Provides a user-friendly interface for users to interact with the web app.

#### Functionalities

- Displays title and available models.

- Allows users to select input mode (microphone or file upload).

- Handles user input and displays the transcribed text to the user.

___

### [Audio Input Module](#table-ocontents)

#### Dependencies

- `pyaudio`, `wave`

#### Responsibilities

- Handles audio input from either the microphone or uploaded audio files.

#### Functionalities

- Creates an audio stream for recording or reading audio input.

- Records audio from the microphone and saves it to a file.

- Reads audio data from uploaded audio files.

___

### [Speech Recognition Module](#table-ocontents)

#### Dependencies

- `WhisperAI` model, `torch`

#### Responsibilities

- Transcribes the audio input into text using the WhisperAI model.

#### Functionalities

- Loads the selected WhisperAI model.

- Transcribes the audio input using the loaded model.

- Returns the transcribed text.

### [Text Output Module](#table-ocontents)

#### Dependencies

- None

#### Responsibilities

- Displays the transcribed text to the user.

#### Functionalities

- Receives the transcribed text from the Speech Recognition module.

- Displays the transcribed text to the user in a formatted manner.

___

## [Data Flow](#table-ocontents)

The data flow within the 🗣 ⇢ _`TalkSee`_  ⇢ 👀 web app is as follows:

- The User Interface module captures user input, including the selected model and audio input mode.

- The Audio Input module handles the audio input based on the selected mode, either recording from the microphone or reading from an uploaded audio file.

- The Speech Recognition module loads the selected WhisperAI model and transcribes the audio input into text.

- The Text Output module receives the transcribed text and displays it to the user.

___

## [User Interaction](#table-ocontents)

The 🗣 ⇢ _`TalkSee`_  ⇢ 👀 web app provides the following user interactions:

Selection of the `WhisperAI` model from the list of available options:

___ 
 |  Size  | Parameters | Multilingual model | Required VRAM | Relative speed |
 |:------:|:----------:|:------------------:|:-------------:|:--------------:|
 |  tiny  |    39 M    |       `tiny`       |     ~1 GB     |      ~32x      |
 |  base  |    74 M    |       `base`       |     ~1 GB     |      ~16x      |
 | small  |   244 M    |      `small`       |     ~2 GB     |      ~6x       |
 | medium |   769 M    |      `medium`      |     ~5 GB     |      ~2x       |
 | large  |   1550 M   |      `large`       |    ~10 GB     |       1x       |

___

- Selection of the audio input mode (`microphone` or `file upload`).

- Recording of audio from the microphone.

- Uploading of audio files for transcription.

- Display of the transcribed text to the user.

___

## [Dependencies](#table-ocontents)

The 🗣 ⇢ _`TalkSee`_  ⇢ 👀 web app relies on the following external libraries and resources:

- [Streamlit](https://streamlit.io/): Provides the user interface framework.

- [WhisperAI](https://github.com/openai/whisper) model: Provides the speech recognition functionality.

- [pyaudio](https://pypi.org/project/PyAudio/): Handles audio input from the microphone.

- [torch](https://pytorch.org/docs/stable/torch.html): Supports model loading and inference.

___

## [Development Environment](#table-ocontents)

- The 🗣 ⇢ _`TalkSee`_  ⇢ 👀 web app was developed using Python 3.9. 

- The required libraries and dependencies are specified in the code through import statements and package requirements.

___

## [Future Enhancements](#table-ocontents)

Some possible future enhancements for the 🗣 ⇢ _`TalkSee`_  ⇢ 👀 program include:

- Support for `additional speech recognition models`.

- `Real-time transcription` of live audio input.

- Improved error handling and user feedback.

- Integration with cloud storage services for seamless file upload and storage.

___

## 🗣 ⇢ [TalkSee](#table-ocontents) ⇢ 👀

🗣 ⇢ _`TalkSee`_  ⇢ 👀 a speech-to-text web app that provides an easy-to-use interface for transcribing audio files or live audio input into text.

___

[END](#top)