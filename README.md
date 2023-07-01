# 🗣 ⇢ _`TalkSee`_  ⇢ 👀


## Software Design Document (SDD)

Video Demo: <URL HERE> ...

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://talksee.streamlit.app/)

___
## _`Table o'Contents`_

- [Introduction](#introduction)
- [System Overview](#system-overview)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Features](#features)
- [Functionality](#functionality)
- [Future Enhancements](#future-enhancements)

___

## [Introduction](#table-ocontents)

🗣 ⇢ _`TalkSee`_  ⇢ 👀 is a `speech-to-text` application that allows users to transcribe audio files or microphone input using the `WhisperAI ASR` models.

___

## [System Overview](#table-ocontents)

### _Model Selection_

> Provides a GUI to to select a `WhisperAI model`.

### _Audio Input_

> Supports two modes of audio input: `microphone` input and `file` upload.

### _Speech Recognition/Transcription_

> Employs a `WhisperAI ASR` model to transcribe the user audio input into text.

### _Text Output_

> Displays the transcribed text to the user.

___

## [Dependencies](#table-ocontents)

The 🗣 ⇢ _`TalkSee`_  ⇢ 👀 web app relies on the following external libraries and resources:

- [Python 3.x](https://www.python.org/downloads/)

- [python-dotenv](https://github.com/theskumar/python-dotenv): Provides environment variables.

- [os](https://docs.python.org/3/library/os.html): Provides operating system interface.

- [time](https://docs.python.org/3/library/time.html): Provides time functionality.

- [io](https://docs.python.org/3/library/io.html): Provides input/output functionality.

___

- [Streamlit](https://streamlit.io/): Provides the user interface framework;

- [audio_recorder_streamlit](https://pypi.org/project/audio-recorder-streamlit/): Provides the audio input stream;

- [PyTorch](https://pytorch.org/docs/stable/torch.html): Provides the neural network library for GPU processing;

- [WhisperAI ASR](https://github.com/openai/whisper): Provides the speech recognition functionality;

___

## [Installation](#table-ocontents)

1. Clone the repository:

```sh
gh repo clone PedroZappa/TalkSee
```

2. Change the current directory to the cloned repository:

```sh
cd TalkSee
```

3. Install the `required packages` from the requirements.txt file:

```sh
pip install -r requirements.txt
```

4. Create a `.streamlit/secrets.toml` file in the project directory and add the `MODELS_PATH` variable:

```sh
touch .streamlit/secrets.toml | echo 'MODELS_PATH="models"' >> .streamlit/secrets.toml
```

5. Run Streamlit application:

```sh
streamlit run main.py
```

___

## [Features](#table-ocontents)

- `Streamlit`-based user interface for easy interaction.

- Select `WhisperAI ASR` model from the list of available models:

___

 |  Size  | Parameters | Multilingual model | Required VRAM | Relative speed |
 |:------:|:----------:|:------------------:|:-------------:|:--------------:|
 |  tiny  |    39 M    |       `tiny`       |     ~1 GB     |      ~32x      |
 |  base  |    74 M    |       `base`       |     ~1 GB     |      ~16x      |
 | small  |   244 M    |      `small`       |     ~2 GB     |      ~6x       |
 | medium |   769 M    |      `medium`      |     ~5 GB     |      ~2x       |
 | large  |   1550 M   |      `large`       |    ~10 GB     |       1x       |

- Checks if `CUDA` is available for `GPU` processing, else runs on `CPU`.

- Support for both `microphone input` and audio `file upload`.

- Display of the transcribed text to the user.

___

## [Functionality](#table-ocontents)

1. Select `WhisperAI ASR` model from the available options.
2. Choose an `input mode` (Mic or File).
    - If using the `Mic`, click the "microphone-icon" button to start recording audio. The recording will stop automatically after 2 seconds of silence.
    - If using `File`, upload an audio file in `.wav`, `.mp3` or `.m4a` formats.
3. Click the `Transcribe` button to transcribe the audio file.
4. Display transcribed text in "`Transcription`" section.

___

## [Future Enhancements](#table-ocontents)

Some possible future enhancements for 🗣 ⇢ _`TalkSee`_  ⇢ 👀 include:

- Support for `additional speech recognition models`.

- Support for `mobile devices`.

- `Real-time transcription` of live audio input.

- Improved error handling and user feedback.

- Integration with cloud storage services for seamless file upload and storage.

- `Generate an image` with the transcribed text as a prompt.

___

___

[BACK TO TOP](#top)


