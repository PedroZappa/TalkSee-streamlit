# 🗣 ⇢ _`TalkSee`_  ⇢ 👀 : a `speech-2-text`

## Software Design Document (SDD)

Video Demo: <URL HERE> ...

___

## _`Table o'Contents`_

- [Introduction](#introduction)
- [System Overview](#system-overview)
- [Dependencies](#dependencies)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
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

> Uses the WhisperAI model to transcribe the user audio input into text.

### _Text Output_

> Displays the transcribed text to the user.

___

## [Dependencies](#table-ocontents)

The 🗣 ⇢ _`TalkSee`_  ⇢ 👀 web app relies on the following external libraries and resources:

- [Python 3.x](https://www.python.org/downloads/)

- [python-dotenv](https://github.com/theskumar/python-dotenv): Provides environment variables.

- [Streamlit](https://streamlit.io/): Provides the user interface framework;

- [stqdm](https://pypi.org/project/stqdm/): Provides progress bar display for streamlit apps;

- [PyTorch](https://pytorch.org/docs/stable/torch.html): Supports model loading and inference;

- [WhisperAI ASR](https://github.com/openai/whisper): Provides the speech recognition functionality;

- [PyAudio](https://pypi.org/project/PyAudio/): Handles audio input stream from the microphone;

- [Wave](https://docs.python.org/3/library/wave.html): Write audio input to file;

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

- Support for both `microphone input` and audio `file upload`.

- Real-time transcription progress display using `stqdm`.

- Display of the transcribed text to the user.

___

## [Installation](#table-ocontents)

1. Clone the repository:

```sh
git clone https://github.com/pedrozappa/talksee.git
```

2. Change the current directory to the cloned repository:

```sh
cd talksee
```

3. Install the `required packages` from the requirements.txt file:

```sh
pip install -r requirements.txt
```

4. Create a `.env` file in the project directory and add the `MODELS_PATH` variable:

```sh
touch .env | echo 'MODELS_PATH=/path/to/whisper/models' >> .env
```

5. Run Streamlit application:

```sh
streamlit run app.py
```

___

## [Usage](#table-ocontents)

1. Select `WhisperAI ASR` model from the available options.
2. Choose an `input mode` (Mic or File).
    - If using the Mic, click the "Record" button to start recording audio. Click "Stop" when you're done.
    - If using File, upload an audio file in WAV format.
3. Click the "Transcribe" button to transcribe the audio.
4. View the transcribed text in the "Transcription" section.

___

## [Future Enhancements](#table-ocontents)

Some possible future enhancements for 🗣 ⇢ _`TalkSee`_  ⇢ 👀 include:

- Support for `additional speech recognition models`.

- `Real-time transcription` of live audio input.

- Improved error handling and user feedback.

- Integration with cloud storage services for seamless file upload and storage.

- `Generate an image` with the transcribed text as a prompt.

___

___

[BACK TO TOP](#top)

