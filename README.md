# Visual Assistant for the Visually Impaired

## 🌟 **Empowering the Visually Impaired with Open-Source AI Technology** 🌟

Welcome to the **Visual Assistant** project! This open-source application leverages advanced AI technologies to help visually impaired individuals better understand and interact with their surroundings. By combining computer vision, speech recognition, and text-to-speech capabilities, this tool provides real-time insights and assistance, enhancing independence and quality of life.

## 🚀 **Features**

- **Real-Time Video Recording**: Capture video from your webcam to monitor and analyze the environment.
- **Audio Recording and Transcription**: Record audio inputs and transcribe them into text using Whisper AI.
- **Image Processing with Moondream**: Extract the best frame from recorded videos and generate descriptive captions.
- **Text-to-Speech (TTS)**: Convert AI-generated responses into audible speech using Facebook's MMS-TTS.
- **User-Friendly GUI**: Interactive interface built with Tkinter for easy navigation and control.
- **Keyboard Shortcuts**: Control recording and processing using simple keyboard inputs.
- **Open-Source & Collaborative**: Fully open-source with a GitHub repository encouraging community contributions.

## 🛠 **Technology Stack**

- **Programming Language**: Python
- **Libraries & Frameworks**:
  - [OpenCV](https://opencv.org/) for video capture and processing
  - [NumPy](https://numpy.org/) for numerical operations
  - [Pillow](https://python-pillow.org/) for image handling
  - [SoundDevice](https://python-sounddevice.readthedocs.io/) & [SoundFile](https://pysoundfile.readthedocs.io/) for audio recording
  - [Torch](https://pytorch.org/) for deep learning models
  - [Moondream](https://github.com/your-repo/moondream) for image captioning and querying
  - [Transformers](https://huggingface.co/transformers/) for Whisper and TTS models
  - [Scipy](https://www.scipy.org/) for audio file handling
  - [Pynput](https://pynput.readthedocs.io/) for keyboard event handling
  - [Tkinter](https://docs.python.org/3/library/tkinter.html) for GUI

## 📥 **Installation**

### Prerequisites

- **Python 3.7 or higher**: Ensure Python is installed on your system. You can download it from [here](https://www.python.org/downloads/).
- **Git**: To clone the repository. Download from [here](https://git-scm.com/downloads).

### Clone the Repository

```bash
git clone https://github.com/Shayan5422/visual-assistant.git
cd visual-assistant
```

### Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

Ensure you have `pip` installed. Then, install the required Python packages:

```bash
pip install -r requirements.txt
```

**Note**: If `moondream` is not available on PyPI, ensure to install it from the provided GitHub repository:

```bash
pip install git+https://github.com/your-vikhyat/moondream.git
```

### Download Pre-trained Models

1. **Moondream Model**: Download the `moondream-0_5b-int8.mf` model and place it in the designated directory (e.g., `/path/to/models/`).

2. **Whisper Model**: The application will automatically download the `openai/whisper-tiny.en` model when first run.

3. **TTS Model**: The application will also download the `facebook/mms-tts-eng` model upon initialization.

## 🏃 **Usage**

### Running the Application

#### Command-Line Interface (CLI)

To run the CLI-based `VideoTextProcessor`:

```bash
python video_text_processor.py
```

**Instructions**:

- **Spacebar**: Press and hold the spacebar to start recording audio.
- **Release Spacebar**: Stop recording audio, capture video, process the best frame, transcribe audio, generate a response, convert it to speech, and play the audio.
- **Escape Key**: Press the `Esc` key to exit the program.

#### Graphical User Interface (GUI)

To launch the Tkinter-based `VideoRecorderApp`:

```bash
python video_recorder_app.py
```

**Instructions**:

- **Start Recording**: Click the "Start Recording" button to begin recording video and audio.
- **Stop Recording**: Click the "Stop Recording" button to end recording and process the captured data.
- **Quit**: Click the "Quit" button to exit the application.

### Directory Structure

- `videos/`: Stores recorded video files.
- `audio/`: Stores recorded audio files and TTS outputs.
- `best_frame.jpg`: Stores the best frame extracted from the video for processing.

## 🤝 **Contributing**

Contributions are welcome! Whether you're a developer, designer, or accessibility advocate, your input can help enhance this tool.

### How to Contribute

1. **Fork the Repository**: Click the "Fork" button on GitHub to create your own copy.
2. **Create a Branch**: `git checkout -b feature/YourFeature`
3. **Commit Changes**: `git commit -m "Add your feature"`
4. **Push to Branch**: `git push origin feature/YourFeature`
5. **Open a Pull Request**: Describe your changes and submit for review.

### Reporting Issues

If you encounter any bugs or have feature requests, please open an issue [here](https://github.com/Shayan5422/visual-assistant/issues).

## 📄 **License**

This project is licensed under the [MIT License](LICENSE).

## 📧 **Contact**

For any inquiries or support, please contact [shayan.hashemi27@gmail.com](mailto:shayan.hashemi27@gmail.com).

## 🌐 **Acknowledgements**

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition.
- [Facebook MMS-TTS](https://github.com/facebookresearch/mms-tts) for text-to-speech.
- [Moondream](https://github.com/vikhyat/moondream) for image processing and captioning.

---

*Thank you for using the Visual Assistant! Together, we can make the world more accessible for everyone.*
