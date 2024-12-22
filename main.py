import cv2
import numpy as np
import moondream as md
from PIL import Image
from transformers import VitsModel, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import scipy.io.wavfile
from pynput import keyboard
import threading
import sounddevice as sd
import soundfile as sf
import time
import os
import subprocess
import sys
import atexit

# Constants
AUDIO_FILENAME = "recorded_audio.wav"
VIDEO_FILENAME_TEMPLATE = "recorded_video_{}.avi"
BEST_FRAME_PATH = "best_frame.jpg"
TTS_OUTPUT_TEMPLATE = "TTS_output_{}.wav"
FPS = 10
FRAMES_PER_VIDEO = 10

class ResourceManager:
    def __init__(self):
        self.moondream_model = None
        self.whisper_model = None
        self.whisper_processor = None
        self.tts_model = None
        self.tts_tokenizer = None
        self.pipe_recognizer = None
        self.cap = None
        self.model_lock = threading.Lock()
        
    def initialize_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam")
            
    def initialize_moondream(self, model_path):
        with self.model_lock:
            if self.moondream_model is None:
                self.moondream_model = md.vl(model=model_path)
                
    def initialize_whisper(self):
        if self.whisper_model is None:
            device = "cuda" if torch.cuda.is_available() else "mps"
            model_id = "openai/whisper-tiny.en"
            self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True, 
                use_safetensors=True
            )
            self.whisper_model.to(device)
            self.whisper_processor = AutoProcessor.from_pretrained(model_id)
            self.pipe_recognizer = pipeline(
                "automatic-speech-recognition",
                model=self.whisper_model,
                tokenizer=self.whisper_processor.tokenizer,
                feature_extractor=self.whisper_processor.feature_extractor,
                device=0 if device == "cuda" else -1,
            )
            
    def initialize_tts(self):
        if self.tts_model is None:
            self.tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
            self.tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
            
    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Resources cleaned up")

class VideoTextProcessor:
    def __init__(self, model_path):
        self.resource_manager = ResourceManager()
        self.MODEL_PATH = model_path
        self.record_count = 0
        self.record_lock = threading.Lock()
        self.space_pressed = False
        self.audio_thread = None
        self.stop_event = threading.Event()
        
        # Register cleanup
        atexit.register(self.resource_manager.cleanup)

    def extract_best_frame(self, video_path, output_path):
        """Extract the best frame from video based on Laplacian variance."""
        try:
            cap_temp = cv2.VideoCapture(video_path)
            best_frame = None
            max_score = -1

            while cap_temp.isOpened():
                ret, frame = cap_temp.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian_var > max_score:
                    max_score = laplacian_var
                    best_frame = frame

            cap_temp.release()

            if best_frame is not None:
                cv2.imwrite(output_path, best_frame)
                print(f"Best frame saved to {output_path}")
            else:
                print("No frame found.")
        except Exception as e:
            print(f"Error extracting best frame: {e}")

    def record_audio(self, stop_event, filename):
        """Record audio using sounddevice."""
        try:
            fs = 16000  # Sample rate
            print("Recording audio...")
            recording = []

            def callback(indata, frames, time, status):
                if stop_event.is_set():
                    raise sd.CallbackStop()
                if status:
                    print(f"Recording status: {status}")
                recording.append(indata.copy())

            with sd.InputStream(samplerate=fs, channels=1, callback=callback):
                while not stop_event.is_set():
                    sd.sleep(100)

            if recording:
                audio_data = np.concatenate(recording, axis=0)
                sf.write(filename, audio_data, fs)
                print(f"Audio recorded to {filename}")
            else:
                print("No audio data was recorded.")
        except Exception as e:
            print(f"Error during audio recording: {e}")

    def record_video(self, frames_per_video=FRAMES_PER_VIDEO, output_dir='videos', filename='sample', fps=FPS):
        """Record video from webcam."""
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            self.resource_manager.initialize_camera()
            frame_width = int(self.resource_manager.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.resource_manager.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_path = os.path.join(output_dir, f"{filename}.avi")
            out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
            
            print(f"Recording {filename}.")

            frame_count = 0
            while frame_count < frames_per_video:
                ret, frame = self.resource_manager.cap.read()
                if ret:
                    out.write(frame)
                    frame_count += 1
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Recording stopped by user.")
                        break
                else:
                    print("Failed to grab frame.")
                    break
            
            out.release()
            print(f"Recording finished. Video saved as {video_path}")
            return video_path
        except Exception as e:
            print(f"Error recording video: {e}")
            return None

    def process_with_moondream(self, image_path, text):
        """Process image and text with Moondream model."""
        try:
            self.resource_manager.initialize_moondream(self.MODEL_PATH)
            image = Image.open(image_path)
            encoded_image = self.resource_manager.moondream_model.encode_image(image)
            caption = self.resource_manager.moondream_model.caption(encoded_image)["caption"]
            print("Caption:", caption)
            answer = self.resource_manager.moondream_model.query(encoded_image, text)["answer"]
            print("Answer:", answer)
            return answer
        except Exception as e:
            print(f"Error in Moondream processing: {e}")
            return None

    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper model."""
        try:
            self.resource_manager.initialize_whisper()
            result = self.resource_manager.pipe_recognizer(audio_path)
            print("Transcribed Text:", result["text"])
            return result["text"]
        except Exception as e:
            print(f"Error in audio transcription: {e}")
            return None

    def text_to_speech(self, text, output_path):
        """Convert text to speech using Facebook MMS-TTS."""
        try:
            self.resource_manager.initialize_tts()
            inputs = self.resource_manager.tts_tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                output = self.resource_manager.tts_model(**inputs).waveform
            
            waveform = output.squeeze().cpu().numpy()
            if waveform.max() > 1.0 or waveform.min() < -1.0:
                waveform = waveform / max(abs(waveform.max()), abs(waveform.min()))
            
            sampling_rate = int(self.resource_manager.tts_model.config.sampling_rate)
            scipy.io.wavfile.write(output_path, rate=sampling_rate, data=waveform)
            print(f"TTS output saved to {output_path}")
        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")

    def play_audio(self, file_path):
        """Play audio file using system default player."""
        try:
            if sys.platform.startswith('darwin'):
                subprocess.call(['open', file_path])
            elif os.name == 'nt':
                os.startfile(file_path)
            elif os.name == 'posix':
                subprocess.call(['xdg-open', file_path])
            else:
                print(f"Cannot play audio on this OS: {sys.platform}")
        except Exception as e:
            print(f"Error playing audio: {e}")

    def on_press(self, key):
        """Handle keyboard press events."""
        try:
            if key == keyboard.Key.space and not self.space_pressed:
                self.space_pressed = True
                print("Door opened!")
                self.stop_event.clear()
                self.audio_thread = threading.Thread(
                    target=self.record_audio, 
                    args=(self.stop_event, AUDIO_FILENAME)
                )
                self.audio_thread.start()
        except AttributeError:
            pass

    def on_release(self, key):
        """Handle keyboard release events."""
        if key == keyboard.Key.space and self.space_pressed:
            self.space_pressed = False
            self.stop_event.set()
            if self.audio_thread is not None:
                self.audio_thread.join()
            print("Stop recording audio")
            
            with self.record_lock:
                self.record_count += 1
                current_count = self.record_count
            
            # Record video
            filename = f"recorded_video_{current_count}"
            video_path = self.record_video(
                frames_per_video=FRAMES_PER_VIDEO, 
                output_dir='videos', 
                filename=filename, 
                fps=FPS
            )

            if video_path:
                # Extract best frame
                self.extract_best_frame(video_path, BEST_FRAME_PATH)

                # Transcribe audio
                if os.path.exists(AUDIO_FILENAME):
                    text = self.transcribe_audio(AUDIO_FILENAME)
                else:
                    text = ""
                    print("Audio file not found. Skipping transcription.")

                # Process with Moondream
                if os.path.exists(BEST_FRAME_PATH) and text:
                    answer = self.process_with_moondream(BEST_FRAME_PATH, text)
                else:
                    answer = "No valid input for Moondream processing."

                # Convert to speech
                if answer:
                    tts_output = f"TTS_output_{current_count}.wav"
                    self.text_to_speech(answer, tts_output)
                
                    # Play generated audio
                    if os.path.exists(tts_output):
                        self.play_audio(tts_output)
                    else:
                        print("TTS output file not found.")

            print("Ready for the next command.")

        # Exit on Escape key
        if key == keyboard.Key.esc:
            print("Escape key pressed. Exiting program.")
            return False

def main():
    try:
        processor = VideoTextProcessor("/Users/shayanhashemi/Downloads/Video_to_text/moondream-0_5b-int8.mf")
        print("Press and hold the Spacebar to open the door and start recording audio.")
        print("Release the Spacebar to stop recording audio, capture video, and process.")
        print("Press 'Esc' key at any time to exit the program.")

        with keyboard.Listener(
            on_press=processor.on_press, 
            on_release=processor.on_release
        ) as listener:
            listener.join()
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        processor.resource_manager.cleanup()

if __name__ == "__main__":
    main()              