import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import torch
import moondream as md
from transformers import VitsModel, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import scipy.io.wavfile
import subprocess
import sys

class VideoRecorderApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Video & Audio Recorder")
        self.model_path = model_path
        
        # Constants
        self.FPS = 10
        self.FRAMES_PER_VIDEO = 10
        
        # Initialize state variables
        self.recording = False
        self.camera_active = False
        self.audio_recording = False
        self.current_video_filename = None
        self.current_audio_filename = None
        
        # Create output directories
        os.makedirs('videos', exist_ok=True)
        os.makedirs('audio', exist_ok=True)
        
        self.setup_gui()
        self.initialize_camera()
        self.initialize_models()
        
    def setup_gui(self):
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video frame
        self.video_frame = ttk.Label(self.main_frame)
        self.video_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Control buttons
        self.record_button = ttk.Button(self.main_frame, text="Start Recording", 
                                      command=self.toggle_recording)
        self.record_button.grid(row=1, column=0, pady=5, padx=5)
        
        self.quit_button = ttk.Button(self.main_frame, text="Quit", 
                                    command=self.cleanup_and_quit)
        self.quit_button.grid(row=1, column=1, pady=5, padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.main_frame, textvariable=self.status_var)
        self.status_label.grid(row=2, column=0, columnspan=2, pady=5)
        
    def initialize_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.camera_active = True
        self.update_video_feed()
        
    def initialize_models(self):
        try:
            # Initialize Moondream
            self.moondream_model = md.vl(model=self.model_path)
            
            # Initialize Whisper
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
            
            # Initialize TTS
            self.tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
            self.tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
            
            self.status_var.set("Models initialized successfully")
        except Exception as e:
            self.status_var.set(f"Error initializing models: {str(e)}")
            
    def update_video_feed(self):
        if self.camera_active:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (854, 480))
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.video_frame.configure(image=photo)
                self.video_frame.image = photo
            self.root.after(10, self.update_video_feed)
            
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        self.recording = True
        self.record_button.configure(text="Stop Recording")
        self.status_var.set("Recording...")
        
        # Generate filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_video_filename = f"videos/video_{timestamp}.avi"
        self.current_audio_filename = f"audio/audio_{timestamp}.wav"
        
        # Start video recording
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_writer = cv2.VideoWriter(
            self.current_video_filename,
            cv2.VideoWriter_fourcc(*'XVID'),
            self.FPS,
            (frame_width, frame_height)
        )
        
        # Start audio recording
        self.audio_recording = True
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.start()
        
        # Start frame capture
        self.capture_frames()
        
    def record_audio(self):
        try:
            fs = 16000  # Sample rate
            recording = []
            
            def callback(indata, frames, time, status):
                if status:
                    print(f"Audio status: {status}")
                recording.append(indata.copy())
                
            with sd.InputStream(samplerate=fs, channels=1, callback=callback):
                while self.audio_recording:
                    sd.sleep(100)
                    
            if recording:
                audio_data = np.concatenate(recording, axis=0)
                sf.write(self.current_audio_filename, audio_data, fs)
                
        except Exception as e:
            self.status_var.set(f"Audio recording error: {str(e)}")
            
    def capture_frames(self):
        if self.recording:
            ret, frame = self.cap.read()
            if ret:
                self.video_writer.write(frame)
            self.root.after(int(1000/self.FPS), self.capture_frames)
            
    def stop_recording(self):
        self.recording = False
        self.audio_recording = False
        self.record_button.configure(text="Start Recording")
        
        # Wait for audio thread to complete
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()
            
        # Close video writer
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
            
        self.status_var.set("Processing recording...")
        
        # Process the recording in a separate thread
        processing_thread = threading.Thread(target=self.process_recording)
        processing_thread.start()
        
    def process_recording(self):
        try:
            # Extract best frame
            best_frame_path = "best_frame.jpg"
            self.extract_best_frame(self.current_video_filename, best_frame_path)
            
            # Transcribe audio
            if os.path.exists(self.current_audio_filename):
                text = self.transcribe_audio(self.current_audio_filename)
            else:
                text = ""
                self.status_var.set("Audio file not found")
                return
            
            # Process with Moondream
            if os.path.exists(best_frame_path) and text:
                answer = self.process_with_moondream(best_frame_path, text)
            else:
                answer = "No valid input for processing"
                self.status_var.set("Processing error: missing inputs")
                return
            
            # Generate speech
            if answer:
                tts_output = f"audio/tts_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                self.text_to_speech(answer, tts_output)
                self.play_audio(tts_output)
                
            self.status_var.set("Processing complete")
            
        except Exception as e:
            self.status_var.set(f"Processing error: {str(e)}")
            
    def extract_best_frame(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        best_frame = None
        max_score = -1
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var > max_score:
                max_score = laplacian_var
                best_frame = frame
                
        cap.release()
        
        if best_frame is not None:
            cv2.imwrite(output_path, best_frame)
            
    def transcribe_audio(self, audio_path):
        result = self.pipe_recognizer(audio_path)
        return result["text"]
        
    def process_with_moondream(self, image_path, text):
        image = Image.open(image_path)
        encoded_image = self.moondream_model.encode_image(image)
        answer = self.moondream_model.query(encoded_image, text)["answer"]
        return answer
        
    def text_to_speech(self, text, output_path):
        inputs = self.tts_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.tts_model(**inputs).waveform
        
        waveform = output.squeeze().cpu().numpy()
        if waveform.max() > 1.0 or waveform.min() < -1.0:
            waveform = waveform / max(abs(waveform.max()), abs(waveform.min()))
        
        sampling_rate = int(self.tts_model.config.sampling_rate)
        scipy.io.wavfile.write(output_path, rate=sampling_rate, data=waveform)
        
    def play_audio(self, file_path):
        try:
            if sys.platform.startswith('darwin'):
                subprocess.call(['open', file_path])
            elif os.name == 'nt':
                os.startfile(file_path)
            elif os.name == 'posix':
                subprocess.call(['xdg-open', file_path])
        except Exception as e:
            self.status_var.set(f"Error playing audio: {str(e)}")
            
    def cleanup_and_quit(self):
        self.camera_active = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.root.quit()
        
def main():
    root = tk.Tk()
    model_path = "/Users/shayanhashemi/Downloads/Video_to_text/moondream-0_5b-int8.mf"  # Update this path
    app = VideoRecorderApp(root, model_path)
    root.mainloop()

if __name__ == "__main__":
    main()