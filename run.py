# Real-time Chinese Speech-to-Text with Hotkeys and LLM Integration
#
# Description:
# This script captures audio from the microphone in real-time and transcribes it from Chinese.
# It uses the Silero VAD model for intelligent segmentation, ensuring that only complete
# utterances are sent to the transcription model. This significantly improves accuracy
# by avoiding cuts in the middle of words or sentences.
# All output files (.txt and .md) are saved in a subfolder named 'output'.
#
# Hotkey Controls:
# - Spacebar: Toggles pausing and resuming of the transcription. Resuming creates a new log file.
# - Enter: Sends the content of the most recent log file to the Gemini LLM for processing.
# - Up/Down Arrows: Increases/Decreases the VAD speech detection threshold.
# - Esc: Gracefully exits the application.
#
# It is optimized to run on an NVIDIA GPU using CUDA.
#
# Author: Gemini
# Date: 2025-10-07

# --- Installation ---
# Before running, you need to install the required Python packages.
# Use 'uv' (a fast package installer) in your terminal or command prompt.
#
# 1. First, install PyTorch with the specific CUDA version for your system (cu121 is recommended):
#    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#
# 2. Then, install the other required libraries:
#    uv pip install faster-whisper sounddevice numpy pynput python-dotenv requests
#
# NOTE: The Silero VAD model will be downloaded automatically by PyTorch on the first run.
#

# --- Imports ---
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading
import queue
import datetime
import os
import requests
import json
from pynput import keyboard
from dotenv import load_dotenv
import torch

# --- Global State Variables ---
is_running = threading.Event()  # Event to control the running state. Cleared (False) by default, meaning paused.
current_log_filename = None
exit_event = threading.Event()
OUTPUT_FOLDER = "output"

# --- Configuration ---
# Using the user-specified turbo model for speed and accuracy
MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"
LANGUAGE = "zh"
SAMPLE_RATE = 16000
# VAD configuration (Silero)
VAD_THRESHOLD = 0.5  # Speech probability threshold. Adjustable via hotkeys.
# FIX: Silero VAD requires a specific chunk size for 16000Hz sample rate.
# The required value is 512 samples.
VAD_CHUNK_SIZE = 512
# intelligent segmentation configuration
SILENCE_DURATION_S = 0.6 # How long a silence indicates the end of an utterance.

audio_queue = queue.Queue()

# --- Helper Functions ---
def load_config():
    """Loads configuration from a .env file and environment variables."""
    load_dotenv()
    return {
        'GEMINI_API_ENDPOINT': os.environ.get('GEMINI_API_ENDPOINT'),
        'GEMINI_API_KEY': os.environ.get('GEMINI_API_KEY')
    }

def get_new_log_filename(extension):
    """Generates a new filename inside the 'output' directory."""
    filename = datetime.datetime.now().strftime("%y%m%d_%H%M%S") + extension
    return os.path.join(OUTPUT_FOLDER, filename)

def call_gemini_api(prompt_text):
    """Sends the provided text to the Gemini API and saves the response."""
    print("\n[INFO] Calling Gemini API...")
    config = load_config()
    api_host, api_key = config.get("GEMINI_API_ENDPOINT"), config.get("GEMINI_API_KEY")

    if not all([api_host, api_key]):
        print("[ERROR] GEMINI_API_ENDPOINT or GEMINI_API_KEY not found in .env file.")
        return

    try:
        with open('system_prompt.md', 'r', encoding='utf-8') as f: system_prompt = f.read()
    except FileNotFoundError:
        system_prompt = "You are a helpful assistant."

    model = "gemini-flash-latest-non-thinking"
    url = f"{api_host}/models/{model}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        llm_text = result['candidates'][0]['content']['parts'][0]['text']
        output_filename = get_new_log_filename(".md")
        with open(output_filename, 'w', encoding='utf-8') as f: f.write(llm_text)
        print(f"[SUCCESS] Gemini response saved to '{output_filename}'")
        print(f"--- Gemini Response ---\n{llm_text}\n-----------------------")
    except requests.exceptions.RequestException as e: print(f"[ERROR] API request failed: {e}")
    except (IndexError, KeyError) as e: print(f"[ERROR] Could not parse Gemini API response: {e}\nFull response: {response.text}")

# --- Core Transcription and VAD Logic ---

def audio_callback(indata, frames, time, status):
    """Callback for the audio stream. Puts audio data (numpy array) into the queue."""
    if is_running.is_set() and not exit_event.is_set():
        if status: print(status, flush=True)
        audio_queue.put(indata.copy())

def transcription_worker(audio_np, whisper_model):
    """This function runs in a separate thread to transcribe a single audio segment."""
    try:
        # --- Start of Debug Info ---
        print(f"\n--- DEBUG: New Transcription Job ---", flush=True)
        print(f"  - Incoming Audio Shape: {audio_np.shape}", flush=True)
        
        # FIX: Flatten the audio array to be 1D, which Whisper expects.
        audio_flat = audio_np.flatten()
        
        print(f"  - Flattened Audio Shape: {audio_flat.shape}", flush=True)
        audio_duration_seconds = len(audio_flat) / SAMPLE_RATE
        print(f"  - Audio Duration: {audio_duration_seconds:.2f} seconds", flush=True)
        print(f"  - Audio dtype: {audio_flat.dtype}", flush=True)

        # Check for near-silence before processing
        if np.max(np.abs(audio_flat)) < 1000:  # A threshold for int16 silent audio
            print("  - DEBUG: Audio appears to be silent, skipping transcription.", flush=True)
            print("------------------------------------", flush=True)
            return
            
        # Convert audio to the format Whisper expects
        audio_fp32 = audio_flat.astype(np.float32) / 32768.0
        
        print(f"  - Processed (float32) min/max: {np.min(audio_fp32):.4f} / {np.max(audio_fp32):.4f}", flush=True)
        print("  - Calling model.transcribe()...", flush=True)
        
        segments, info = whisper_model.transcribe(audio_fp32, language=LANGUAGE, temperature=0.0)
        
        print(f"  - Model detected language: '{info.language}' with probability {info.language_probability:.2f}", flush=True)
        
        transcribed_text = ""
        # Convert the segments iterator to a list to check its length and reuse it
        segment_list = list(segments)
        
        if not segment_list:
            print("  - DEBUG: model.transcribe() returned no segments.", flush=True)
            print("------------------------------------", flush=True)
            return

        print(f"  - Segments found: {len(segment_list)}", flush=True)
        for segment in segment_list:
            print(f"[Transcription] {segment.text}", flush=True)
            transcribed_text += segment.text + "\n"
        
        print("------------------------------------", flush=True)
        # --- End of Debug Info ---

        if transcribed_text.strip() and current_log_filename:
            with open(current_log_filename, 'a', encoding='utf-8') as f:
                f.write(transcribed_text)
    except Exception as e:
        print(f"An error occurred in transcription worker: {e}", flush=True)

def vad_and_segmentation_loop():
    """
    This is the core logic loop. It uses the Silero VAD model to detect speech
    in audio frames and intelligently segments the audio into complete utterances
    before sending them to the transcription worker.
    (这是核心逻辑循环。它使用Silero VAD模型检测音频帧中的语音，
    并将音频智能地分割成完整的语段，然后发送给识别工作线程。)
    """
    print(f"Loading Whisper model '{MODEL_SIZE}'...")
    whisper_model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    print("Whisper model loaded successfully.")

    try:
        # Load Silero VAD model from PyTorch Hub
        print("Loading Silero VAD model...")
        vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      onnx=False)
        print("Silero VAD model loaded successfully.")
    except Exception as e:
        print(f"Failed to load Silero VAD model. Please check your internet connection. Error: {e}")
        exit_event.set()
        return

    speech_buffer = []
    silence_counter = 0
    
    # Calculate how many consecutive silent chunks indicate the end of an utterance
    chunks_per_second = SAMPLE_RATE / VAD_CHUNK_SIZE
    num_silent_chunks_needed = int(SILENCE_DURATION_S * chunks_per_second)

    while not exit_event.is_set():
        # This is the key fix: wait() will block here if is_running is not set (i.e., paused),
        # consuming zero CPU. It unblocks when is_running.set() is called.
        is_running.wait()

        # After waiting, check if the exit event was set, which can happen to unblock the wait.
        if exit_event.is_set():
            break

        try:
            item = audio_queue.get(timeout=0.1)
            
            # A None item is a signal to reset the state, used during pause/resume
            if item is None:
                speech_buffer.clear()
                silence_counter = 0
                continue
            
            audio_chunk_np = item
            # FIX: Squeeze the array to make it 1D, which the VAD model expects.
            audio_chunk_tensor = torch.from_numpy(audio_chunk_np.squeeze()).float() / 32768.0
            
            # Get speech probability from Silero VAD
            speech_prob = vad_model(audio_chunk_tensor, SAMPLE_RATE).item()

            if speech_prob > VAD_THRESHOLD:
                # Speech detected
                speech_buffer.append(audio_chunk_np)
                silence_counter = 0
            else:
                # Silence detected
                if speech_buffer:
                    # If there was speech before this silence, start counting silent chunks
                    silence_counter += 1
                    if silence_counter < num_silent_chunks_needed:
                        # Add a bit of silence as padding
                        speech_buffer.append(audio_chunk_np)
                    else:
                        # End of utterance detected
                        full_utterance = np.concatenate(speech_buffer)
                        speech_buffer.clear()
                        silence_counter = 0
                        # Start a new thread for transcription to avoid blocking
                        threading.Thread(target=transcription_worker, args=(full_utterance, whisper_model)).start()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"An error occurred in VAD loop: {e}")

# --- Hotkey Handling ---
def on_press(key):
    global current_log_filename, VAD_THRESHOLD
    if key == keyboard.Key.space:
        if is_running.is_set():
            # If running, pause it.
            print("\n[HOTKEY] Transcription paused. Press SPACE to resume.", flush=True)
            is_running.clear()
            audio_queue.put(None) # Signal to clear buffers
        else:
            # If paused, resume it.
            current_log_filename = get_new_log_filename(".txt")
            print(f"\n[HOTKEY] Resuming transcription. Logging to '{current_log_filename}'", flush=True)
            audio_queue.put(None) # Signal to clear buffers before starting
            is_running.set()
    elif key == keyboard.Key.enter:
        if current_log_filename:
            # Pause the transcription before sending to API
            if is_running.is_set():
                print("\n[INFO] Pausing transcription to send data...", flush=True)
                is_running.clear()
                audio_queue.put(None)

            try:
                with open(current_log_filename, 'r', encoding='utf-8') as f: prompt = f.read()
                if prompt.strip():
                    print(f"\n[HOTKEY] Sending '{current_log_filename}' to Gemini...", flush=True)
                    threading.Thread(target=call_gemini_api, args=(prompt,)).start()
                else: print("\n[WARN] No data in log file to send.")
            except FileNotFoundError: print(f"[ERROR] Log file '{current_log_filename}' not found.")
        else: print("\n[WARN] No active log file. Resume transcription first.", flush=True)
    elif key == keyboard.Key.esc:
        print("\n[HOTKEY] Escape key pressed. Exiting...", flush=True)
        exit_event.set()
        is_running.set()  # Crucial: Unblock the is_running.wait() call so the loop can terminate.
        audio_queue.put(None) # Unblock queue.get() as well
        return False
    elif key == keyboard.Key.up:
        VAD_THRESHOLD = min(1.0, round(VAD_THRESHOLD + 0.1, 1))
        print(f"\n[HOTKEY] VAD Threshold increased to: {VAD_THRESHOLD:.1f}", flush=True)
    elif key == keyboard.Key.down:
        VAD_THRESHOLD = max(0.1, round(VAD_THRESHOLD - 0.1, 1))
        print(f"\n[HOTKEY] VAD Threshold decreased to: {VAD_THRESHOLD:.1f}", flush=True)


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created directory: '{OUTPUT_FOLDER}'")
        
    print("--- Real-time Chinese STT with LLM Integration ---")
    print("Hotkeys: [SPACE] Pause/Resume | [ENTER] Send to Gemini | [↑/↓] Adjust VAD | [ESC] Exit")
    print("--------------------------------------------------")
    print(f"Initial VAD Threshold: {VAD_THRESHOLD:.1f}")
    print("\nScript is initialized. Press SPACE to start transcribing.")

    # Start the core VAD and segmentation thread
    processing_thread = threading.Thread(target=vad_and_segmentation_loop)
    processing_thread.daemon = True
    processing_thread.start()

    listener = keyboard.Listener(on_press=on_press).start()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=VAD_CHUNK_SIZE, channels=1, dtype='int16', callback=audio_callback):
            processing_thread.join()
    except KeyboardInterrupt: print("\n[INFO] Ctrl+C detected. Forcing exit.")
    except Exception as e: print(f"An error occurred with the audio stream: {e}")
    finally:
        exit_event.set()
        print("Program shut down.")

