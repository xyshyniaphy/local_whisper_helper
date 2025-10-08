# Real-time Chinese Speech-to-Text with Gradio UI, Hotkeys, and LLM Integration
#
# Description:
# This script provides a Gradio web interface for real-time Chinese transcription.
# It uses the Silero VAD model for intelligent segmentation, improving accuracy.
# All output files are saved in a subfolder named 'output'.
# A single markdown file is created per session to log all Gemini outputs.
#
# Author: Gemini
# Date: 2025-10-08

# --- Installation ---
# Use 'uv' (a fast package installer) in your terminal.
#
# 1. Install PyTorch with the specific CUDA version for your system (cu121 is recommended):
#    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#
# 2. Then, install the other required libraries, including Gradio:
#    uv pip install faster-whisper sounddevice numpy python-dotenv requests gradio
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
from dotenv import load_dotenv
import torch
import time
import gradio as gr

# --- Global State & Queues ---
is_running = threading.Event()
exit_event = threading.Event()
restart_stream_event = threading.Event()
gemini_api_lock = threading.Lock()
audio_queue = queue.Queue()
debug_queue = queue.Queue()
output_queue = queue.Queue()
text_buffer = "" # In-memory buffer for transcribed text

# Global references to Gradio components to update them
ui_vad_slider = None

# --- Configuration ---
MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"
LANGUAGE = "zh"
SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = 512
SILENCE_DURATION_S = 0.6
PERIODIC_GEMINI_CALL_S = 300 # 5 minutes
OUTPUT_FOLDER = "output"
# This will be updated by the Gradio slider
VAD_THRESHOLD = 0.5

# --- File & Session Management ---
current_log_filename = None
session_markdown_filename = None

# --- Audio Device Management ---
device_cycle = []
current_cycle_index = 0

def find_audio_devices():
    """Finds all input devices and prepares them for the Gradio UI."""
    global device_cycle
    mic_id, loopback_ids = None, []
    mic_name = "Default Microphone"
    
    try:
        default_device_info = sd.query_devices(kind='input')
        mic_name = default_device_info['name']
        debug_queue.put(f"Default Microphone: '{mic_name}' (ID: {default_device_info['index']})")
    except Exception as e:
        debug_queue.put(f"Could not query default mic. Using system default. Error: {e}")
        mic_id = sd.default.device[0]

    debug_queue.put("Searching for system audio loopback devices...")
    devices = sd.query_devices()
    loopback_keywords = ['Stereo Mix', 'What U Hear', 'Loopback']

    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            if mic_id is None and device['name'] == mic_name:
                mic_id = i
            for keyword in loopback_keywords:
                if keyword in device['name']:
                    loopback_ids.append({'id': i, 'name': device['name']})
                    debug_queue.put(f"  -> Found loopback: '{device['name']}' (ID: {i})")
                    break
    
    if not loopback_ids:
        debug_queue.put("[WARN] No system audio loopback device found. Enable 'Stereo Mix' in Windows.")
    if mic_id is None:
        mic_id = sd.default.device[0]

    # Populate the master device list for the UI
    device_cycle = [{'id': mic_id, 'name': mic_name}] + loopback_ids
    return [d['name'] for d in device_cycle]

# --- Core API and Transcription Logic ---

def load_config():
    load_dotenv()
    return {'GEMINI_API_ENDPOINT': os.environ.get('GEMINI_API_ENDPOINT'), 'GEMINI_API_KEY': os.environ.get('GEMINI_API_KEY')}

def get_new_filename(prefix, extension):
    filename = f"{prefix}_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}{extension}"
    return os.path.join(OUTPUT_FOLDER, filename)

def call_gemini_api(prompt_text, md_filename):
    debug_queue.put("[INFO] Calling Gemini API...")
    config = load_config()
    api_host, api_key = config.get("GEMINI_API_ENDPOINT"), config.get("GEMINI_API_KEY")
    if not all([api_host, api_key]):
        debug_queue.put("[ERROR] Gemini API credentials not found.")
        return

    try:
        with open('system_prompt.md', 'r', encoding='utf-8') as f: system_prompt = f.read()
    except FileNotFoundError: system_prompt = "You are a helpful assistant."

    model, url = "gemini-flash-latest-non-thinking", f"{api_host}/models/gemini-flash-latest-non-thinking:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"parts": [{"text": prompt_text}]}], "systemInstruction": {"parts": [{"text": system_prompt}]}}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        llm_text = result['candidates'][0]['content']['parts'][0]['text']
        
        with open(md_filename, 'a', encoding='utf-8') as f:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"\n\n---\n\n### Analysis at {timestamp}\n\n{llm_text}")
        
        output_queue.put(f"\n\n---\n[GEMINI ANALYSIS - {timestamp}]\n{llm_text}")
        debug_queue.put(f"[SUCCESS] Gemini response appended to '{md_filename}'")
    except Exception as e:
        debug_queue.put(f"[ERROR] Gemini API call failed: {e}")

def send_and_reset_log():
    global text_buffer
    with gemini_api_lock:
        if not session_markdown_filename:
            debug_queue.put("[WARN] No active session to send.")
            return
        
        prompt = text_buffer
        if prompt.strip():
            debug_queue.put("[INFO] Sending content to Gemini and resetting buffer...")
            text_buffer = "" # Reset buffer immediately
            threading.Thread(target=call_gemini_api, args=(prompt, session_markdown_filename)).start()
        else:
            debug_queue.put("[WARN] No new text to send to Gemini.")

def audio_callback(indata, frames, time, status):
    if is_running.is_set():
        audio_queue.put(indata.copy())

def transcription_worker(audio_np, whisper_model):
    global text_buffer
    try:
        audio_flat = audio_np.flatten()
        if np.max(np.abs(audio_flat)) < 1000: return
            
        audio_fp32 = audio_flat.astype(np.float32) / 32768.0
        segments, _ = whisper_model.transcribe(audio_fp32, language=LANGUAGE, temperature=0.0)
        
        for segment in segments:
            transcribed_text = segment.text
            output_queue.put(transcribed_text)
            text_buffer += transcribed_text + "\n"
    except Exception as e:
        debug_queue.put(f"[ERROR] Transcription worker failed: {e}")

def vad_and_segmentation_loop():
    debug_queue.put(f"Loading Whisper model '{MODEL_SIZE}'...")
    whisper_model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    debug_queue.put("Whisper model loaded.")

    try:
        debug_queue.put("Loading Silero VAD model...")
        vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
        debug_queue.put("Silero VAD model loaded.")
    except Exception as e:
        debug_queue.put(f"[FATAL] Failed to load Silero VAD model: {e}")
        exit_event.set()
        return

    speech_buffer, silence_counter = [], 0
    num_silent_chunks_needed = int(SILENCE_DURATION_S * (SAMPLE_RATE / VAD_CHUNK_SIZE))

    while not exit_event.is_set():
        is_running.wait()
        if exit_event.is_set(): break

        try:
            item = audio_queue.get(timeout=0.1)
            audio_chunk_tensor = torch.from_numpy(item.squeeze()).float() / 32768.0
            speech_prob = vad_model(audio_chunk_tensor, SAMPLE_RATE).item()

            if speech_prob > VAD_THRESHOLD:
                speech_buffer.append(item)
                silence_counter = 0
            elif speech_buffer:
                silence_counter += 1
                if silence_counter >= num_silent_chunks_needed:
                    full_utterance = np.concatenate(speech_buffer)
                    speech_buffer.clear()
                    silence_counter = 0
                    threading.Thread(target=transcription_worker, args=(full_utterance, whisper_model)).start()
                else:
                    speech_buffer.append(item)
        except queue.Empty:
            continue
        except Exception as e:
            debug_queue.put(f"[ERROR] VAD loop error: {e}")

def periodic_gemini_call():
    while not exit_event.wait(PERIODIC_GEMINI_CALL_S):
        if is_running.is_set():
            debug_queue.put(f"\n[TIMER] {PERIODIC_GEMINI_CALL_S // 60}-minute timer triggered.")
            send_and_reset_log()

def main_audio_loop():
    global current_cycle_index
    while not exit_event.is_set():
        current_device = device_cycle[current_cycle_index]
        try:
            with sd.InputStream(device=current_device['id'], samplerate=SAMPLE_RATE, blocksize=VAD_CHUNK_SIZE, channels=1, dtype='int16', callback=audio_callback):
                restart_stream_event.wait() # Block until a restart is needed
        except sd.PortAudioError as e:
            debug_queue.put(f"\n[ERROR] Failed to open '{current_device['name']}'. It may be disabled.")
            device_cycle.pop(current_cycle_index)
            if not device_cycle:
                debug_queue.put("[FATAL] No working audio devices found.")
                exit_event.set()
                break
            current_cycle_index %= len(device_cycle)
            new_device = device_cycle[current_cycle_index]
            debug_queue.put(f"Switching to next device: '{new_device['name']}'")
            # No wait, just loop to retry with the new device
        finally:
            restart_stream_event.clear()
            if exit_event.is_set(): break

# --- Gradio UI Control Functions ---

def start_pause_transcription(current_status):
    global session_markdown_filename
    if current_status == "Stop": # If it was stopped, now we start
        if not session_markdown_filename:
            session_markdown_filename = get_new_filename("session", ".md")
            debug_queue.put(f"New session started. Analysis will be saved to '{session_markdown_filename}'")
        is_running.set()
        debug_queue.put("[UI] Transcription started.")
        return "Pause"
    else: # If it was running, now we pause
        is_running.clear()
        debug_queue.put("[UI] Transcription paused.")
        return "Start"

def stop_transcription():
    is_running.clear()
    debug_queue.put("[UI] Transcription stopped. Press Start to begin a new session.")
    # In a real app, you might do more cleanup. For now, this is a soft stop.
    # To fully stop, user should close the app.
    return "Start"

def change_vad_threshold(value):
    global VAD_THRESHOLD
    VAD_THRESHOLD = value
    debug_queue.put(f"[UI] VAD Threshold set to: {value:.2f}")

def change_audio_source(device_name):
    global current_cycle_index
    debug_queue.put(f"[UI] Audio source changed to '{device_name}'.")
    send_and_reset_log() # Process previous text
    is_running.clear() # Pause
    
    # Find the index of the selected device
    for i, device in enumerate(device_cycle):
        if device['name'] == device_name:
            current_cycle_index = i
            break
            
    restart_stream_event.set() # Signal the main loop to restart the stream
    return "Start" # Update button to "Start"

# --- Gradio UI Output Generators ---

def update_debug_output():
    full_log = ""
    while not exit_event.is_set():
        try:
            message = debug_queue.get(timeout=0.1)
            full_log = f"{full_log}\n{message}".strip()
            yield full_log
        except queue.Empty:
            yield full_log # Keep updating to maintain state

def update_main_output():
    full_text = ""
    while not exit_event.is_set():
        try:
            message = output_queue.get(timeout=0.1)
            full_text += message + " "
            yield full_text.strip().replace("\n ", "\n")
        except queue.Empty:
            yield full_text.strip().replace("\n ", "\n")

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    available_devices = find_audio_devices()
    
    # Start all background threads
    threading.Thread(target=vad_and_segmentation_loop, daemon=True).start()
    threading.Thread(target=periodic_gemini_call, daemon=True).start()
    threading.Thread(target=main_audio_loop, daemon=True).start()

    with gr.Blocks() as demo:
        gr.Markdown("# Real-time Chinese Transcription and Analysis")
        with gr.Row():
            start_pause_btn = gr.Button("Start")
            stop_btn = gr.Button("Stop")
        with gr.Row():
            ui_vad_slider = gr.Slider(minimum=0.1, maximum=1.0, value=VAD_THRESHOLD, step=0.05, label="VAD Speech Threshold")
            audio_source_radio = gr.Radio(available_devices, value=available_devices[0], label="Audio Source")
        
        debug_area = gr.Textbox(label="Debug Information", lines=5, max_lines=5, interactive=False)
        output_area = gr.Textbox(label="Transcription Output", lines=15, interactive=False)

        # Wire up event listeners
        start_pause_btn.click(start_pause_transcription, inputs=[start_pause_btn], outputs=[start_pause_btn])
        stop_btn.click(stop_transcription, outputs=[start_pause_btn])
        ui_vad_slider.change(change_vad_threshold, inputs=[ui_vad_slider])
        audio_source_radio.change(change_audio_source, inputs=[audio_source_radio], outputs=[start_pause_btn])

        # Register UI update generators
        demo.load(update_debug_output, outputs=debug_area, every=1)
        demo.load(update_main_output, outputs=output_area, every=1)

    print("Launching Gradio UI...")
    demo.queue().launch()
    print("Gradio UI closed.")
    exit_event.set() # Signal all threads to exit when UI is closed
    restart_stream_event.set() # Unblock main audio loop if it's waiting

