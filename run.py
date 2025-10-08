# Real-time Chinese Speech-to-Text with Gradio UI, Hotkeys, and LLM Integration
#
# Description:
# This script provides a Gradio web interface for real-time Chinese transcription.
# It uses the Silero VAD model for intelligent segmentation, improving accuracy.
# All output files are saved in a date-stamped subfolder (e.g., 'output/251008').
# Three files are created per session:
#   1. session_fixed_[...].md: LLM-corrected version of the transcription.
#   2. session_summary_[...].md: LLM-generated summary of the fixed text.
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
fixed_text_output_queue = queue.Queue() # For Tab 1
summary_output_queue = queue.Queue()   # For Tab 2
text_buffer = "" # In-memory buffer for transcribed text

# --- Model Placeholders ---
whisper_model = None
vad_model = None

# --- Configuration ---
MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"
LANGUAGE = "zh"
SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = 512
SILENCE_DURATION_S = 0.6
PERIODIC_GEMINI_CALL_S = 300 # 5 minutes
OUTPUT_FOLDER = "output"
VAD_THRESHOLD = 0.5

# --- File & Session Management ---
daily_output_folder = "" # Will be set to something like 'output/251008'
session_fixed_filename = None
session_summary_filename = None

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
            if mic_id is None and device['name'] == mic_name: mic_id = i
            for keyword in loopback_keywords:
                if keyword in device['name']:
                    loopback_ids.append({'id': i, 'name': device['name']})
                    debug_queue.put(f"  -> Found loopback: '{device['name']}' (ID: {i})")
                    break
    
    if not loopback_ids: debug_queue.put("[WARN] No system audio loopback device found.")
    if mic_id is None: mic_id = sd.default.device[0]

    device_cycle = [{'id': mic_id, 'name': mic_name}] + loopback_ids
    return [d['name'] for d in device_cycle]

# --- Core API and Transcription Logic ---

def load_config():
    """Loads configuration from a .env file."""
    load_dotenv()
    return {'GEMINI_API_ENDPOINT': os.environ.get('GEMINI_API_ENDPOINT'), 'GEMINI_API_KEY': os.environ.get('GEMINI_API_KEY')}

def get_new_filename(prefix, extension):
    """Generates a new filename inside the daily output directory."""
    filename = f"{prefix}_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}{extension}"
    return os.path.join(daily_output_folder, filename)

def call_gemini_api(prompt_text, system_prompt_file, model_name, output_filename, output_queue, analysis_type):
    """A generalized function to call the Gemini API for different tasks."""
    debug_queue.put(f"[INFO] Calling Gemini API for {analysis_type}...")
    config = load_config()
    api_host, api_key = config.get("GEMINI_API_ENDPOINT"), config.get("GEMINI_API_KEY")
    if not all([api_host, api_key]):
        debug_queue.put("[ERROR] Gemini API credentials not found.")
        return

    try:
        with open(system_prompt_file, 'r', encoding='utf-8') as f: system_prompt = f.read()
    except FileNotFoundError:
        debug_queue.put(f"[ERROR] System prompt file not found: {system_prompt_file}")
        return

    url = f"{api_host}/models/{model_name}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"parts": [{"text": prompt_text}]}], "systemInstruction": {"parts": [{"text": system_prompt}]}}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        llm_text = result['candidates'][0]['content']['parts'][0]['text']
        
        with open(output_filename, 'a', encoding='utf-8') as f:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"\n\n---\n\n### {analysis_type} at {timestamp}\n\n{llm_text}")
        
        output_queue.put(f"---\n[{analysis_type.upper()} - {timestamp}]\n{llm_text}")
        debug_queue.put(f"[SUCCESS] {analysis_type} response appended to '{output_filename}'")
    except Exception as e:
        debug_queue.put(f"[ERROR] Gemini API call for {analysis_type} failed: {e}")

def send_and_reset_log():
    """Sends the transcribed text to be fixed by Gemini and resets the buffer."""
    global text_buffer
    with gemini_api_lock:
        if not session_fixed_filename:
            debug_queue.put("[WARN] No active session to send.")
            return
        
        prompt = text_buffer
        if prompt.strip():
            debug_queue.put("[INFO] Sending content for fixing and resetting buffer...")
            text_buffer = ""
            threading.Thread(target=call_gemini_api, args=(
                prompt, 'system_prompt.md', 'gemini-flash-latest-non-thinking', 
                session_fixed_filename, fixed_text_output_queue, 'Fixed Text'
            )).start()
        else:
            debug_queue.put("[WARN] No new text to send for fixing.")

def audio_callback(indata, frames, time, status):
    """Callback for the audio stream, placing data in the queue."""
    if is_running.is_set(): audio_queue.put(indata.copy())

def transcription_worker(audio_np):
    """Transcribes a single audio chunk and puts the result in the output queue."""
    global text_buffer, whisper_model
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
    """The main loop for detecting speech and segmenting audio."""
    global vad_model
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
                    threading.Thread(target=transcription_worker, args=(np.concatenate(speech_buffer),)).start()
                    speech_buffer.clear()
                    silence_counter = 0
                else:
                    speech_buffer.append(item)
        except queue.Empty:
            continue
        except Exception as e:
            debug_queue.put(f"[ERROR] VAD loop error: {e}")

def periodic_gemini_call():
    """Periodically sends accumulated text to Gemini for processing."""
    while not exit_event.wait(PERIODIC_GEMINI_CALL_S):
        if is_running.is_set():
            debug_queue.put(f"\n[TIMER] {PERIODIC_GEMINI_CALL_S // 60}-minute timer triggered.")
            send_and_reset_log()

def main_audio_loop():
    """Manages the audio input stream, including restarts and device switching."""
    global current_cycle_index
    while not exit_event.is_set():
        current_device = device_cycle[current_cycle_index]
        try:
            with sd.InputStream(device=current_device['id'], samplerate=SAMPLE_RATE, blocksize=VAD_CHUNK_SIZE, channels=1, dtype='int16', callback=audio_callback):
                restart_stream_event.wait()
        except sd.PortAudioError as e:
            debug_queue.put(f"\n[ERROR] Failed to open '{current_device['name']}'. It may be disabled.")
            device_cycle.pop(current_cycle_index)
            if not device_cycle:
                debug_queue.put("[FATAL] No working audio devices found."); exit_event.set(); break
            current_cycle_index %= len(device_cycle)
            debug_queue.put(f"Switching to next device: '{device_cycle[current_cycle_index]['name']}'")
        finally:
            restart_stream_event.clear()
            if exit_event.is_set(): break

# --- Gradio UI Control Functions ---

def start_pause_transcription(current_status):
    """Handles the Start/Pause button logic."""
    global session_fixed_filename, session_summary_filename
    if current_status == "Start":
        if not session_fixed_filename:
            session_fixed_filename = get_new_filename("session_fixed", ".md")
            session_summary_filename = get_new_filename("session_summary", ".md")
            debug_queue.put(f"New session. Fixed text: '{session_fixed_filename}'")
            debug_queue.put(f"             Summary: '{session_summary_filename}'")
        is_running.set()
        debug_queue.put("[UI] Transcription started.")
        return "Pause"
    else:
        is_running.clear()
        debug_queue.put("[UI] Transcription paused.")
        send_and_reset_log()
        return "Start"

def stop_transcription():
    """Handles the Stop button logic."""
    global session_fixed_filename, session_summary_filename
    is_running.clear()
    send_and_reset_log()
    debug_queue.put("[UI] Transcription stopped. Press Start for a new session.")
    session_fixed_filename, session_summary_filename = None, None
    return "Start"

def change_vad_threshold(value):
    """Updates the VAD threshold from the UI slider."""
    global VAD_THRESHOLD
    VAD_THRESHOLD = value
    debug_queue.put(f"[UI] VAD Threshold set to: {value:.2f}")

def change_audio_source(device_name):
    """Handles changing the audio source from the UI."""
    global current_cycle_index
    debug_queue.put(f"[UI] Audio source changed to '{device_name}'.")
    send_and_reset_log()
    is_running.clear()
    
    for i, device in enumerate(device_cycle):
        if device['name'] == device_name: current_cycle_index = i; break
            
    restart_stream_event.set()
    return "Start"

def summarize_session():
    """Handles the 'Summarize' button click."""
    with gemini_api_lock:
        if not session_fixed_filename or not session_summary_filename:
            debug_queue.put("[WARN] Cannot summarize. No active session.")
            return
        try:
            with open(session_fixed_filename, 'r', encoding='utf-8') as f:
                fixed_text = f.read()
            if fixed_text.strip():
                debug_queue.put("[UI] Summarize button clicked. Sending full fixed text for summary.")
                # FIX: Use the "non-thinking" model as requested for all calls.
                threading.Thread(target=call_gemini_api, args=(
                    fixed_text, 'summarize_prompt.md', 'gemini-flash-latest-non-thinking',
                    session_summary_filename, summary_output_queue, 'Summary'
                )).start()
            else:
                debug_queue.put("[WARN] No fixed text available to summarize.")
        except FileNotFoundError:
            debug_queue.put(f"[ERROR] Fixed text file not found: {session_fixed_filename}")

# --- Gradio UI Output Generators ---

def update_ui_loop(q, is_transcription=False):
    """A generic generator function to update a Gradio Textbox from a queue."""
    full_text = ""
    while not exit_event.is_set():
        try:
            message = q.get(timeout=0.1)
            if is_transcription:
                full_text += message + " "
            else:
                full_text = f"{full_text}\n{message}".strip()
            yield full_text
        except queue.Empty:
            yield full_text

# --- Main Execution ---
def initialize_models():
    """Loads heavyweight models before starting threads."""
    global whisper_model, vad_model
    debug_queue.put(f"Loading Whisper model '{MODEL_SIZE}'...")
    whisper_model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    debug_queue.put("Whisper model loaded.")
    try:
        debug_queue.put("Loading Silero VAD model...")
        torch.hub.set_dir(os.path.join(os.getcwd(), "torch_cache"))
        vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
        debug_queue.put("Silero VAD model loaded.")
    except Exception as e:
        debug_queue.put(f"[FATAL] Failed to load Silero VAD model: {e}"); exit_event.set()

if __name__ == "__main__":
    # --- Setup Daily Output Folder ---
    today_str = datetime.datetime.now().strftime("%y%m%d")
    daily_output_folder = os.path.join(OUTPUT_FOLDER, today_str)
    if not os.path.exists(daily_output_folder):
        os.makedirs(daily_output_folder)
        print(f"Created daily output directory: {daily_output_folder}")

    available_devices = find_audio_devices()
    
    with gr.Blocks() as demo:
        gr.Markdown("# Real-time Chinese Transcription and Analysis")
        with gr.Row():
            start_pause_btn = gr.Button("Start")
            summarize_btn = gr.Button("Summarize")
            stop_btn = gr.Button("Stop")
        with gr.Row():
            ui_vad_slider = gr.Slider(minimum=0.1, maximum=1.0, value=VAD_THRESHOLD, step=0.05, label="VAD Speech Threshold")
            audio_source_radio = gr.Radio(available_devices, value=available_devices[0] if available_devices else None, label="Audio Source")
        
        # FIX: Ensure autoscroll is set for the desired text areas.
        debug_area = gr.Textbox(label="Debug Information", lines=5, max_lines=5, interactive=False, autoscroll=True)
        output_area = gr.Textbox(label="Transcription Output", lines=10, interactive=False, autoscroll=True)
        
        with gr.Tabs():
            with gr.TabItem("Fixed Text"):
                fixed_text_area = gr.Textbox(label=None, lines=10, interactive=False, autoscroll=True)
            with gr.TabItem("Summary"):
                summary_area = gr.Textbox(label=None, lines=10, interactive=False, autoscroll=True)

        # Wire up event listeners
        start_pause_btn.click(start_pause_transcription, inputs=[start_pause_btn], outputs=[start_pause_btn])
        summarize_btn.click(summarize_session, outputs=None)
        stop_btn.click(stop_transcription, outputs=[start_pause_btn])
        ui_vad_slider.change(change_vad_threshold, inputs=[ui_vad_slider])
        audio_source_radio.change(change_audio_source, inputs=[audio_source_radio], outputs=[start_pause_btn])

        # Register UI update generators
        demo.load(lambda: update_ui_loop(debug_queue), outputs=debug_area)
        demo.load(lambda: update_ui_loop(output_queue, is_transcription=True), outputs=output_area)
        demo.load(lambda: update_ui_loop(fixed_text_output_queue), outputs=fixed_text_area)
        demo.load(lambda: update_ui_loop(summary_output_queue), outputs=summary_area)

    # Pre-load models before launching threads and UI
    initialize_models()

    if not exit_event.is_set():
        # Start all background worker threads
        threading.Thread(target=vad_and_segmentation_loop, daemon=True).start()
        threading.Thread(target=periodic_gemini_call, daemon=True).start()
        threading.Thread(target=main_audio_loop, daemon=True).start()

        print("Launching Gradio UI...")
        demo.queue().launch()
        print("Gradio UI closed.")

    # Clean shutdown
    exit_event.set()
    restart_stream_event.set()

