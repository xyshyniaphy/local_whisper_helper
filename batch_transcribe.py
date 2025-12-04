# Batch Audio Transcription and Summarization Script (Azure OpenAI Integration)
#
# Description:
# This script converts audio files (m4a, mp3, wav, etc.) to text using the Faster-Whisper
# library with CUDA acceleration.
#
# Features:
# 1. Prints Audio Duration and Output Text Length.
# 2. Adds Timestamps (e.g., [00:12:30]) to every line in the output file.
# 3. Optional Summarization using Azure OpenAI (ChatGPT).
#
# Requirements:
# uv pip install faster-whisper torch python-dotenv requests

import os
import time
import json
import requests
import datetime
from pathlib import Path
from dotenv import load_dotenv
from faster_whisper import WhisperModel

# --- Configuration Loader ---
def load_config():
    """
    Loads configuration from the .env file.
    Returns a dictionary containing all necessary settings.
    """
    load_dotenv()

    def get_env_var(key, default, type_converter):
        value = os.environ.get(key, default)
        try:
            return type_converter(value)
        except (ValueError, TypeError):
            print(f"[WARN] Invalid value for {key} in .env. Using default: {default}")
            return type_converter(default)

    config = {
        # Processing Flags
        'DO_SUMMARIZE': get_env_var('DO_SUMMARIZE', 'false', lambda x: str(x).strip().lower() == 'true'),

        # Azure OpenAI / Foundry Configuration
        'AZURE_OPENAI_ENDPOINT': get_env_var('AZURE_OPENAI_ENDPOINT', "", str),
        'AZURE_OPENAI_API_KEY': get_env_var('AZURE_OPENAI_API_KEY', "", str),
        'AZURE_OPENAI_DEPLOYMENT_NAME': get_env_var('AZURE_OPENAI_DEPLOYMENT_NAME', "gpt-4o", str),
        'AZURE_OPENAI_API_VERSION': get_env_var('AZURE_OPENAI_API_VERSION', "2024-02-15-preview", str),
        
        # Whisper Model Settings
        'MODEL_SIZE': get_env_var('MODEL_SIZE', 'deepdml/faster-whisper-large-v3-turbo-ct2', str),
        'LANGUAGE': get_env_var('LANGUAGE', 'zh', str),
        
        # File Paths
        'INPUT_FOLDER': get_env_var('INPUT_FOLDER', 'input_audio', str),
        'OUTPUT_FOLDER': get_env_var('OUTPUT_FOLDER', 'batch_output', str),
        
        # Networking
        'MAX_RETRIES': get_env_var('MAX_RETRIES', 3, int),
        'DELAY': get_env_var('DELAY', 5, int),
    }
    return config

# --- Azure OpenAI Logic ---
def call_azure_openai_summary(config, prompt_text, system_prompt_text=None):
    """
    Calls the Azure OpenAI API to summarize text.
    """
    endpoint = config['AZURE_OPENAI_ENDPOINT'].rstrip('/')
    deployment = config['AZURE_OPENAI_DEPLOYMENT_NAME']
    api_version = config['AZURE_OPENAI_API_VERSION']
    api_key = config['AZURE_OPENAI_API_KEY']

    if not all([endpoint, api_key, deployment]):
        print("[ERROR] Missing Azure OpenAI configuration.")
        return None

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    messages = []
    if system_prompt_text:
        messages.append({"role": "system", "content": system_prompt_text})
    messages.append({"role": "user", "content": prompt_text})

    payload = {
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4096 
    }

    for attempt in range(config['MAX_RETRIES']):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"[ERROR] Azure OpenAI API call failed (Attempt {attempt + 1}): {e}")
            if attempt < config['MAX_RETRIES'] - 1:
                time.sleep(config['DELAY'])
    
    print("[ERROR] All retries failed for Azure summarization.")
    return None

def load_text_file(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def format_timestamp(seconds):
    """Converts seconds (float) to HH:MM:SS format."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

# --- Main Transcription Class ---
class BatchTranscriber:
    def __init__(self, config):
        self.config = config
        self.supported_formats = ('.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg')
        self.model = None
        os.makedirs(self.config['INPUT_FOLDER'], exist_ok=True)
        os.makedirs(self.config['OUTPUT_FOLDER'], exist_ok=True)

    def load_model(self):
        print(f"[INFO] Loading Whisper model: {self.config['MODEL_SIZE']} on CUDA...")
        try:
            self.model = WhisperModel(
                self.config['MODEL_SIZE'], 
                device="cuda", 
                compute_type="float16"
            )
            print("[INFO] Model loaded successfully.")
        except Exception as e:
            print(f"[FATAL] Failed to load Whisper model: {e}")
            exit(1)

    def transcribe_file(self, audio_path):
        """
        Transcribes audio, extracts duration, and formats lines with timestamps.
        Returns: (timestamped_text, plain_text_for_summary)
        """
        print(f"[INFO] Transcribing: {os.path.basename(audio_path)}...")
        start_time = time.time()
        
        try:
            # beam_size=5 improves accuracy
            segments, info = self.model.transcribe(
                str(audio_path), 
                language=self.config['LANGUAGE'],
                beam_size=5,
                vad_filter=True # 無音区間のハルシネーション対策に必須
            )
            
            # --- Requirement: Print Audio Length ---
            print(f"      -> Audio Duration: {info.duration:.2f} seconds")

            timestamped_lines = []
            plain_lines = []

            # Iterate through segments
            for segment in segments:
                # --- Requirement: Timestamp for each line ---
                ts = format_timestamp(segment.start)
                text = segment.text.strip()
                
                # Format: [HH:MM:SS] Text
                timestamped_lines.append(f"[{ts}] {text}")
                plain_lines.append(text)

            full_timestamped_text = "\n".join(timestamped_lines)
            full_plain_text = "\n".join(plain_lines)

            # --- Requirement: Print Output Text Length ---
            print(f"      -> Output Text Length: {len(full_timestamped_text)} characters")
            
            process_duration = time.time() - start_time
            print(f"[SUCCESS] Processed in {process_duration:.2f}s.")
            
            return full_timestamped_text, full_plain_text
            
        except Exception as e:
            print(f"[ERROR] Transcription failed for {audio_path}: {e}")
            return None, None

    def process_summarization(self, plain_text, base_filename):
        """
        Summarizes the PLAIN text (without timestamps) to save tokens and improve quality.
        """
        if not self.config['DO_SUMMARIZE']:
            print("[INFO] Summarization skipped.")
            return

        print("[INFO] Starting Azure OpenAI Summarization...")
        
        system_prompt_sum = load_text_file('summarize_prompt.md')
        if not system_prompt_sum:
            system_prompt_sum = "You are a helpful assistant. Please summarize the following transcription clearly."
            
        context_content = load_text_file('context.md')
        summary_prompt = f'{context_content}\n\n"""TEXT TO SUMMARIZE"""\n\n{plain_text}'
        
        summary_text = call_azure_openai_summary(
            self.config, 
            summary_prompt, 
            system_prompt_text=system_prompt_sum
        )

        if summary_text:
            sum_path = os.path.join(self.config['OUTPUT_FOLDER'], f"{base_filename}_summary.md")
            with open(sum_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            print(f"[SUCCESS] Summary saved to: {os.path.basename(sum_path)}")
        else:
            print("[WARN] Summarization failed.")

    def run_batch(self):
        self.load_model()
        
        input_dir = Path(self.config['INPUT_FOLDER'])
        files = [f for f in input_dir.iterdir() if f.suffix.lower() in self.supported_formats]
        
        if not files:
            print(f"[WARN] No audio files found in '{self.config['INPUT_FOLDER']}'.")
            return

        print(f"[INFO] Found {len(files)} audio files to process.")

        for audio_file in files:
            print(f"\n--- Processing: {audio_file.name} ---")
            
            # Step 1: Transcribe
            # We get two versions: one for file saving (with TS), one for AI (clean)
            ts_text, plain_text = self.transcribe_file(audio_file)
            
            if ts_text:
                # Save Timestamped Transcript
                base_name = audio_file.stem
                raw_path = os.path.join(self.config['OUTPUT_FOLDER'], f"{base_name}_raw.txt")
                
                with open(raw_path, 'w', encoding='utf-8') as f:
                    f.write(ts_text)
                print(f"[SUCCESS] Transcript saved to: {os.path.basename(raw_path)}")
                
                # Step 2: Summarize (using the plain text version)
                self.process_summarization(plain_text, base_name)
            
        print("\n[DONE] Batch processing finished.")

if __name__ == "__main__":
    app_config = load_config()
    processor = BatchTranscriber(app_config)
    processor.run_batch()