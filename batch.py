# main.py

"""
Batch STT Transcript Correction Tool.

This script reads all .txt files from an 'stt_input' directory, processes them
in chunks to handle large files, corrects transcription errors using a two-stage
LLM approach, and saves the corrected files to an 'stt_output' directory,
preserving the original filenames. It also generates a detailed debug log for
each file.
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TextIO

import requests
from dotenv import load_dotenv

# --- Constants and Configuration ---
INPUT_DIR = Path("stt_input")
OUTPUT_DIR = Path("stt_output")
CHUNK_SIZE_BYTES = 10000  # Process text in chunks of this size.
PROCESS_LOOP_DELAY_S = 60  # 1-minute delay between chunks to avoid rate limits.
MAX_API_RETRIES = 3

# Type alias for configuration dictionary for clarity.
Config = Dict[str, Optional[str]]


def load_config() -> Config:
    """Loads configuration from the .env file.

    Returns:
        A dictionary containing the configuration values.
    """
    load_dotenv()
    delay_str = os.getenv("API_RETRY_DELAY_S", "5")
    try:
        api_retry_delay = int(delay_str)
    except (ValueError, TypeError):
        print(f"[WARN] Invalid API_RETRY_DELAY_S value '{delay_str}'. "
              f"Defaulting to 5 seconds.")
        api_retry_delay = 5

    return {
        "GEMINI_API_ENDPOINT": os.getenv("GEMINI_API_ENDPOINT"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "OPEN_ROUTER_API": os.getenv("OPEN_ROUTER_API"),
        "OPEN_ROUTER_KEY": os.getenv("OPEN_ROUTER_KEY"),
        "STT_FIX_MODEL": os.getenv("STT_FIX_MODEL", "gemini-1.5-flash-latest"),
        "API_RETRY_DELAY_S": str(api_retry_delay),
    }


def get_review_context(
    stt_chunk: str,
    config: Config,
    review_models: List[str],
    review_prompt: str,
    debug_file: TextIO,
) -> str:
    """
    Gets review suggestions from multiple LLMs via OpenRouter.

    This function queries several smaller or specialized models to get diverse
    feedback on the STT text chunk, which is then used as context for the
    final correction model. It also logs the prompts sent to the debug file.

    Args:
        stt_chunk: The chunk of text to be reviewed.
        config: The application configuration dictionary.
        review_models: A list of model identifiers to query via OpenRouter.
        review_prompt: The system prompt to guide the review models.
        debug_file: The file handle for writing debug information.

    Returns:
        A string containing the concatenated reviews from all models.
    """
    print("  -> Getting STT reviews from helper LLMs...")
    api_url = config.get("OPEN_ROUTER_API")
    api_key = config.get("OPEN_ROUTER_KEY")
    delay_s = int(config.get("API_RETRY_DELAY_S", "5"))

    if not all([api_url, api_key]):
        print("[ERROR] OpenRouter API URL or Key not found in .env file.")
        return ""

    all_reviews: List[str] = []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    debug_file.write("### Review Stage Prompts\n\n")

    for model_name in review_models:
        print(f"    - Querying review model: {model_name}")
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": review_prompt},
                {"role": "user", "content": stt_chunk},
            ],
        }

        # Write debug information before the API call
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        debug_file.write(f"**Timestamp:** {timestamp}\n")
        debug_file.write(f"**Model:** `{model_name}`\n")
        debug_file.write("**User Prompt Sent:**\n")
        debug_file.write(f"```\n{stt_chunk}\n```\n---\n\n")
        debug_file.flush()

        for attempt in range(MAX_API_RETRIES):
            try:
                response = requests.post(
                    api_url, headers=headers, json=payload, timeout=90
                )
                response.raise_for_status()
                result = response.json()
                review_text = result["choices"][0]["message"]["content"]
                all_reviews.append(f"--- Review from {model_name} ---\n{review_text}")
                print(f"    - Success from {model_name}.")
                break
            except requests.exceptions.RequestException as e:
                print(f"    [ERROR] API call to {model_name} failed "
                      f"(Attempt {attempt + 1}/{MAX_API_RETRIES}): {e}")
                if attempt < MAX_API_RETRIES - 1:
                    time.sleep(delay_s)
            except (KeyError, IndexError) as e:
                print(f"    [ERROR] Could not parse response from {model_name}: {e}")
                break

    return "\n\n".join(all_reviews)


def get_corrected_text(
    stt_chunk: str,
    review_context: str,
    config: Config,
    system_prompt: str,
    debug_file: TextIO,
) -> Optional[str]:
    """
    Calls the primary Gemini API to perform the final text correction.

    Args:
        stt_chunk: The original chunk of STT text.
        review_context: Contextual reviews from other LLMs.
        config: The application configuration dictionary.
        system_prompt: The system prompt to guide the main correction model.
        debug_file: The file handle for writing debug information.

    Returns:
        The corrected text as a string, or None if the API call fails.
    """
    print("  -> Sending text to primary LLM for final correction...")
    api_host = config.get("GEMINI_API_ENDPOINT")
    api_key = config.get("GEMINI_API_KEY")
    model_name = config.get("STT_FIX_MODEL")
    delay_s = int(config.get("API_RETRY_DELAY_S", "5"))

    if not all([api_host, api_key, model_name]):
        print("[ERROR] Gemini API credentials or model name not found in .env.")
        return None

    user_prompt = (
        f"### STT_REVIEW_CONTEXT\n{review_context}\n\n"
        f"### STT_TEXT_TO_FIX\n{stt_chunk}"
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    debug_file.write("### Final Correction Stage Prompt\n\n")
    debug_file.write(f"**Timestamp:** {timestamp}\n")
    debug_file.write(f"**Model:** `{model_name}`\n")
    debug_file.write("**User Prompt Sent:**\n")
    debug_file.write(f"```\n{user_prompt}\n```\n---\n\n")
    debug_file.flush()

    url = f"{api_host}/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    for attempt in range(MAX_API_RETRIES):
        try:
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=180
            )
            response.raise_for_status()
            result = response.json()
            corrected_text = result["candidates"][0]["content"]["parts"][0]["text"]
            print("  -> Correction successful.")
            return corrected_text
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] Gemini API call failed "
                  f"(Attempt {attempt + 1}/{MAX_API_RETRIES}): {e}")
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(delay_s)
        except (KeyError, IndexError) as e:
            print(f"  [ERROR] Could not parse Gemini response: {e}")
            return None

    print("  [ERROR] All retries failed for Gemini API call.")
    return None


def _process_and_write_chunk(
    chunk_to_process: str,
    chunk_count: int,
    outfile: TextIO,
    debugfile: TextIO,
    config: Config,
    review_models: List[str],
    review_prompt: str,
    system_prompt: str,
) -> None:
    """
    A common helper function to process a single chunk of text and write results.

    This function encapsulates the logic for getting reviews, getting the final
    correction, and writing the output and debug information to the respective
    files.

    Args:
        chunk_to_process: The string content of the chunk to process.
        chunk_count: The sequential number of the chunk for logging.
        outfile: The file handle for the corrected text output.
        debugfile: The file handle for the debug log.
        config: The application configuration dictionary.
        review_models: A list of model identifiers for the review stage.
        review_prompt: The prompt for the review models.
        system_prompt: The prompt for the main correction model.
    """
    byte_size = len(chunk_to_process.encode("utf-8"))
    print(f"\n[INFO] Processing chunk #{chunk_count} ({byte_size} bytes)...")
    debugfile.write(f"# CHUNK {chunk_count}\n\n")

    review_context = get_review_context(
        chunk_to_process, config, review_models, review_prompt, debugfile
    )
    corrected_chunk = get_corrected_text(
        chunk_to_process, review_context, config, system_prompt, debugfile
    )

    if corrected_chunk:
        outfile.write(corrected_chunk)
    else:
        print("[WARN] Failed to correct chunk. Writing original chunk to output.")
        outfile.write(chunk_to_process)

    outfile.flush()


def process_file(
    input_path: Path,
    output_path: Path,
    config: Config,
    review_models: List[str],
    review_prompt: str,
    system_prompt: str,
) -> None:
    """
    Reads an entire file, cleans it, splits it into sentence-aware chunks,
    and processes each chunk.

    Args:
        input_path: Path to the source .txt file.
        output_path: Path to write the corrected .txt file.
        config: The application configuration dictionary.
        review_models: A list of model identifiers for the review stage.
        review_prompt: The prompt for the review models.
        system_prompt: The prompt for the main correction model.

    Raises:
        IOError: If there's an issue reading the input or writing the output file.
    """
    print(f"\n--- Processing file: {input_path.name} ---")
    debug_path = output_path.with_suffix(".debug.md")

    try:
        with open(input_path, "r", encoding="utf-8") as infile:
            full_text = infile.read()

        # 1. Efficiently remove all whitespace characters (space, tab, newline, etc.)
        clean_text = re.sub(r"\s+", "", full_text)

        # 2. Split by Chinese period "。", keeping the delimiter with the preceding part.
        # The lookbehind `(?<=。)` splits the string *after* the delimiter.
        sentences = re.split(r"(?<=。)", clean_text)
        # Filter out any empty strings that might result from the split
        sentences = [s for s in sentences if s]

        if not sentences:
            print(f"[INFO] File '{input_path.name}' is empty after cleaning. Skipping.")
            # Create empty output files to signify completion
            output_path.touch()
            debug_path.touch()
            return

        with open(output_path, "w", encoding="utf-8") as outfile, \
             open(debug_path, "w", encoding="utf-8") as debugfile:

            chunk_count = 0
            current_chunk_parts: List[str] = []
            current_byte_size = 0

            # 3. Combine sentences into chunks of the desired size
            for sentence in sentences:
                sentence_bytes = len(sentence.encode("utf-8"))

                # If adding the next sentence exceeds the size, process the current chunk
                if current_byte_size > 0 and (current_byte_size + sentence_bytes) > CHUNK_SIZE_BYTES:
                    chunk_count += 1
                    chunk_to_process = "".join(current_chunk_parts)
                    
                    _process_and_write_chunk(
                        chunk_to_process, chunk_count, outfile, debugfile,
                        config, review_models, review_prompt, system_prompt
                    )

                    # Reset for the next chunk, starting with the current sentence
                    current_chunk_parts = [sentence]
                    current_byte_size = sentence_bytes
                    
                    print(f"[INFO] Waiting for {PROCESS_LOOP_DELAY_S} seconds...")
                    time.sleep(PROCESS_LOOP_DELAY_S)
                else:
                    # Otherwise, add the sentence to the current chunk
                    current_chunk_parts.append(sentence)
                    current_byte_size += sentence_bytes

            # 4. Process the final remaining chunk
            if current_chunk_parts:
                chunk_count += 1
                final_chunk = "".join(current_chunk_parts)
                _process_and_write_chunk(
                    final_chunk, chunk_count, outfile, debugfile,
                    config, review_models, review_prompt, system_prompt
                )

    except FileNotFoundError:
        raise IOError(f"Input file not found: {input_path}")
    except Exception as e:
        raise IOError(f"An error occurred during file processing for "
                      f"{input_path.name}: {e}")

    print(f"--- Finished processing {input_path.name} ---")


def main() -> None:
    """
    Main function to orchestrate the STT correction process.
    """
    print("Starting STT Batch Correction Script...")
    config = load_config()

    required_files = [
        Path("review_llm_ids.txt"),
        Path("review_prompt.md"),
        Path("system_prompt.md"),
    ]

    for f in required_files:
        if not f.exists():
            print(f"[FATAL] Required file not found: {f}. Exiting.")
            return

    try:
        review_models = [
            line.strip() for line in open("review_llm_ids.txt", "r",
                                          encoding="utf-8") if line.strip()
        ]
        review_prompt = open("review_prompt.md", "r", encoding="utf-8").read()
        system_prompt = open("system_prompt.md", "r", encoding="utf-8").read()
    except IOError as e:
        print(f"[FATAL] Could not read prompt files: {e}. Exiting.")
        return

    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    txt_files_to_process = list(INPUT_DIR.glob("*.txt"))
    if not txt_files_to_process:
        print(f"[INFO] No .txt files found in '{INPUT_DIR}'. Nothing to do.")
        return

    print(f"Found {len(txt_files_to_process)} files to process.")

    for input_file_path in txt_files_to_process:
        output_file_path = OUTPUT_DIR / input_file_path.name
        try:
            process_file(
                input_file_path,
                output_file_path,
                config,
                review_models,
                review_prompt,
                system_prompt,
            )
        except Exception as e:
            print(f"\n[CRITICAL] Failed to process {input_file_path.name}. "
                  f"Error: {e}. Moving to next file.\n")
            continue

    print("\nAll files have been processed.")


if __name__ == "__main__":
    main()