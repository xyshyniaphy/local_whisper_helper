# main.py

"""
Batch STT Transcript Correction Tool.

This script reads all .txt files from an 'stt_input' directory, processes them
in chunks to handle large files, corrects transcription errors using a two-stage
LLM approach, and saves the corrected files to an 'stt_output' directory,
preserving the original filenames.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# --- Constants and Configuration ---
INPUT_DIR = Path("stt_input")
OUTPUT_DIR = Path("stt_output")
CHUNK_SIZE_BYTES = 10000  # Process text in chunks of this size.
PROCESS_LOOP_DELAY_S = 60  # 1-minute delay between chunks to avoid rate limits.
MAX_API_RETRIES = 5

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
    stt_chunk: str, config: Config, review_models: List[str], review_prompt: str
) -> str:
    """
    Gets review suggestions from multiple LLMs via OpenRouter.

    This function queries several smaller or specialized models to get diverse
    feedback on the STT text chunk, which is then used as context for the
    final correction model.

    Args:
        stt_chunk: The chunk of text to be reviewed.
        config: The application configuration dictionary.
        review_models: A list of model identifiers to query via OpenRouter.
        review_prompt: The system prompt to guide the review models.

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

    for model_name in review_models:
        print(f"    - Querying review model: {model_name}")
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": review_prompt},
                {"role": "user", "content": stt_chunk},
            ],
        }

        for attempt in range(MAX_API_RETRIES):
            try:
                response = requests.post(
                    api_url, headers=headers, json=payload, timeout=90
                )
                response.raise_for_status()  # Raises HTTPError for bad responses
                result = response.json()
                review_text = result["choices"][0]["message"]["content"]
                all_reviews.append(f"--- Review from {model_name} ---\n{review_text}")
                print(f"    - Success from {model_name}.")
                break  # Exit retry loop on success
            except requests.exceptions.RequestException as e:
                print(f"    [ERROR] API call to {model_name} failed "
                      f"(Attempt {attempt + 1}/{MAX_API_RETRIES}): {e}")
                if attempt < MAX_API_RETRIES - 1:
                    time.sleep(delay_s)
            except (KeyError, IndexError) as e:
                print(f"    [ERROR] Could not parse response from {model_name}: {e}")
                break  # Malformed response, no need to retry

    return "\n\n".join(all_reviews)


def get_corrected_text(
    stt_chunk: str, review_context: str, config: Config, system_prompt: str
) -> Optional[str]:
    """
    Calls the primary Gemini API to perform the final text correction.

    Args:
        stt_chunk: The original chunk of STT text.
        review_context: Contextual reviews from other LLMs.
        config: The application configuration dictionary.
        system_prompt: The system prompt to guide the main correction model.

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

    # Construct the final prompt for the correction model
    user_prompt = (
        f"### STT_REVIEW_CONTEXT\n{review_context}\n\n"
        f"### STT_TEXT_TO_FIX\n{stt_chunk}"
    )

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


def process_file(
    input_path: Path,
    output_path: Path,
    config: Config,
    review_models: List[str],
    review_prompt: str,
    system_prompt: str,
) -> None:
    """
    Reads, processes, and writes a single file in chunks.

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
    chunk_count = 0

    try:
        with open(input_path, "r", encoding="utf-8") as infile, \
             open(output_path, "w", encoding="utf-8") as outfile:

            chunk_lines: List[str] = []
            current_byte_size: int = 0

            for line in infile:
                # Remove all whitespace from the line as per requirements.
                # `str.split()` with no arguments handles all whitespace types.
                processed_line = "".join(line.strip().split())
                if not processed_line:
                    continue  # Skip empty lines

                # Add a newline for semantic separation within the chunk
                line_to_add = processed_line + "\n"
                chunk_lines.append(line_to_add)
                current_byte_size += len(line_to_add.encode("utf-8"))

                if current_byte_size >= CHUNK_SIZE_BYTES:
                    chunk_count += 1
                    print(f"\n[INFO] Processing chunk #{chunk_count} "
                          f"({current_byte_size} bytes)...")
                    
                    # Process the collected chunk
                    chunk_to_process = "".join(chunk_lines)
                    review_context = get_review_context(
                        chunk_to_process, config, review_models, review_prompt
                    )
                    corrected_chunk = get_corrected_text(
                        chunk_to_process, review_context, config, system_prompt
                    )

                    if corrected_chunk:
                        outfile.write(corrected_chunk)
                        outfile.flush()  # Write to disk immediately
                    else:
                        print("[WARN] Failed to correct chunk. Writing original "
                              "chunk to output.")
                        outfile.write(chunk_to_process)

                    # Reset for the next chunk
                    chunk_lines.clear()
                    current_byte_size = 0
                    
                    print(f"[INFO] Waiting for {PROCESS_LOOP_DELAY_S} seconds...")
                    time.sleep(PROCESS_LOOP_DELAY_S)

            # Process any remaining text left in the buffer after the loop
            if chunk_lines:
                chunk_count += 1
                print(f"\n[INFO] Processing final chunk #{chunk_count} "
                      f"({current_byte_size} bytes)...")
                
                final_chunk = "".join(chunk_lines)
                review_context = get_review_context(
                    final_chunk, config, review_models, review_prompt
                )
                corrected_chunk = get_corrected_text(
                    final_chunk, review_context, config, system_prompt
                )

                if corrected_chunk:
                    outfile.write(corrected_chunk)
                else:
                    print("[WARN] Failed to correct final chunk. Writing "
                          "original chunk to output.")
                    outfile.write(final_chunk)

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
    # 1. Setup and validation
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

    # Ensure directories exist
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 2. Find and process files
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
            # Catch exceptions per-file to allow the script to continue
            # with other files if one fails.
            print(f"\n[CRITICAL] Failed to process {input_file_path.name}. "
                  f"Error: {e}. Moving to next file.\n")
            continue

    print("\nAll files have been processed.")


if __name__ == "__main__":
    main()