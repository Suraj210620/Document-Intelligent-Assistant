import anthropic # type: ignore
import json
import os


# Initializing the Anthropic client
# It automatically reads ANTHROPIC_API_KEY from environment variable
client = anthropic.Anthropic()


def build_prompt(raw_text, layout):
    """
    Build a detailed prompt for Claude to clean and structure the extracted text.

    Args:
        raw_text: raw OCR extracted text string
        layout: structured layout dict from layout.py

    Returns:
        prompt string to send to Claude
    """
    headers_text = "\n".join(layout.get("headers", [])) if layout else ""
    body_text = "\n".join(layout.get("body", [])) if layout else ""
    footers_text = "\n".join(layout.get("footers", [])) if layout else ""

    prompt = f"""You are a document analysis assistant. Below is raw text extracted from a computer screenshot using OCR. The text may contain minor spelling errors, broken words, or formatting issues caused by OCR.

Your tasks:
1. Fix any OCR errors and clean up the text
2. Identify what type of content this screenshot contains (e.g. code editor, browser, terminal, settings, chat window, document, etc.)
3. Extract all meaningful information as structured key-value pairs
4. Return ONLY a valid JSON object — no explanation, no markdown, no extra text

--- EXTRACTED TEXT ---
HEADERS:
{headers_text if headers_text else "(none detected)"}

BODY:
{body_text if body_text else raw_text}

FOOTERS:
{footers_text if footers_text else "(none detected)"}
--- END OF TEXT ---

Return a JSON object with this structure:
{{
  "document_type": "type of screenshot content",
  "title": "main title or heading if found",
  "summary": "one sentence description of what this screenshot shows",
  "key_information": {{
    "field_name": "value",
    "another_field": "value"
  }},
  "full_text_cleaned": "the complete cleaned text in reading order",
  "ocr_corrections": ["list of corrections made, e.g. 'teh -> the'"],
  "confidence": "high / medium / low — your confidence in the extraction"
}}

Important:
- Use snake_case for all key names
- If a field has no value, use null
- Keep all original content — do not summarize or remove information
- Return ONLY the JSON object, nothing else"""

    return prompt


def call_claude_api(prompt):
    print("[INFO] Sending text to Claude API...")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    response_text = message.content[0].text
    print("[OK] Response received from Claude API")
    print(f"[INFO] Input tokens used  : {message.usage.input_tokens}")
    print(f"[INFO] Output tokens used : {message.usage.output_tokens}")

    return response_text


def parse_json_response(response_text):
    """
    Parse Claude's response into a Python dictionary.
    Handles cases where Claude accidentally wraps JSON in markdown code blocks.

    Args:
        response_text: raw text response from Claude

    Returns:
        parsed dictionary or None if parsing fails
    """
    # Strip markdown code fences if present (e.g. ```json ... ```)
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```) and last line (```)
        cleaned = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(cleaned)
        print("[OK] JSON parsed successfully")
        return parsed
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON response: {e}")
        print("[INFO] Raw response saved for inspection")
        return None


def save_json_output(data, original_image_path, output_folder="output"):
    """
    Save the structured JSON output to a file.
    Filename will be: original_name_result.json

    Args:
        data: parsed dictionary from Claude
        original_image_path: path to original image (used for naming)
        output_folder: folder to save the JSON file

    Returns:
        path to the saved JSON file
    """
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.basename(original_image_path)
    name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_folder, f"{name}_result.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Final result saved: {output_path}")
    return output_path


def save_raw_response(response_text, original_image_path, output_folder="output"):
    """
    Save Claude's raw response as a fallback if JSON parsing fails.

    Args:
        response_text: raw text response from Claude
        original_image_path: path to original image
        output_folder: folder to save the raw response
    """
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.basename(original_image_path)
    name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_folder, f"{name}_raw_response.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response_text)

    print(f"[INFO] Raw response saved: {output_path}")


def refine_with_llm(raw_text, layout, original_image_path):
    """
    Master LLM post-processing function — runs all steps in order.

    Steps:
        1. Build a detailed prompt from raw text and layout
        2. Send prompt to Claude API
        3. Parse the JSON response
        4. Save the final structured result

    Args:
        raw_text: raw OCR text string from ocr.py
        layout: structured layout dict from layout.py
        original_image_path: path to original image (used for naming output)

    Returns:
        structured result dict and path to saved JSON file
    """
    print("\n====== Phase 4: LLM Post-Processing ======")

    if not raw_text:
        print("[ERROR] No text provided for LLM processing.")
        return None, None

    # Step 1: Build prompt
    print("[INFO] Building prompt...")
    prompt = build_prompt(raw_text, layout)

    # Step 2: Call Claude API
    response_text = call_claude_api(prompt)

    # Step 3: Parse response
    result = parse_json_response(response_text)

    # Step 4: Save output
    if result:
        output_path = save_json_output(result, original_image_path)
    else:
        # Save raw response as fallback
        save_raw_response(response_text, original_image_path)
        output_path = None

    print("====== LLM Post-Processing Complete ======\n")
    return result, output_path

#----------------------------------------------------
if __name__ == "__main__":
    from Preprocess import preprocess_image
    from OCR import extract_text
    from layout import analyze_layout

    test_image = "input_images/test_img.png"  

    # Step 1: Pre-process
    processed_image, _ = preprocess_image(test_image)

    if processed_image is not None:

        # Step 2: Extract text
        raw_text = extract_text(processed_image)

        # Step 3: Analyze layout
        layout, _ = analyze_layout(processed_image, test_image)

        # Step 4: Refine with Claude
        result, result_path = refine_with_llm(raw_text, layout, test_image)

        if result:
            print("=" * 50)
            print("FINAL STRUCTURED RESULT:")
            print("=" * 50)
            print(f"Document Type : {result.get('document_type', 'N/A')}")
            print(f"Title         : {result.get('title', 'N/A')}")
            print(f"Summary       : {result.get('summary', 'N/A')}")
            print(f"Confidence    : {result.get('confidence', 'N/A')}")
            print("\nKey Information:")
            for key, value in result.get("key_information", {}).items():
                print(f"  {key}: {value}")
            if result.get("ocr_corrections"):
                print("\nOCR Corrections Made:")
                for correction in result.get("ocr_corrections", []):
                    print(f"  → {correction}")
            print("=" * 50)
            print(f"\nFull result saved to: {result_path}")
        else:
            print("[ERROR] LLM processing failed. Check raw response file in output/")

    else:
        print("[ERROR] Pre-processing failed. Cannot run LLM processing.")