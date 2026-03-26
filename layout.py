import pytesseract
import cv2
import os
import json

# Tell Python exactly where Tesseract is installed on Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def detect_text_blocks(processed_image):
    """
    Detect individual text blocks and their positions in the image.
    Uses Tesseract's built-in layout analysis to find text regions.

    Args:
        processed_image: numpy array (output from preprocess.py)

    Returns:
        list of dicts, each representing a detected text block with
        its text content and position on the page
    """
    print("\n--- Starting Layout Analysis ---")

    if processed_image is None:
        print("[ERROR] No image provided for layout analysis.")
        return None

    # --psm 3 = fully automatic page segmentation (detects columns, blocks)
    data = pytesseract.image_to_data(
        processed_image,
        lang='eng',
        config='--psm 3',
        output_type=pytesseract.Output.DICT
    )

    # Group words into blocks using Tesseract's block_num field
    blocks = {}
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        confidence = int(data['conf'][i])
        block_num = data['block_num'][i]
        line_num = data['line_num'][i]

        # Only process actual words with valid confidence
        if word and confidence > 0:
            key = (block_num, line_num)
            if key not in blocks:
                blocks[key] = {
                    "block_num": block_num,
                    "line_num": line_num,
                    "words": [],
                    "confidences": [],
                    "left": data['left'][i],
                    "top": data['top'][i],
                    "right": data['left'][i] + data['width'][i],
                    "bottom": data['top'][i] + data['height'][i]
                }
            blocks[key]["words"].append(word)
            blocks[key]["confidences"].append(confidence)

            # Expand bounding box to cover all words in the block
            blocks[key]["right"] = max(
                blocks[key]["right"],
                data['left'][i] + data['width'][i]
            )
            blocks[key]["bottom"] = max(
                blocks[key]["bottom"],
                data['top'][i] + data['height'][i]
            )

    # Convert blocks dict to a clean list
    block_list = []
    for key, block in sorted(blocks.items()):
        text_line = " ".join(block["words"])
        avg_confidence = round(
            sum(block["confidences"]) / len(block["confidences"]), 1
        )
        block_list.append({
            "block_num": block["block_num"],
            "line_num": block["line_num"],
            "text": text_line,
            "avg_confidence": avg_confidence,
            "position": {
                "left": block["left"],
                "top": block["top"],
                "right": block["right"],
                "bottom": block["bottom"]
            }
        })

    print(f"[OK] Detected {len(block_list)} text lines across layout blocks")
    print("--- Layout Analysis Complete ---\n")
    return block_list


def classify_blocks(block_list):
    """
    Classify each detected text block by its likely role in the document.
    Uses simple heuristics based on text length, position, and confidence.

    Classifications:
        - header    : short text near the top of the page
        - footer    : short text near the bottom of the page
        - body      : normal paragraph text
        - noise     : very short or low confidence fragments

    Args:
        block_list: list of block dicts from detect_text_blocks()

    Returns:
        same list with a 'type' field added to each block
    """
    print("--- Classifying Text Blocks ---")

    if not block_list:
        return []

    # Finding page boundaries from all block positions
    all_tops = [b["position"]["top"] for b in block_list]
    all_bottoms = [b["position"]["bottom"] for b in block_list]
    page_top = min(all_tops)
    page_bottom = max(all_bottoms)
    page_height = page_bottom - page_top

    for block in block_list:
        text = block["text"]
        top = block["position"]["top"]
        confidence = block["avg_confidence"]
        word_count = len(text.split())

        # Classify based on position and content
        if confidence < 40 or word_count == 0:
            block["type"] = "noise"
        elif top < page_top + (page_height * 0.1) and word_count <= 10:
            block["type"] = "header"
        elif top > page_top + (page_height * 0.9) and word_count <= 10:
            block["type"] = "footer"
        else:
            block["type"] = "body"

    # Printing summary
    summary = {}
    for block in block_list:
        summary[block["type"]] = summary.get(block["type"], 0) + 1

    for block_type, count in summary.items():
        print(f"   {block_type.capitalize()}: {count} line(s)")

    print("--- Classification Complete ---\n")
    return block_list


def build_structured_layout(block_list):
    """
    Organize classified blocks into a structured layout dictionary.
    Groups blocks by type: headers, body, footers.

    Args:
        block_list: classified block list from classify_blocks()

    Returns:
        dict with keys: headers, body, footers, full_text
    """
    print("--- Building Structured Layout ---")

    layout = {
        "headers": [],
        "body": [],
        "footers": [],
        "full_text": ""
    }

    for block in block_list:
        block_type = block.get("type", "body")
        if block_type == "header":
            layout["headers"].append(block["text"])
        elif block_type == "footer":
            layout["footers"].append(block["text"])
        elif block_type == "body":
            layout["body"].append(block["text"])
        # noise blocks are intentionally skipped

    # Build full text in reading order (header → body → footer)
    all_text = layout["headers"] + layout["body"] + layout["footers"]
    layout["full_text"] = "\n".join(all_text)

    print(f"[OK] Layout built:")
    print(f"     Headers : {len(layout['headers'])} line(s)")
    print(f"     Body    : {len(layout['body'])} line(s)")
    print(f"     Footers : {len(layout['footers'])} line(s)")
    print("--- Structured Layout Complete ---\n")
    return layout


def save_layout(layout, original_image_path, output_folder="output"):
    """
    Save the structured layout as a JSON file.
    Filename will be: original_name_layout.json

    Args:
        layout: structured layout dict from build_structured_layout()
        original_image_path: path to original image (used for naming)
        output_folder: folder to save the JSON file

    Returns:
        path to the saved JSON file
    """
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.basename(original_image_path)
    name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_folder, f"{name}_layout.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2, ensure_ascii=False)

    print(f"[OK] Layout saved: {output_path}")
    return output_path


def analyze_layout(processed_image, original_image_path):
    """
    Master layout analysis function — runs all steps in order.

    Steps:
        1. Detect text blocks and positions
        2. Classify blocks (header, body, footer, noise)
        3. Build structured layout dictionary
        4. Save layout as JSON

    Args:
        processed_image: numpy array (output from preprocess.py)
        original_image_path: path to original image (used for naming output)

    Returns:
        structured layout dict and path to saved JSON file
    """
    print("\n====== Phase 3: Layout Analysis ======")

    # Step 1: Detect blocks
    block_list = detect_text_blocks(processed_image)
    if not block_list:
        print("[ERROR] No text blocks detected.")
        return None, None

    # Step 2: Classify blocks
    block_list = classify_blocks(block_list)

    # Step 3: Build structured layout
    layout = build_structured_layout(block_list)

    # Step 4: Save layout
    output_path = save_layout(layout, original_image_path)

    print("====== Layout Analysis Complete ======\n")
    return layout, output_path


# -------------------------------------------------------
# -------------------------------------------------------
if __name__ == "__main__":
    from Preprocess import preprocess_image

    # Step 1: Pre-process the image first
    test_image = "input_images/test_img.png"  
    processed_image, processed_path = preprocess_image(test_image)

    if processed_image is not None:

        # Step 2: Run layout analysis
        layout, layout_path = analyze_layout(processed_image, test_image)

        if layout:
            print("=" * 50)
            print("LAYOUT PREVIEW:")
            print("=" * 50)
            if layout["headers"]:
                print(f"HEADERS ({len(layout['headers'])}):")
                for h in layout["headers"]:
                    print(f"  → {h}")
            print(f"\nBODY ({len(layout['body'])} lines):")
            for line in layout["body"][:10]:  # Show first 10 body lines
                print(f"  → {line}")
            if len(layout["body"]) > 10:
                print(f"  ... ({len(layout['body']) - 10} more lines in file)")
            if layout["footers"]:
                print(f"\nFOOTERS ({len(layout['footers'])}):")
                for f in layout["footers"]:
                    print(f"  → {f}")
            print("=" * 50)
            print(f"\nFull layout saved to: {layout_path}")

    else:
        print("[ERROR] Pre-processing failed. Cannot run layout analysis.")