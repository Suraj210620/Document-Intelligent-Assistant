import pytesseract
import cv2
import os

# Tell Python exactly where Tesseract is installed on Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text(processed_image):
    """
    Extract raw text from a preprocessed image using Tesseract OCR.
    
    Args:
        processed_image: numpy array (output from preprocess.py)
    
    Returns:
        extracted text as a string
    """
    print("\n--- Starting OCR Text Extraction ---")

    if processed_image is None:
        print("[ERROR] No image provided to OCR engine.")
        return None

    # Runs Tesseract OCR
    # lang='eng' means English language
    # --psm 6 means: assume a single uniform block of text (good for screenshots)
    text = pytesseract.image_to_string(
        processed_image,
        lang='eng',
        config='--psm 6'
    )

    if not text.strip():
        print("[WARNING] OCR returned empty text. Try a clearer image.")
        return None

    print("[OK] Text extracted successfully")
    print(f"[INFO] Total characters extracted: {len(text)}")
    print("--- OCR Extraction Complete ---\n")

    return text


def save_extracted_text(text, original_image_path, output_folder="output"):
    """
    Save the extracted text to a .txt file in the output folder.
    Filename will be: original_name_extracted.txt
    
    Args:
        text: extracted text string
        original_image_path: path to the original image (used for naming)
        output_folder: folder to save the text file
    
    Returns:
        path to the saved text file
    """
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.basename(original_image_path)
    name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_folder, f"{name}_extracted.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[OK] Extracted text saved: {output_path}")
    return output_path


def extract_text_with_details(processed_image):
    """
    Extract text along with confidence scores for each word.
    Useful later for identifying low-confidence words that may need LLM correction.
    
    Args:
        processed_image: numpy array (output from preprocess.py)
    
    Returns:
        list of dicts with word, confidence, and position info
    """
    print("\n--- Extracting Text with Confidence Scores ---")

    data = pytesseract.image_to_data(
        processed_image,
        lang='eng',
        config='--psm 6',
        output_type=pytesseract.Output.DICT
    )

    results = []
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        confidence = int(data['conf'][i])

        # Only include actual words (ignore empty strings and -1 confidence)
        if word and confidence > 0:
            results.append({
                "word": word,
                "confidence": confidence,
                "left": data['left'][i],
                "top": data['top'][i],
                "width": data['width'][i],
                "height": data['height'][i]
            })

    # Print low confidence words as a warning
    low_confidence = [r for r in results if r['confidence'] < 60]
    if low_confidence:
        print(f"[WARNING] {len(low_confidence)} low-confidence words detected:")
        for r in low_confidence:
            print(f"   Word: '{r['word']}' — Confidence: {r['confidence']}%")
    else:
        print("[OK] All words extracted with high confidence")

    print("--- Detail Extraction Complete ---\n")
    return results

if __name__ == "__main__":
    from Preprocess import preprocess_image

    # Step 1: Pre-process the image first
    test_image = "input_images/test_img.png"  
    processed_image, processed_path = preprocess_image(test_image)

    if processed_image is not None:

        # Step 2: Extract text
        text = extract_text(processed_image)

        if text:
            # Step 3: Save extracted text to file
            saved_path = save_extracted_text(text, test_image)

            # Step 4: Show extracted text in terminal
            print("=" * 50)
            print("EXTRACTED TEXT PREVIEW:")
            print("=" * 50)
            print(text[:500])  # Shows first 500 characters
            if len(text) > 500:
                print(f"... ({len(text) - 500} more characters saved in file)")
            print("=" * 50)

            # Step 5: Shows confidence details
            print("\nRunning confidence analysis...")
            details = extract_text_with_details(processed_image)
            print(f"Total words extracted: {len(details)}")

    else:
        print("[ERROR] Pre-processing failed. Cannot run OCR.")