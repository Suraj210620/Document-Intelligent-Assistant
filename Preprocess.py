import cv2
import numpy as np
import os

def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return None

    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return None

    print(f"[OK] Image loaded: {image_path}")
    return image


def convert_to_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("[OK] Converted to grayscale")
    return gray


def apply_binarization(gray_image):
    """
    Applying Otsu's thresholding to convert grayscale to pure black & white.
    This makes text sharper and easier for OCR to read.
    """
    _, binary = cv2.threshold(
        gray_image, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print("[OK] Binarization applied (Otsu's threshold)")
    return binary


def reduce_noise(binary_image):
    """
    Applying morphological operations to remove small noise dots.
    Keeps text intact while cleaning up background speckles.
    """
    kernel = np.ones((1, 1), np.uint8)
    denoised = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    denoised = cv2.medianBlur(denoised, 3)
    print("[OK] Noise reduction applied")
    return denoised


def save_processed_image(image, original_path, output_folder="output"):
    """
    Here we save the processed image to the output folder.

    """
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.basename(original_path)
    name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_folder, f"{name}_processed.png")

    cv2.imwrite(output_path, image)
    print(f"[OK] Processed image saved: {output_path}")
    return output_path


def preprocess_image(image_path):
    """
    Master pre-processing function — runs all steps in order.
    
    Steps:
        1. Load image
        2. Convert to grayscale
        3. Apply binarization
        4. Reduce noise
        5. Save processed image

    Returns the processed image (numpy array) and the saved output path.
    """
    print("\n--- Starting Image Pre-processing ---")

    # Step 1: Load
    image = load_image(image_path)
    if image is None:
        return None, None

    # Step 2: Grayscale
    gray = convert_to_grayscale(image)

    # Step 3: Binarization
    binary = apply_binarization(gray)

    # Step 4: Noise reduction
    processed = reduce_noise(binary)

    # Step 5: Save
    output_path = save_processed_image(processed, image_path)

    print("--- Pre-processing Complete ---\n")
    return processed, output_path

if __name__ == "__main__":
    test_image = "input_images/test_img.png"  
    processed_image, saved_path = preprocess_image(test_image)

    if processed_image is not None:
        print(f"Success! Processed image saved to: {saved_path}")
    else:
        print("Pre-processing failed. Check the error messages above.")