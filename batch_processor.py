import os
import json
import time
from pathlib import Path

from Preprocess import preprocess_image
from OCR import extract_text, save_extracted_text
from layout import analyze_layout
from LLM_processor import refine_with_llm


def get_image_files(folder_path):
    """
    Scan a folder and return all supported image file paths.
    Supports PNG, JPG, JPEG, BMP, TIFF formats.

    Args:
        folder_path: path to folder containing images

    Returns:
        list of image file paths found in the folder
    """
    supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []

    folder = Path(folder_path)

    if not folder.exists():
        print(f"[ERROR] Folder not found: {folder_path}")
        return []

    for file in sorted(folder.iterdir()):
        if file.suffix.lower() in supported_formats:
            image_files.append(str(file))

    print(f"[OK] Found {len(image_files)} image(s) in: {folder_path}")
    for img in image_files:
        print(f"   → {os.path.basename(img)}")

    return image_files


def process_single_image(image_path, output_folder="output"):
    """
    Run the full pipeline on a single image.
    Calls all 4 phases in order: preprocess → OCR → layout → LLM.

    Args:
        image_path: path to the image file
        output_folder: folder to save all output files

    Returns:
        dict with status and result details for this image
    """
    filename = os.path.basename(image_path)
    print(f"\n{'='*55}")
    print(f"Processing: {filename}")
    print(f"{'='*55}")

    result_summary = {
        "file": filename,
        "status": "failed",
        "output_files": [],
        "error": None
    }

    try:
        # Phase 1: Pre-processing
        processed_image, processed_path = preprocess_image(image_path)
        if processed_image is None:
            result_summary["error"] = "Pre-processing failed"
            return result_summary
        result_summary["output_files"].append(processed_path)

        # Phase 2: OCR text extraction
        text = extract_text(processed_image)
        if text is None:
            result_summary["error"] = "OCR extraction failed"
            return result_summary
        saved_text_path = save_extracted_text(text, image_path, output_folder)
        result_summary["output_files"].append(saved_text_path)

        # Phase 3: Layout analysis
        layout, layout_path = analyze_layout(processed_image, image_path)
        if layout is None:
            result_summary["error"] = "Layout analysis failed"
            return result_summary
        result_summary["output_files"].append(layout_path)

        # Phase 4: LLM post-processing
        result, result_path = refine_with_llm(text, layout, image_path)
        if result is None:
            result_summary["error"] = "LLM processing failed"
            return result_summary
        result_summary["output_files"].append(result_path)

        # Mark as successful
        result_summary["status"] = "success"
        result_summary["document_type"] = result.get("document_type", "unknown")
        result_summary["summary"] = result.get("summary", "")
        result_summary["confidence"] = result.get("confidence", "unknown")

        print(f"[OK] Successfully processed: {filename}")

    except Exception as e:
        result_summary["error"] = str(e)
        print(f"[ERROR] Failed to process {filename}: {e}")

    return result_summary


def save_batch_report(batch_results, output_folder="output"):
    """
    Save a summary report of the entire batch processing run.
    Includes status, timing, and results for every image processed.

    Args:
        batch_results: list of result dicts from process_single_image()
        output_folder: folder to save the report

    Returns:
        path to the saved batch report JSON file
    """
    os.makedirs(output_folder, exist_ok=True)
    report_path = os.path.join(output_folder, "batch_report.json")

    # Calculate summary stats
    total = len(batch_results)
    successful = sum(1 for r in batch_results if r["status"] == "success")
    failed = total - successful

    report = {
        "batch_summary": {
            "total_images": total,
            "successful": successful,
            "failed": failed,
            "success_rate": f"{round((successful / total) * 100)}%" if total > 0 else "0%"
        },
        "results": batch_results
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Batch report saved: {report_path}")
    return report_path


def print_batch_summary(batch_results, total_time):
    """
    Print a clean summary table of the batch processing results.

    Args:
        batch_results: list of result dicts from process_single_image()
        total_time: total processing time in seconds
    """
    total = len(batch_results)
    successful = sum(1 for r in batch_results if r["status"] == "success")
    failed = total - successful

    print(f"\n{'='*55}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*55}")
    print(f"Total images processed : {total}")
    print(f"Successful             : {successful}")
    print(f"Failed                 : {failed}")
    print(f"Success rate           : {round((successful / total) * 100)}%" if total > 0 else "0%")
    print(f"Total time             : {round(total_time, 2)} seconds")
    print(f"Avg time per image     : {round(total_time / total, 2)} seconds" if total > 0 else "N/A")
    print(f"{'='*55}")

    print("\nPer Image Results:")
    print(f"{'='*55}")
    for r in batch_results:
        status_icon = "✓" if r["status"] == "success" else "✗"
        print(f"  {status_icon} {r['file']}")
        if r["status"] == "success":
            print(f"      Type       : {r.get('document_type', 'N/A')}")
            print(f"      Confidence : {r.get('confidence', 'N/A')}")
            print(f"      Summary    : {r.get('summary', 'N/A')[:60]}...")
        else:
            print(f"      Error      : {r.get('error', 'Unknown error')}")
    print(f"{'='*55}\n")


def run_batch(input_folder="batch_input", output_folder="output", delay=1):
    """
    Master batch processing function.
    Processes all images in a folder through the full pipeline.

    Args:
        input_folder: folder containing images to process
        output_folder: folder to save all output files
        delay: seconds to wait between images (avoids API rate limiting)

    Returns:
        list of result dicts for all processed images
    """
    print("\n" + "="*55)
    print("DOCUMENT INTELLIGENT ASSISTANT — BATCH MODE")
    print("="*55)

    # Step 1: Get all image files
    image_files = get_image_files(input_folder)

    if not image_files:
        print("[ERROR] No images found in folder. Please add images and try again.")
        return []

    print(f"\n[INFO] Starting batch processing of {len(image_files)} image(s)...")
    print(f"[INFO] Output folder: {output_folder}\n")

    # Step 2: Process each image
    batch_results = []
    start_time = time.time()

    for index, image_path in enumerate(image_files, start=1):
        print(f"\n[{index}/{len(image_files)}] Starting next image...")

        result = process_single_image(image_path, output_folder)
        batch_results.append(result)

        # Small delay between images to avoid overwhelming the API
        if index < len(image_files):
            print(f"[INFO] Waiting {delay}s before next image...")
            time.sleep(delay)

    total_time = time.time() - start_time

    # Step 3: Save batch report
    save_batch_report(batch_results, output_folder)

    # Step 4: Print summary
    print_batch_summary(batch_results, total_time)

    return batch_results



# Running batch processing
if __name__ == "__main__":
    run_batch(
        input_folder="batch_input",   
        output_folder="output",        # folder for all results
        delay=1                        # seconds between images
    )