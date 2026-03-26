from Preprocess import preprocess_image
from OCR import extract_text, save_extracted_text
from layout import analyze_layout
from LLM_processor import refine_with_llm
def run_pipeline(image_path):
    # Stage 1: Pre-process the image
    processed_image, processed_path = preprocess_image(image_path)
    
    if processed_image is None:
        print("[ERROR] Pipeline stopped — pre-processing failed.")
        return
    
    # Stage 2: Extract text using OCR
    text = extract_text(processed_image)
    
    if text is None:
        print("[ERROR] Pipeline stopped — OCR extraction failed.")
        return
    
    saved_path = save_extracted_text(text, image_path)
    print(f"Pipeline Stage 2 complete. Extracted text saved at: {saved_path}")

    # Stage 3: Analyze layout
    layout_info, layout_path = analyze_layout(processed_image, image_path)
    if layout_info is None:
        print("[ERROR] Pipeline stopped — layout analysis failed.")
        return
    
    print(f"Pipeline Stage 3 complete. Layout saved at: {layout_path}")

    # Stage 4: Refine with LLM
    result, result_path = refine_with_llm(text, layout_info, image_path)
    if result is None:
        print("[ERROR] Pipeline stopped — LLM processing failed.")
        return
    print(f"Pipeline Stage 4 complete. LLM result saved at: {result_path}")

    # Summary
    print("\nPipeline completed successfully!")
    print(f"Processed Image: {processed_path}")
    print(f"Extracted Text: {saved_path}")
    print(f"Layout Info: {layout_path}")
    print(f"LLM Result: {result_path}")
    print(f"Final Result: {result}")

# Run the pipeline
if __name__ == "__main__":
    run_pipeline("input_images/test_img.png")  