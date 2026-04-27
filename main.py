import cv2
import numpy as np
import os

def detect_algae_impurities(image_path, sample_type, index, output_dir, roi_fraction=0.98):
    # Step 1: Read the image
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Step 2: ROI (Region of Interest)
    height, width = img.shape[:2]
    roi_height, roi_width = int(height * roi_fraction), int(width * roi_fraction)
    start_x, start_y = (width - roi_width) // 2, (height - roi_height) // 2
    cropped_img = img[start_y:start_y + roi_height, start_x:start_x + roi_width]
    
    # Create a grayscale background with BGR channels
    gray_bg = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    display_img = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2BGR)

    # Step 3: Pre-processing for detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray_bg)

    # Step 4: Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        enhanced_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 4)

    # Step 5: Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

    # Step 6: Contour Detection
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area: 40 to 5000 pixels
    valid_contours = [cnt for cnt in contours if 40 < cv2.contourArea(cnt) < 5000]

    # Step 7: Drawing Circular Annotations
    for cnt in valid_contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius) + 2 
        cv2.circle(display_img, center, radius, (0, 255, 0), 2)

    # Step 8: Calculate Status and Overlay Text
    total_count = len(valid_contours)
    is_drinkable = total_count < 20
    status_text = "Drinkable" if is_drinkable else "NOT Drinkable"
    status_color = (0, 255, 0) if is_drinkable else (0, 0, 255)

    cv2.putText(display_img, f"Status: {status_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(display_img, f"Impurities: {total_count}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # --- ADDED: SAVE THE OUTPUT IMAGE ---
    # Ensure filename is unique based on type and index
    save_name = f"result_{sample_type.lower().replace(' ', '_')}_{index}.jpg"
    cv2.imwrite(os.path.join(output_dir, save_name), display_img)
    # ------------------------------------

    cv2.imshow("Detection Result", display_img)
    cv2.waitKey(500) 

    return total_count, status_text


# --- FILE EXECUTION ---
sample_directories = {
    "River Water": r"C:\Users\mahes\OneDrive\Pictures\Documents\algae_detection\river_water",
    "Drinking Water": r"C:\Users\mahes\OneDrive\Pictures\Documents\algae_detection\drinking_water"
}

# --- ADDED: CREATE OUTPUT FOLDER ---
output_folder = r"C:\Users\mahes\OneDrive\Pictures\Documents\algae_detection\detection_results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# -----------------------------------

for sample_type, folder_path in sample_directories.items():
    if not os.path.exists(folder_path):
        print(f"Directory Error: {folder_path} not found.")
        continue

    counts = []
    print(f"\nProcessing {sample_type} samples...")
    file_prefix = sample_type.lower().replace(" ", "_")

    for i in range(1, 11):
        filename = f"{file_prefix}_{i}.jpg"
        image_path = os.path.join(folder_path, filename)

        # Updated to pass the required arguments for saving
        result = detect_algae_impurities(image_path, sample_type, i, output_folder)

        if result:
            count, status = result
            counts.append(count)
            print(f"File: {filename} | Detected: {count} | Status: {status}")
        else:
            print(f"File: {filename} | Error: Could not read image.")

    if counts:
        avg = np.mean(counts)
        overall = "Drinkable" if avg < 20 else "Not Drinkable"
        print("-" * 30)
        print(f"Results for {sample_type}:")
        print(f"Average Impurities: {avg:.2f}")
        print(f"Overall Recommendation: {overall}")
        print(f"Processed images saved to: {output_folder}")
        print("-" * 30)

cv2.destroyAllWindows()