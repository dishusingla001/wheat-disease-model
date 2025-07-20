import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r'D:\Wheat-diesease\wheat-disease-model\wheat-disease-detection\cropDiseaseDataset\Wheat_crown_root_rot\00051.jpg'

def segment_units_and_calculate_areas(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjust this range for your specific disease color
    lower_disease = np.array([10, 50, 50])
    upper_disease = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower_disease, upper_disease)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours → these are your diseased units
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = mask_clean.shape[0] * mask_clean.shape[1]
    unit_areas = []
    output_image = image.copy()

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 50:  # ignore very tiny spots
            unit_areas.append(area)
            cv2.drawContours(output_image, [contour], -1, (0, 0, 255), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(output_image, f'{i+1}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)

    # Calculate total affected % area
    total_disease_area = sum(unit_areas)
    percent_affected = (total_disease_area / total_area) * 100

    print(f"🟠 Number of diseased units: {len(unit_areas)}")
    print(f"🔴 Total affected area: {percent_affected:.2f}%")

    # Show results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("Segmented Units")
    plt.show()

    return len(unit_areas), percent_affected

segment_units_and_calculate_areas(image_path)