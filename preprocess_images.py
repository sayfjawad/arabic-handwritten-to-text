import os
import cv2

INPUT_DIR = "images"
OUTPUT_DIR = "processed_images"

# Start conservative. This avoids huge visual token counts.
MAX_SIDE = 1000

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")):
        continue

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename.rsplit(".", 1)[0] + "_processed.png")

    img = cv2.imread(input_path)
    if img is None:
        print(f"Could not read: {input_path}")
        continue

    h, w = img.shape[:2]
    print(f"Original {filename}: {w}x{h}")

    scale = min(1.0, MAX_SIDE / max(h, w))

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized {filename}: {new_w}x{new_h}")
    else:
        print(f"No resize needed for {filename}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mild denoise
    gray = cv2.fastNlMeansDenoising(gray, None, h=5)

    # Mild contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    cv2.imwrite(output_path, gray)
    print(f"Saved: {output_path}")
