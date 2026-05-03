import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter


# -----------------------------
# 1) UPDATE THESE TWO PATHS
# -----------------------------
TUMOUR_DIR = "model/brain_tumor_dataset/yes"
NO_TUMOUR_DIR = "model/brain_tumor_dataset/no"

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def list_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = []
    for p in Path(folder).glob("*"):
        if p.suffix.lower() in exts:
            files.append(str(p))
    return sorted(files)


def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def simple_preprocess(gray):
    # Example preprocessing similar to your pipeline idea:
    # Otsu threshold + largest contour crop
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return cv2.resize(gray, (128, 128))

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    crop = gray[y:y+h, x:x+w]
    if crop.size == 0:
        crop = gray
    crop = cv2.resize(crop, (128, 128))
    return crop


def main():
    tumour_files = list_images(TUMOUR_DIR)
    no_tumour_files = list_images(NO_TUMOUR_DIR)

    if len(tumour_files) == 0 or len(no_tumour_files) == 0:
        print("❌ No images found. Please check TUMOUR_DIR and NO_TUMOUR_DIR paths.")
        return

    # -----------------------------
    # Fig 1: Class distribution
    # -----------------------------
    counts = {"Tumour": len(tumour_files), "No Tumour": len(no_tumour_files)}
    plt.figure(figsize=(6,4))
    plt.bar(counts.keys(), counts.values())
    plt.title("Class Distribution")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_class_distribution.png", dpi=220)
    plt.close()

    # -----------------------------
    # Fig 2: Resolution histogram
    # -----------------------------
    widths, heights = [], []
    for f in tumour_files + no_tumour_files:
        img = cv2.imread(f)
        if img is None:
            continue
        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.hist(widths, bins=20)
    plt.title("Width Distribution")
    plt.xlabel("Width (px)")
    plt.ylabel("Count")

    plt.subplot(1,2,2)
    plt.hist(heights, bins=20)
    plt.title("Height Distribution")
    plt.xlabel("Height (px)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_resolution_histogram.png", dpi=220)
    plt.close()

    # -----------------------------
    # Fig 3: Sample grid
    # -----------------------------
    sample_t = tumour_files[:6]
    sample_n = no_tumour_files[:6]
    samples = sample_t + sample_n
    labels = ["Tumour"]*len(sample_t) + ["No Tumour"]*len(sample_n)

    plt.figure(figsize=(12,5))
    for i, (fp, lb) in enumerate(zip(samples, labels), 1):
        img = read_gray(fp)
        plt.subplot(2,6,i)
        plt.imshow(img, cmap="gray")
        plt.title(lb, fontsize=8)
        plt.axis("off")
    plt.suptitle("MRI Sample Grid (Tumour vs No Tumour)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_sample_grid.png", dpi=220)
    plt.close()

    # -----------------------------
    # Fig 4: Pixel intensity histogram
    # -----------------------------
    pix = []
    for f in tumour_files[:200] + no_tumour_files[:200]:
        g = read_gray(f)
        if g is None:
            continue
        pix.extend(g.flatten().tolist())

    plt.figure(figsize=(7,4))
    plt.hist(pix, bins=50, range=(0,255))
    plt.title("Pixel Intensity Histogram")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_pixel_intensity_hist.png", dpi=220)
    plt.close()

    # -----------------------------
    # Fig 5: Before/After preprocessing
    # -----------------------------
    demo_files = tumour_files[:3] + no_tumour_files[:3]
    plt.figure(figsize=(12,6))
    for i, fp in enumerate(demo_files):
        g = read_gray(fp)
        p = simple_preprocess(g)

        plt.subplot(2,6,i+1)
        plt.imshow(g, cmap="gray")
        plt.title("Before", fontsize=8)
        plt.axis("off")

        plt.subplot(2,6,i+7)
        plt.imshow(p, cmap="gray")
        plt.title("After", fontsize=8)
        plt.axis("off")

    plt.suptitle("Before/After Preprocessing")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig5_before_after_preprocessing.png", dpi=220)
    plt.close()

    print("✅ Saved all figures in:", OUT_DIR.resolve())
    for f in sorted(OUT_DIR.glob("*.png")):
        print(" -", f.name)


if __name__ == "__main__":
    main()