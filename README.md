# WORKSHOP--4-Coin-Detection-using-OpenCV-in-Python
# SEC-DIP – Coin Detection using OpenCV in Python

## Name and Register Number

**Name:** C J Rohit
**Reg. No:** 212224243005

---

## Aim

To develop an image processing system that can automatically detect and count coins in an image using **Python and OpenCV**, while visualizing the intermediate image-processing steps such as grayscale conversion, blurring, thresholding, morphological operations, and coin detection.

---

## Objectives

1. To apply fundamental computer vision techniques to identify circular objects such as coins.
2. To understand image preprocessing and feature extraction techniques using OpenCV.
3. To visualize the intermediate processing results involved in coin detection.
4. To detect and count the number of coins using **Blob Detection** and **Contour Detection** methods.
5. To compare the results obtained from both detection methods.

---

## Software Required

* Python 3.x
* OpenCV (`cv2`)
* NumPy
* Matplotlib
* Jupyter Notebook

---

## Input Image

**CoinsA.png**

The input image contains multiple coins that are detected using image processing techniques.

---

## Algorithm

### Step 1: Start

Import the required Python libraries such as OpenCV, NumPy, and Matplotlib.

### Step 2: Input Image

Load the coin image using `cv2.imread()`.

### Step 3: Grayscale Conversion

Convert the input image into grayscale to simplify image processing and reduce the number of colour channels.

### Step 4: Channel Selection

Examine the individual colour channels and select the channel that provides better separation between the coins and the background.

### Step 5: Image Smoothing

Apply Gaussian Blur to reduce noise and small intensity variations before thresholding.

### Step 6: Thresholding

Apply thresholding to separate the coin regions from the background.

Otsu's thresholding method is used to automatically determine a suitable threshold value.

### Step 7: Morphological Operations

Apply morphological operations to clean the thresholded image.

* **Opening:** Removes small noise and unwanted foreground objects.
* **Closing:** Fills small gaps and holes and improves the continuity of coin regions.
* **Dilation:** Expands foreground regions where required.
* **Erosion:** Removes small unwanted regions and refines the detected objects.

### Step 8: Blob Detection

Use OpenCV's `SimpleBlobDetector` to identify coin-like circular regions based on properties such as circularity, convexity, and inertia.

### Step 9: Contour Detection

Find contours from the processed binary image and filter them based on area and circularity.

### Step 10: Draw Detections

Draw the detected coin regions and assign numbers to the detected coins.

### Step 11: Count Coins

Calculate and display the total number of coins detected using both Blob Detection and Contour Detection.

### Step 12: Compare Results

Compare the number of coins detected by both methods and discuss their advantages and limitations.

### Step 13: End

---

## Program

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the input image
image = cv2.imread("CoinsA.png")

# Check image
if image is None:
    raise FileNotFoundError("CoinsA.png not found.")

# Convert to RGB for displaying
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Split colour channels
B, G, R = cv2.split(image)

# Select Blue channel
selected_channel = B

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(selected_channel, (5, 5), 0)

# Apply Otsu Thresholding
threshold_value, binary = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Morphological operations
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)

closing = cv2.morphologyEx(
    opening,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

# ------------------------------------------------
# BLOB DETECTION
# ------------------------------------------------

params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0
params.minDistBetweenBlobs = 2

params.filterByArea = False

params.filterByCircularity = True
params.minCircularity = 0.80

params.filterByConvexity = True
params.minConvexity = 0.80

params.filterByInertia = True
params.minInertiaRatio = 0.80

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(closing)

blob_count = len(keypoints)

blob_output = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    (0, 255, 0),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

print("Number of coins detected using Blob Detection:",
      blob_count)

# ------------------------------------------------
# CONTOUR DETECTION
# ------------------------------------------------

contours, hierarchy = cv2.findContours(
    closing,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

valid_contours = []

for contour in contours:

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        continue

    circularity = (
        4 * np.pi * area
    ) / (perimeter * perimeter)

    if area > 1000 and circularity > 0.65:
        valid_contours.append(contour)

contour_count = len(valid_contours)

contour_output = image.copy()

cv2.drawContours(
    contour_output,
    valid_contours,
    -1,
    (0, 255, 0),
    3
)

# Add serial numbers
for i, contour in enumerate(valid_contours, 1):

    M = cv2.moments(contour)

    if M["m00"] != 0:

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.putText(
            contour_output,
            str(i),
            (cx - 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

print("Number of coins detected using Contour Detection:",
      contour_count)

# ------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------

plt.figure(figsize=(16, 10))

plt.subplot(2, 4, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(selected_channel, cmap="gray")
plt.title("Selected Blue Channel")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(binary, cmap="gray")
plt.title("Otsu Threshold")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(opening, cmap="gray")
plt.title("Morphological Opening")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(closing, cmap="gray")
plt.title("Morphological Closing")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(cv2.cvtColor(blob_output, cv2.COLOR_BGR2RGB))
plt.title(f"Blob Detection: {blob_count}")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.imshow(cv2.cvtColor(contour_output, cv2.COLOR_BGR2RGB))
plt.title(f"Contour Detection: {contour_count}")
plt.axis("off")

plt.tight_layout()
plt.show()
```

---

## Channel Selection

The individual **Blue, Green, and Red channels** are examined before segmentation.

The **Blue channel** is selected because it provides useful contrast between the metallic coins and the red-coloured background. This makes the coin regions easier to separate during thresholding.

---

## Thresholding

**Otsu's thresholding** is used to convert the selected channel into a binary image.

Otsu's method automatically determines a threshold value from the image histogram.

The binary image separates the foreground coin regions from the background, which makes subsequent morphological processing and object detection easier.

---

## Morphological Operations

### Opening

Opening consists of:

**Erosion → Dilation**

It is used to remove small isolated noise and unwanted foreground structures.

### Closing

Closing consists of:

**Dilation → Erosion**

It is used to fill small holes and connect small gaps within the detected coin regions.

A **5 × 5 elliptical kernel** is used because the objects being detected are approximately circular.

---

## Blob Detection

The `SimpleBlobDetector` detects candidate coin regions based on geometric properties.

The detector uses:

* Circularity
* Convexity
* Inertia
* Distance between blobs

The number of detected blobs is printed by the program.

**Blob Detection Count:** `__________ coins`

---

## Contour Detection

Contours are extracted from the morphologically processed binary image.

Contours are filtered using:

* Area
* Circularity

Small regions caused by noise are removed, while larger circular regions corresponding to coins are retained.

**Contour Detection Count:** `__________ coins`

---

## Findings and Observations

### Observation 1 – Grayscale

The colour image is converted to grayscale to simplify processing and reduce the image to a single intensity channel.

### Observation 2 – Channel Selection

The Blue channel provides useful contrast between the coins and the background and is therefore selected for segmentation.

### Observation 3 – Thresholding

Otsu thresholding produces a binary representation that separates the major coin regions from the background.

### Observation 4 – Morphological Processing

Opening reduces small unwanted regions, while closing fills gaps and improves the continuity of the detected coin regions.

### Observation 5 – Blob Detection

Blob Detection identifies coin-like regions using geometric properties. It can provide centre points and approximate sizes of the detected blobs.

### Observation 6 – Contour Detection

Contour Detection identifies the boundaries of connected regions. It provides actual object boundaries but may merge touching coins into a single contour.

---

## Comparison of Detection Methods

| Feature                | Blob Detection                       | Contour Detection                   |
| ---------------------- | ------------------------------------ | ----------------------------------- |
| Detection basis        | Blob properties                      | Object boundaries                   |
| Uses circularity       | Yes                                  | Yes                                 |
| Uses area              | Optional                             | Yes                                 |
| Provides boundary      | Approximate                          | Yes                                 |
| Handles isolated coins | Good                                 | Good                                |
| Touching coins         | May separate depending on parameters | May merge into one contour          |
| Output                 | Keypoints                            | Contours                            |
| Main advantage         | Simple detection of coin-like blobs  | Provides detailed object boundaries |

---

## Final Results

After running all cells, record the actual counts printed by the notebook:

```text
Blob Detection Count    : ______ coins
Contour Detection Count : ______ coins
```

The two methods may produce different counts because they use different detection principles and are sensitive to thresholding, morphology, object overlap, and parameter settings.

---
##OUTPUT
<img width="361" height="409" alt="download" src="https://github.com/user-attachments/assets/c6401311-612c-4782-b130-84c04045dc6f" />


## Conclusion

The coin detection system was successfully implemented using **Python and OpenCV**. The image was processed through grayscale conversion, channel selection, Gaussian smoothing, thresholding, and morphological operations before performing coin detection.

Two approaches were implemented:

1. **Blob Detection**
2. **Contour Detection**

Blob Detection identifies coin-like regions using geometric properties, whereas Contour Detection identifies connected object boundaries.

The final coin counts should be obtained from the executed notebook output. Differences between the two counts can occur because touching coins may form connected regions, while noise and thresholding can also affect the detected objects.

Therefore, appropriate preprocessing and parameter selection are important for obtaining reliable coin detection results.

---

## Developed By

**C J Rohit**
**Reg. No: 212224243005**

---

## Result

Thus, coin detection and counting were successfully performed using **Blob Detection and Contour Detection** techniques in OpenCV, with all important intermediate image-processing stages visualized and documented.
