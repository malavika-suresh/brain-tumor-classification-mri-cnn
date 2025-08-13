from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt

# --------------------------
# Crop brain contour function
# --------------------------
def crop_brain_contour(image, plot=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    if plot:
        plt.figure(figsize=(8,4))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title('Cropped Image')

        plt.show()

    return new_image


# --------------------------
# Load the trained model
# --------------------------
best_model = load_model(filepath='cnn-parameters-improvement-24-0.86.model')
IMG_WIDTH, IMG_HEIGHT = (240, 240)
image_width, image_height = (IMG_WIDTH, IMG_HEIGHT)

# --------------------------
# Read and preprocess image
# --------------------------
image = cv2.imread(r'brain_tumor_dataset\yes\2Y.jpg')
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
threshdup = cv2.dilate(thresh, None, iterations=2)
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(threshdup, kernel, iterations=1)
opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# --------------------------
# Crop & resize for model
# --------------------------
image_cropped = crop_brain_contour(image, plot=False)
image_resized = cv2.resize(image_cropped, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
image_resized = image_resized / 255.0

# --------------------------
# Prediction
# --------------------------
tumor_prob = best_model.predict(image_resized.reshape(1, 240, 240, 3))
print("Tumor Probability:", tumor_prob)

output = image.copy()

# --------------------------
# Detection & visualization
# --------------------------
if tumor_prob > 0.60:
    cv2.putText(output, "Brain Tumour Detected", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 2000:
            threshold = cv2.rectangle(img_gray.copy(), (x, y), (x + w, y + h), (0, 0, 255), 2)
            threshold1 = cv2.rectangle(opening.copy(), (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            crop = threshold1[y:y+h, x:x+w]

            # Show using matplotlib
            plt.figure(figsize=(8,4))
            plt.subplot(1, 2, 1)
            plt.imshow(threshold, cmap='gray')
            plt.title("Threshold")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(crop, cmap='gray')
            plt.title("Cropped Threshold")
            plt.axis("off")

            plt.show()

else:
    cv2.putText(output, "Normal", (60, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# --------------------------
# Final output visualization
# --------------------------
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detection Result")
plt.axis("off")
plt.show()
