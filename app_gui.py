import tkinter as tk
from tkinter import *
import PIL
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import cv2  # NEW

# Load the trained model
model = load_model("mnist_model.h5")

# Constants
CANVAS_SIZE = 400  # Bigger canvas for easier drawing
MODEL_INPUT_SIZE = 28

# Create window
window = Tk()
window.title("Digit Recognizer")

# Create canvas to draw
canvas = Canvas(window, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white')
canvas.grid(row=0, column=0, pady=2, sticky=W)

# Initialize PIL Image to draw on
image1 = PIL.Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 'white')
draw = ImageDraw.Draw(image1)

# Function to draw lines
def paint(event):
    x1, y1 = (event.x - 12), (event.y - 12)
    x2, y2 = (event.x + 12), (event.y + 12)
    canvas.create_oval(x1, y1, x2, y2, fill='black')
    draw.ellipse([x1, y1, x2, y2], fill='black')

canvas.bind("<B1-Motion>", paint)

# Function to clear canvas
def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill='white')
    label.config(text="Draw a digit and click Predict")

# Function to predict
def predict():
    # Convert image to numpy array
    img = np.array(image1)

    # Invert colors (black digit on white background)
    img = cv2.bitwise_not(img)

    # Find bounding box of the digit
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = img[y:y+h, x:x+w]

        # Resize while preserving aspect ratio
        resized_digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

        # Create a blank 28x28 image and paste resized digit into center
        final_img = np.ones((28, 28), dtype=np.uint8) * 255
        x_offset = (28 - 20) // 2
        y_offset = (28 - 20) // 2
        final_img[y_offset:y_offset+20, x_offset:x_offset+20] = resized_digit

        # Normalize and reshape for prediction
        final_img = final_img.astype("float32") / 255.0
        final_img = final_img.reshape(1, 28, 28, 1)

        # Predict
        result = model.predict(final_img)
        predicted_digit = np.argmax(result)
        label.config(text=f"Predicted Digit: {predicted_digit}")
    else:
        label.config(text="No digit detected")


# Buttons and Labels
btn_predict = Button(window, text="Predict", command=predict)
btn_predict.grid(row=1, column=0, pady=2)

btn_clear = Button(window, text="Clear", command=clear)
btn_clear.grid(row=2, column=0, pady=2)

label = Label(window, text="Draw a digit and click Predict", font=("Helvetica", 14))
label.grid(row=3, column=0, pady=2)

# Start GUI
window.mainloop()
