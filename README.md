# Handwritten Digit Recognition with Deep Learning 🧠✍️

This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset and allows users to test their own digits using a simple GUI.

## 🚀 Features

- Trains a CNN model on the MNIST dataset
- GUI-based testing for real-time digit prediction
- Uses TensorFlow, Keras, and Tkinter

## 🖥️ Files Overview
```bash
 `model_train.py`       Contains the model architecture, training logic, and saves the trained model as `mnist_model.h5`. 
 `app_gui.py`           A graphical user interface where you can draw a digit and get real-time predictions. 
 `requirements.txt`     Lists all required Python packages.
 `README.md`            Project documentation and description.
```
## 📝 How to Use

1. **Install Dependencies**
```bash
   pip install -r requirements.txt
```
2. **Train the Model (if needed)**
```bash
    python model_train.py
```
3. **Run the Digit Recognition GUI**
```bash
    python app_gui.py
```
4. Draw any digit (0–9) in the canvas area and click Predict!

⚠️ Notes

 The model may occasionally misclassify certain digits like 1,7 especially if drawn ambiguously — this is expected behavior for models trained on real-world handwritten data.

 Performance can be improved with more data, augmentation, or tuning, but this version is optimized for simplicity and ease of use.

 ## 🖼️ Screenshots


![GUI Interface](Screenshot%202025-06-07%20150421.png)


![Prediction](Screenshot%202025-06-07%20150535.png)


![Preview](Screenshot%202025-06-07%20150555.png)

## 📜 License

This project is licensed under the [MIT License](LICENSE).


