# Hand Gesture Recognition Neural Network 🖐🤖  

Hello there, I'm **Swan**.  
I built this **Neural Network from scratch** to recognize basic hand gestures. It predicts **four gestures**:  
✅ Open Hand  
✅ Closed Hand  
✅ Thumbs Up  
✅ "F*** Off" Emote (excuse the spelling mistake 😆)  

This project uses **OpenCV** and **MediaPipe** to collect hand landmark data and train a simple neural network for classification.  

---

## **📂 Project Structure**  

This repo contains **four Python files**:  

1. **`classes_and_functions.py`**  
   - Contains all the necessary **classes and functions** for training the neural network.  

2. **`MakeDataFinal.py`**  
   - Use this script to **record more training/testing data**.  
   - Uses **OpenCV and MediaPipe** to capture **1,200 samples per run**.  
   - Just set the gesture name, and it will be **automatically saved in `data.json`**.  

3. **`NN.py`**  
   - The **core neural network training script**.  
   - Reads data from `data.json`, trains the model, and saves the trained weights and biases in **`model_weights.npz`**.  

4. **`testing.py`**  
   - Run this script to **test real-time hand gesture recognition**.  

---

## **🧠 About the Neural Network**  

This is a **lightweight feedforward neural network** with **three layers** (including the input layer).  

- **Input Layer**: 63 neurons (since there are **21 hand landmarks**, each with **x, y, z** coordinates).  
- **Hidden Layer**: 64 neurons.  
- **Output Layer**: 4 neurons (to classify the hand gestures).  

The network is **simple yet effective**, making it fast to train and test.  

---

## **🚀 How to Use**  

### **1️⃣ Install Dependencies**  
Make sure you have the required libraries installed:  
```bash
pip install numpy opencv-python mediapipe
