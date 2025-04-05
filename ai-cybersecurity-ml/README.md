
# 🛡️ AI-Powered Cybersecurity System – Machine Learning Module (Member 1)

## 👤 Task Completed by Member 1 – Machine Learning Engineer

- Loaded and preprocessed the NSL-KDD dataset.
- Trained a Random Forest model to classify network traffic as `normal` or `attack`.
- Saved the trained model and scaler to the `models/` directory.
- Built a Flask API (`scripts/api.py`) to serve predictions using the trained model.
- Tested the prediction API with Postman.

---

## 🧰 Project Setup

### 🔧 Requirements

- Python 3.9+
- Libraries:
  - pandas
  - scikit-learn
  - Flask
  - joblib
  - numpy

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the ML Pipeline

### 🛠 Preprocess & Train the Model

```bash
python scripts/preprocess_and_train.py
```

### 🌐 Run the API

```bash
python scripts/api.py
```

Your API will run at:
🔗 http://127.0.0.1:5000/predict

---

### 🧪 Test with Postman

1. Open Postman and select `POST` request
2. Enter URL: `http://127.0.0.1:5000/predict`
3. Go to **Body** → select **raw** → choose **JSON**
4. Paste this sample data:

```json
{
  "features": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
}
```

📨 Response:
```json
{
  "prediction": "attack"
}
```

---

✅ **Task Completion – Member 1 Done**  
You can now share this with Member 2 (Network Security & Threat Detection)

---

## 📁 Project Structure

```
ai-cybersecurity-ml/
├── datasets/            ← NSL-KDD dataset
├── models/              ← Trained model and scaler
├── scripts/
│   ├── preprocess_and_train.py
│   └── api.py
├── requirements.txt
└── README.md
```
