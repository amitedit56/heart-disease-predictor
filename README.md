# 🫀 CardioSense — Heart Disease Predictor

<img width="1835" height="410" alt="image" src="https://github.com/user-attachments/assets/1e210913-fe29-4a76-8cd3-0e64901358e5" />
<img width="1087" height="861" alt="image" src="https://github.com/user-attachments/assets/5a047db7-e2ff-46b2-ad03-7aac644a56ee" />
<img width="1072" height="840" alt="image" src="https://github.com/user-attachments/assets/7d89d091-3fe2-4577-84c8-82d2caa25d46" />
<img width="707" height="742" alt="image" src="https://github.com/user-attachments/assets/4784c2d2-bc1a-41b8-a212-b58ac7a42b9b" />


> AI-powered cardiovascular risk assessment using 11 validated clinical markers.

🔗 **Live Demo:** [cardiosenseapp.streamlit.app](https://cardiosenseapp.streamlit.app)

---

## 📌 Overview

CardioSense is a machine learning web application that predicts the risk of heart disease based on patient clinical data. Built with **Streamlit** and powered by a **Logistic Regression** model trained on 918 patient records.

---

## 🎯 Features

- ✅ Instant heart disease risk prediction
- 📊 Probability score with visual progress bar
- 🧪 11 clinical input markers
- 💻 Clean, professional dark UI
- ⚕️ Built for clinical decision support

---

## 📈 Model Performance

| Model | Accuracy | F1 Score |
|---|---|---|
| **Logistic Regression** | **86.41%** | **88.04%** |
| KNN | 85.33% | 87.08% |
| Naive Bayes | 85.33% | 86.83% |
| SVM | 84.78% | 86.79% |
| Decision Tree | 79.35% | 81.90% |

---

## 🧬 Input Features

| Feature | Description |
|---|---|
| Age | Patient age in years |
| Sex | Male / Female |
| ChestPainType | ATA / NAP / ASY / TA |
| RestingBP | Resting blood pressure (mmHg) |
| Cholesterol | Serum cholesterol (mg/dL) |
| FastingBS | Fasting blood sugar > 120 mg/dL |
| RestingECG | Normal / ST / LVH |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise induced angina |
| Oldpeak | ST depression value |
| ST_Slope | Slope of peak exercise ST segment |

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/amitedit56/heart-disease-predictor.git
cd heart-disease-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🗂️ Project Structure

```
heart-disease-predictor/
│
├── app.py                        # Main Streamlit app
├── LogisticRegression_heart.pkl  # Trained model
├── scaler.pkl                    # StandardScaler
├── columns.pkl                   # Feature columns
├── heart_disease_models          # all models notebook file
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Streamlit** — Web framework
- **Scikit-learn** — ML model
- **Pandas & NumPy** — Data processing
- **Joblib** — Model serialization

---

## ⚠️ Disclaimer

This tool is intended to **assist clinical decision-making only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified cardiologist.

---

## 👨‍💻 Author

**Amit** — [@amitedit56](https://github.com/amitedit56)

---

⭐ **Star this repo if you found it useful!**
