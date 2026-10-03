# 🌾 Crop Suitability Classification System

A multi-class machine-learning project that predicts the most suitable crop (22 classes) from soil nutrients, soil pH and climatic conditions, with an interactive Streamlit web application.

**Author:**        Harischandra Prasad Vissamsetti, M.C.A
**Program:**       Final Capstone Project (Track-1)
**Project Name:**  Crop Suitability Classification System

---

## 📌 Overview

Choosing a crop that matches local soil and climate is a core agronomic decision. This project builds a zero-cost, reproducible classifier that predicts the most suitable crop for a new observation and reports the model's confidence and the top-5 crop probabilities.

| Item | Details |
|---|---|
| Task | Multi-class classification (22 crops) |
| Inputs | Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, rainfall |
| Output | Predicted crop, confidence level, top-5 probabilities |
| Models compared | Logistic Regression, K-Nearest Neighbours, Decision Tree |
| Selected model | Logistic Regression |
| Tools | Python, scikit-learn, pandas, matplotlib, seaborn, Streamlit (all free and open source) |

## 📊 Dataset

The project uses the **Crop Recommendation Dataset** from Kaggle (real, non-synthetic), stored as `data/crop_data.csv`.

- **Source:** Ingle, A. *Crop Recommendation Dataset.* Kaggle. https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
- **Size:** 2,200 records, 8 columns, 22 crops with 100 samples each (perfectly balanced)
- **Data quality:** no missing values and no duplicate rows were found; the pipeline still handles both
- **Columns:** `N, P, K, temperature, humidity, ph, rainfall, label`

| Feature | Meaning | Unit |
|---|---|---|
| N, P, K | Soil nitrogen, phosphorus, potassium | kg/ha |
| temperature | Average temperature | °C |
| humidity | Relative humidity | % |
| ph | Soil pH | - |
| rainfall | Rainfall | mm |

> The dataset was compiled from agricultural data for India. Values are realistic but are not raw field-by-field measurements, so accuracy on noisy real-world data is untested.

License: The dataset is the Crop Recommendation Dataset by Atharva Ingle, obtained from Kaggle and distributed under the Apache License 2.0. The file was renamed to crop_data.csv; its contents are unmodified. A copy of the license is included in data/DATA_LICENSE.txt. This project is not affiliated with or endorsed by the dataset's author.

## 📁 Project Structure

```
crop_capstone/
├── app/
│   └── streamlit_app.py        # Streamlit web application
├── data/
│   └── crop_data.csv           # Kaggle Crop Recommendation Dataset
├── models/
│   ├── crop_model.joblib       # Trained pipeline (impute -> scale -> model)
│   └── metadata.json           # Classes, feature ranges, model info
├── notebooks/
│   ├── crop_suitability.ipynb  # Notebook version of the analysis (data, EDA, models, evaluation)
│   └── colab_run.ipynb         # Colab runner (runs the scripts and shows results) 
├── reports/
│   ├── figures/                # EDA and evaluation charts
│   ├── metrics.json            # All metrics
│   ├── classification_report.txt
│   ├── eda_summary.txt
│   
├── src/
│   ├── preprocessing.py        # Shared cleaning + pipeline (training and app)
│   ├── eda.py                  # Exploratory data analysis
│   └── train.py                # Tuning, evaluation, error analysis, model saving
├── requirements.txt
└── README.md
```

## 🔬 Methodology

1. **Data quality checks:** missing values, duplicates, impossible values, class balance.
2. **Cleaning:** drop duplicates, null out physically impossible values, median-impute.
3. **Split first:** stratified 80/20 train/test split before any fitting, so there is no data leakage.
4. **Pipeline:** imputer and scaler live inside a scikit-learn `Pipeline`, fitted on training data only.
5. **Tuning:** grid search with stratified 5-fold cross-validation on macro-F1.
6. **Selection:** by cross-validated score only, never by test score. Models within 0.01 CV macro-F1 of the best are treated as comparable and the simplest is preferred.
7. **Evaluation:** accuracy, precision, recall, F1, confusion matrix, one-vs-rest ROC-AUC, permutation importance and error analysis.

## 📈 Results (20% hold-out test set, 440 samples)

| Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|
| **Logistic Regression (selected)** | **98.4%** | **0.984** | **0.9999** |
| KNN | 98.2% | 0.982 | 0.9987 |
| Decision Tree | 97.9% | 0.979 | 0.9893 |

Charts and per-crop details are in `reports/`.

---

## 💻 Run in VS Code

**Prerequisites:** Python 3.10 or newer and VS Code with the **Python** and **Jupyter** extensions.

**1. Get the project**
```bash
git clone  https://github.com/Harischandra-Prasad-Vissamsetti/Crop-Suitability-Classification.git
cd <Crop-Suitability-Classification>
```
Open the folder in VS Code (**File → Open Folder**). It must be the project root.

**2. Create and activate a virtual environment**

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
In VS Code press `Ctrl+Shift+P`, run **Python: Select Interpreter**, and choose the `.venv` one.

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the pipeline** (always from the project root)
```bash
python src/eda.py
python src/train.py
```
This creates the charts and metrics in `reports/` and saves the model to `models/`.

**5. Launch the app**
```bash
streamlit run app/streamlit_app.py
```
Open http://localhost:8501 if the browser does not open automatically.

**6. Notebook (optional):** open `notebooks/crop_suitability.ipynb`, select the `.venv` kernel and click **Run All**.

---

## ☁️ Run in Google Colab

**1. Open a notebook** at https://colab.research.google.com and click **New notebook**.

**2. Get the project** (choose one)
```python
# Option A: clone from GitHub
!git clone https://github.com/Harischandra-Prasad-Vissamsetti/Crop-Suitability-Classification.git
%cd <Crop-Suitability-Classification>
```
or upload `crop_capstone.zip` using the Files panel and run:
```python
!unzip -q crop_capstone.zip
%cd crop_capstone
```

**3. Install Streamlit** (other libraries are already in Colab)
```python
!pip install -q streamlit
```

**4. Train the model**
```python
!python src/eda.py
!python src/train.py
```
Re-run training in Colab so the saved model matches Colab's scikit-learn version.

**5. View the charts**
```python
from IPython.display import Image, display
import glob
for f in sorted(glob.glob("reports/figures/*.png")):
    display(Image(f))
```

**6. Quick prediction test**
```python
import sys, joblib, pandas as pd
sys.path.insert(0, "src")
from preprocessing import FEATURES
model = joblib.load("models/crop_model.joblib")
x = pd.DataFrame([dict(N=80, P=47, K=40, temperature=23.7, humidity=82, ph=6.4, rainfall=233)], columns=FEATURES)
p = model.predict_proba(x)[0]
for i in p.argsort()[::-1][:5]:
    print(f"{model.classes_[i]:12s} {p[i]:.1%}")
```
The expected top result is rice.

**7. Run the Streamlit app in Colab (temporary public link)**
```python
!npm install -q localtunnel
!streamlit run app/streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false --server.headless true &>/content/logs.txt &
import time, urllib
time.sleep(8)
print("Password:", urllib.request.urlopen("https://ipv4.icanhazip.com").read().decode().strip())
!npx localtunnel --port 8501
```
Open the printed `loca.lt` link and enter the printed IP address as the password. Free tunnels can drop or show red "Failed to fetch module" errors. Press `Ctrl+Shift+R`, or run the app locally in VS Code for demos.

---

## 🧪 Sample Inputs for Testing

| Crop | N | P | K | Temp °C | Humidity % | pH | Rainfall mm |
|---|---|---|---|---|---|---|---|
| Rice | 80 | 47 | 40 | 23.7 | 82.2 | 6.4 | 233.1 |
| Chickpea | 39 | 68 | 79 | 18.9 | 16.7 | 7.4 | 79.7 |
| Apple | 24 | 136.5 | 200 | 22.6 | 92.4 | 5.9 | 113 |
| Coffee | 103 | 29 | 30 | 25.7 | 57.6 | 6.8 | 157.8 |
| Cotton | 117 | 46 | 19 | 24 | 80 | 6.8 | 80.2 |

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` for the dataset | Run commands from the project root and keep the CSV in `data/` |
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt` |
| App says "Trained model not found" | Run `python src/train.py` first |
| `streamlit` is not recognized | Use `python -m streamlit run app/streamlit_app.py` |
| Port 8501 in use | Add `--server.port 8502` |
| Model version warning | Re-run `python src/train.py` on your machine |

## ⚠️ Limitations

- Only seven features; no soil type, season, irrigation, altitude or market data.
- The data represents typical conditions for each crop, so inputs should be seasonal averages, not instantaneous readings.
- The app does not connect to live sensors or weather services.
- The dataset is India-based, so predictions elsewhere are less reliable.
- Educational prototype, not a production system. Results support, not replace, professional agronomic advice.

## 🔭 Future Scope

Live weather API integration, CSV batch predictions, ensemble models (Random Forest, Gradient Boosting), SHAP explanations, probability calibration, seasonal features and a multi-language interface.

## 📚 References

1. Ingle, A. *Crop Recommendation Dataset.* Kaggle. https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
2. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12, 2825-2830.
3. McKinney, W. (2010). Data structures for statistical computing in Python. *Proc. SciPy*.
4. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering* 9(3).
5. Streamlit documentation. https://streamlit.io

## 📄 License and Credits

Educational capstone project. The dataset belongs to its original author on Kaggle; check its license before redistributing the raw file.

© 2026 Harischandra Prasad Vissamsetti, M.C.A



# ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white) ![Google Colab](https://img.shields.io/badge/Google%20Colab-Supported-F9AB00?logo=googlecolab&logoColor=white) ![VS Code](https://img.shields.io/badge/VS%20Code-Supported-007ACC?logo=visualstudiocode&logoColor=white)
