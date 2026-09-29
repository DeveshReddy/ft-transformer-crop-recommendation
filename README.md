Yes. \*\*Copy everything inside this single block\*\*, paste it into `README.md`, then \*\*Ctrl + S\*\*.



````markdown

\# AI-Powered Crop Recommendation System Using FT-Transformer



An AI-powered crop recommendation system that uses \*\*FT-Transformer\*\*, a transformer-based architecture designed for tabular data, to recommend suitable crops based on soil and environmental conditions.



The system accepts important agricultural parameters such as \*\*Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, and rainfall\*\* and predicts suitable crop classes.



\---



\## 🌱 Project Overview



Crop selection depends on several soil and environmental factors. Traditional crop recommendation approaches often rely on predefined rules or conventional machine learning algorithms.



This project uses \*\*FT-Transformer\*\* to learn relationships between multiple agricultural features and recommend suitable crops.



The system also provides additional agricultural information such as:



\- Crop suitability

\- Prediction confidence

\- NPK fertilizer information

\- Water requirements

\- Crop calendar information

\- Suggested water sources

\- Weather-based prediction using live weather data



\---



\## 🎯 Objectives



\- Build an AI-based crop recommendation system.

\- Use FT-Transformer for tabular agricultural data.

\- Combine agricultural datasets from multiple sources.

\- Apply proper preprocessing and data splitting techniques.

\- Provide crop predictions through an interactive Streamlit application.

\- Provide additional crop-specific agricultural recommendations.



\---



\## 🏗️ System Architecture



```text

&#x20;               ┌─────────────────────────────┐

&#x20;               │   Agricultural Input Data   │

&#x20;               │ N, P, K, pH, Temp, Humidity │

&#x20;               │          Rainfall           │

&#x20;               └──────────────┬──────────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;               ┌─────────────────────────────┐

&#x20;               │     Data Preprocessing      │

&#x20;               │  Cleaning \& Label Encoding  │

&#x20;               │       StandardScaler        │

&#x20;               └──────────────┬──────────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;               ┌─────────────────────────────┐

&#x20;               │       FT-Transformer       │

&#x20;               │  Feature Tokenization       │

&#x20;               │  Transformer Encoder        │

&#x20;               │  Self-Attention             │

&#x20;               │  Feed-Forward Layers        │

&#x20;               └──────────────┬──────────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;               ┌─────────────────────────────┐

&#x20;               │      Crop Prediction       │

&#x20;               │  Probability Distribution  │

&#x20;               │       Top-K Crops           │

&#x20;               └──────────────┬──────────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;               ┌─────────────────────────────┐

&#x20;               │     Agricultural Advisory  │

&#x20;               │ NPK • Water • Crop Calendar│

&#x20;               │       • Water Sources      │

&#x20;               └─────────────────────────────┘

```



\---



\## 🧠 FT-Transformer



The core machine learning model is \*\*FT-Transformer (Feature Tokenizer Transformer)\*\*, a transformer architecture designed for tabular data.



Instead of treating all numerical features as a single vector, the model converts individual features into learnable representations called \*\*tokens\*\*.



\### Main Components



\- Feature Tokenization

\- Learnable CLS Token

\- Transformer Encoder

\- Multi-Head Self-Attention

\- Feed-Forward Network

\- GEGLU activation

\- Classification Head



The model learns relationships between soil and environmental features before producing crop classification probabilities.



\---



\## 🔄 Data Processing Pipeline



```text

Kaggle Dataset ─────┐

&#x20;                   │

&#x20;                   ▼

&#x20;             Dataset Integration

&#x20;                   │

Mendeley Dataset ───┘

&#x20;                   │

&#x20;                   ▼

&#x20;            Schema Mapping

&#x20;                   │

&#x20;                   ▼

&#x20;             Label Normalization

&#x20;                   │

&#x20;                   ▼

&#x20;             Data Validation

&#x20;                   │

&#x20;                   ▼

&#x20;           Stratified Data Split

&#x20;                   │

&#x20;                   ▼

&#x20;             StandardScaler

&#x20;                   │

&#x20;                   ▼

&#x20;           FT-Transformer Model

```



\### Preprocessing



The preprocessing pipeline includes:



1\. Combining agricultural datasets.

2\. Mapping different dataset schemas.

3\. Normalizing crop labels.

4\. Handling missing rainfall values where required.

5\. Encoding crop labels using `LabelEncoder`.

6\. Splitting the data into training, validation, and testing sets.

7\. Fitting `StandardScaler` only on the training data.

8\. Applying the trained scaler to validation, test, and inference data.



\---



\## 🏋️ Model Training



The FT-Transformer model is trained using:



\- \*\*Loss Function:\*\* Cross-Entropy Loss

\- \*\*Optimizer:\*\* AdamW

\- \*\*Learning Rate Scheduling:\*\* ReduceLROnPlateau

\- \*\*Evaluation:\*\* Validation performance

\- \*\*Early Stopping:\*\* Used to prevent unnecessary training

\- \*\*Checkpointing:\*\* Best model state is saved for inference



The trained checkpoint stores important information required for prediction, including:



\- Model weights

\- Scaler

\- Label encoder

\- Number of classes

\- Feature columns



\---



\## 🔮 Inference Pipeline



```text

User Input / Weather Data

&#x20;         │

&#x20;         ▼

&#x20;    Feature Vector

&#x20;         │

&#x20;         ▼

&#x20;    StandardScaler

&#x20;         │

&#x20;         ▼

&#x20;  FT-Transformer Model

&#x20;         │

&#x20;         ▼

&#x20;   Crop Probabilities

&#x20;         │

&#x20;         ▼

&#x20;     Top-K Crops

&#x20;         │

&#x20;         ▼

&#x20;Agricultural Advisory

```



During inference, the saved model checkpoint is loaded and the input features are transformed using the same preprocessing pipeline used during training.



The model then generates crop probabilities and identifies the most suitable crop classes.



\---



\## 🌦️ Weather Integration



The application supports weather-based inputs.



Using the weather integration mode, environmental information such as:



\- Temperature

\- Humidity

\- Rainfall



can be obtained using weather data services.



The application can also accept these values manually when live weather information is not being used.



\---



\## 🌾 Agricultural Advisory System



In addition to crop prediction, the application provides crop-specific information through the agricultural metadata module.



The advisory system can provide:



\- NPK fertilizer gap information

\- Water requirements

\- Suggested water sources

\- Planting and harvesting periods

\- Crop season information

\- Crop-specific recommendations



\---



\## 🖥️ Streamlit Application



The project includes an interactive \*\*Streamlit web application\*\*.



The application provides:



\- Manual agricultural input

\- Weather-based input

\- Crop prediction

\- Top-3 prediction confidence

\- Agricultural recommendations

\- Crop information

\- CSV export/history functionality



The application follows a modular structure:



```text

Streamlit UI

&#x20;    │

&#x20;    ▼

Inference Module

&#x20;    │

&#x20;    ▼

FT-Transformer Model

&#x20;    │

&#x20;    ▼

Agricultural Advisory Module

```



\---



\## 📁 Project Structure



```text

ft-transformer-crop-recommendation/

│

├── about/

│

├── checkpoints/

│   ├── best\_model.pt

│   └── history.json

│

├── data/

│   ├── Crop\_recommendation.csv

│   ├── Mendley\_crop.csv

│   └── Merged\_crop\_recommendation.csv

│

├── app.py

├── crop\_metadata.py

├── inference.py

├── model.py

├── train.py

├── requirements.txt

├── .gitignore

└── README.md

```



> \*\*Note:\*\* Dataset files and model checkpoints are excluded from Git using `.gitignore` and may need to be provided locally before running the application.



\---



\## 📄 Main Files



| File | Purpose |

|------|---------|

| `app.py` | Streamlit application and user interface |

| `model.py` | FT-Transformer model architecture |

| `train.py` | Model training pipeline |

| `inference.py` | Model loading and prediction pipeline |

| `crop\_metadata.py` | Crop-specific agricultural information |

| `requirements.txt` | Python dependencies |

| `data/` | Agricultural datasets |

| `checkpoints/` | Trained model checkpoints |



\---



\## ⚙️ Installation



\### 1. Clone the repository



```bash

git clone https://github.com/DeveshReddy/ft-transformer-crop-recommendation.git

cd ft-transformer-crop-recommendation

```



\### 2. Create a virtual environment



```bash

python -m venv .venv

```



\### 3. Activate the virtual environment



\#### Windows



```powershell

.venv\\Scripts\\activate

```



\#### Linux / macOS



```bash

source .venv/bin/activate

```



\### 4. Install dependencies



```bash

pip install -r requirements.txt

```



\---



\## 🚀 Run the Application



Start the Streamlit application using:



```bash

streamlit run app.py

```



The application will open in the browser.



\### Windows alternative



If PowerShell execution policy prevents activation, the project can also be run using the Python executable directly:



```powershell

.\\.venv\_windows\\Scripts\\python.exe -m streamlit run app.py

```



\---



\## 🧪 Example Input



Example agricultural conditions:



```text

Nitrogen     : 22

Phosphorus   : 72

Potassium    : 25

Temperature  : 24.5 °C

Humidity     : 65 %

pH           : 6.8

Rainfall     : 85 mm

```



The model processes these values and produces a crop prediction with confidence scores.



\---



\## 🛠️ Technologies Used



\- Python

\- PyTorch

\- FT-Transformer

\- Pandas

\- NumPy

\- Scikit-learn

\- Streamlit

\- OpenWeatherMap API

\- Matplotlib



\---



\## 🔐 Security



Sensitive configuration such as API keys and environment variables should not be committed to the repository.



The `.env` file is excluded using `.gitignore`.



```text

.env

```



\---



\## 🔬 Research and Machine Learning



The project explores the use of transformer-based deep learning for agricultural tabular data.



The main research focus is applying \*\*FT-Transformer\*\* to learn interactions between soil nutrients and environmental conditions for crop recommendation.



\---



\## 🎓 Academic Project



This project was developed as an academic/research project at:



\*\*SRM Institute of Science and Technology\*\*



Department of Computing Technologies



\---



\## 👨‍💻 Author



\*\*Devesh Reddy\*\*



GitHub:  

https://github.com/DeveshReddy



Project Repository:  

https://github.com/DeveshReddy/ft-transformer-crop-recommendation



\---



\## 📌 Note



This project is intended for educational and research purposes.



Crop recommendations should be considered as decision-support information and should be evaluated together with local agricultural conditions and expert advice.

````



