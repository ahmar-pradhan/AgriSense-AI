AgriSense AI — Smart Crop Assistant
Goal: Detect crop diseases from leaf images + predict crop yield from soil/weather data
Type: Multi-modal ML (CV + Tabular)

Problem Statement
- Farmers loose 20-30% of crops due to undetected diseases
- Manual inspection is slow, expensive and inaccurate
- Our app works: Upload leaf photo -> get disease name
Enter soil data ->get yeild prediction

Tech Stack
Language: Python 3.10+
CV Model: Resnet18 (Pytorch)
Tabular Model: XG Boost
UI: Streamlit
Libraries: Pandas, NumPy, Matplotlib, Seaborn
IDE: Jupyter Notebook

Dataset
Dataset 1: PlantVillage
- 54,000 leaf images
- 38 disease classes
- Crops: Apple, Corn, Tomato, Potato etc.

Dataset 2: Crop Yield
- 28,242 rows, 8 columns
- Features: Area, Item, Year, Rainfall,
Pesticides, Temperature
- Target: hg/ha_yield

  Model Architecture
Model 1 — Disease Detection:
Leaf Image → Resize 224x224 → ResNet18
→ Fine-tuned Layer → Disease Name + Confidence%
Model 2 — Yield Prediction:
Soil/Weather Data → Cleaning & Encoding
→ XGBoost → Yield in kg/hectare
