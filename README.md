# 🚗 Road Accident Risk Prediction  
*A Machine Learning Model to Estimate Accident Likelihood Using Road and Environmental Factors*

---

## 📌 Overview

This project predicts the **accident_risk** of a road scenario based on multiple real-world factors such as:

- Road type  
- Weather  
- Time of day  
- Speed limit  
- Road curvature  
- Road signs  
- Lighting  
- Custom engineered risk metrics  

The dataset is from the **Kaggle Playground Series – Season 5 Episode 10** competition.

While my score wasn’t groundbreaking (yet 👀), the goal of this project was learning — not leaderboard domination.

> **Result:** 0.06623 RMSLE on Kaggle → (*Rank ~3674 / 4082 participants*)  
> That’s **bad but not *"crash-the-car bad"* — more like a *slippery corner but recoverable* kind of bad 😄.  

---

## 🧠 Approach

The project followed a full ML workflow:

### ✔ Feature Engineering
I manually engineered meaningful features based on real-world driving logic:

| Feature | Description |
|--------|------------|
| `slipping_chances` | Based on curvature, speed, road type, lighting, and weather |
| `visibility_factor` | Multiplier based on weather + road sign availability |
| `final_risk` | Combined artificial risk estimation |

These features significantly improved the interpretability and performance of the model.

---

### ✔ Model Pipeline

A `RandomForestRegressor` was selected due to its robustness and ability to handle mixed features.  
The full pipeline included:

- Numeric imputation  
- One-hot encoding for categorical values  
- Integrated preprocessing and prediction via `Pipeline()`  
- Hyperparameter tuning (`RandomizedSearchCV`)

---

## 📊 Model Performance

| Metric | Score |
|--------|------|
| 🧪 Notebook RMSLE (Validation) | **0.04941** |
| 🏁 Kaggle Submission RMSLE | **0.06623** |
| 📍 Expected Rank | **~3674 / 4082** |

The difference between notebook and Kaggle score suggests:

- Slight distribution shift  
- Hidden evaluation constraints  
- Possible overfitting opportunity 😀

---

## 🔍 Evaluation Plot

The following scatter plot visualizes how well the model’s predictions match the real accident risk:

![Predicted vs Actual](model-plot.png)

The overall diagonal trend indicates the model is learning patterns — but there's still noise and uncertainty, especially with sparse regions.

---

## 🧪 Example Prediction

Once trained, the model can predict accident risk for a hypothetical road scenario:

```python
result = predict_accident_risk(
    road_type="highway",
    weather="rainy",
    lighting="night",
    speed_limit=120,
    curvature=30,
    road_signs_present=False,
    time_of_day="night"
)

print("Estimated accident risk:", result)
```

#### Output example: 

```yaml
Estimated accident_risk: 0.78
