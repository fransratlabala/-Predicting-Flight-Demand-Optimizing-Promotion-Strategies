# Predicting Flight Demand & Optimizing Promotion Strategies

##  Project Overview
This repository hosts an end-to-end data science and business intelligence project focused on optimizing airline fleet occupancy and revenue management. Utilizing a raw transactional aviation dataset, I engineered a robust preprocessing pipeline in **Python** to handle extreme high-cardinality routing and heavy class imbalance. 

I built, evaluated, and deployed an optimized **XGBoost Classifier** to forecast upcoming flight demand tiers (Low, Medium, High). The model's predictive output was seamlessly joined back to a human-readable text layer and exported into an interactive, multi-page **Power BI Dashboard**—bridging the gap between advanced machine learning and executive-level corporate strategy.

---

##  Data Architecture & Engineering Pipeline
To preserve business insights while maintaining pure model integrity, the data pipeline was structured into three distinct operational layers:

### 1. Strategic Imputation & Row Recovery (Silver Layer)
Hiring managers often look at data preservation skills. Rather than blindly dropping missing values, columns were evaluated individually based on their underlying business logic:
*   **`Airline` (427 Missing Rows):** Imputed with `"Unknown Airline"` to preserve adjacent numeric passenger and cost data.
*   **`Holiday_Season` (987 Missing Rows - ~20% of data):** Imputed with `"Regular Season"`. A missing holiday flag means a flight took off during standard operations; this creates a vital baseline control group.
*   **`Promotion_Type` (1,689 Missing Rows - ~34% of data):** Imputed with `"No Promotion"`. This represents flights sold at standard retail prices, giving us a baseline to measure promotional ROI.
*   **Negligible Missing Data:** Missing strings in `Departure_City` (39), `Arrival_City` (30), and remaining small metrics made up less than 1% of the dataset. These rows were dropped safely to keep map elements pristine.

### 2. Domain-Specific Feature Engineering
*   **`Load_Factor_Pct`:** Calculated by mapping aircraft types to standard commercial seating capacities (`Airbus A380`: 500, `Boeing 777`: 300, `Boeing 737`/`Airbus A320`: 150) and dividing `Passenger_Count` by `Aircraft_Capacity`. This provides an apples-to-apples comparison of how full a plane is, regardless of its size.
*   **`Departure_Hour`:** Extracted the numeric hour (0 to 23) from string text formats (e.g., `"18:43"`) to allow the model to catch business travel surges versus midnight lulls.

### 3. Advanced Categorical Transformations
*   **Smoothed Target Encoding (High-Cardinality Resolution):** The dataset contained 4,769 unique flight vectors (`Route`). Traditional One-Hot Encoding would cause a catastrophic feature explosion. To prevent this, I applied Smoothed Target Encoding regularized against the training global passenger mean using a smoothing weight (\(m=10\)):
    \[\text{Encoded Value} = \frac{(n \times \text{Route Mean}) + (m \times \text{Global Mean})}{n + m}\]
    To strictly eliminate target leakage and overfitting, averages were calculated **exclusively within the training split** and mapped forward.
*   **Temporal Cyclical Encoding:** Standard numeric mapping fails to capture time loops (e.g., Sunday is right next to Monday, and December is next to January). I converted `Month_of_Travel` and `Day_of_Week` into multi-dimensional coordinate pairs using Sine and Cosine transformations:
    \[\text{Day\_Sin} = \sin\left(\frac{2\pi \times \text{Day\_Num}}{7.0}\right), \quad \text{Day\_Cos} = \cos\left(\frac{2\pi \times \text{Day\_Num}}{7.0}\right)\]
*   **One-Hot Encoding with Baseline Protection:** Low-cardinality columns (`Airline`, `Holiday_Season`, `Weather_Conditions`, `Aircraft_Type`, `Promotion_Type`) were processed using `pd.get_dummies()` with `drop_first=True` to defend the models from multicollinearity (the dummy variable trap).

### 4. Direct Target Leakage Elimination
To build a practical, real-world predictive tool, direct leaking metrics (`Passenger_Count`, `Load_Factor_Pct`) along with post-flight targets (`Demand`, `Demand_Code`) were stripped entirely from the features matrix (\(X\)). This forces the model to forecast demand using only properties available at the time of flight scheduling.

---

##  Machine Learning Framework
The engineered dataset presented an asymmetric class distribution on the target variable (`Demand_Code`): **Low Demand (3,142 flights)**, **Medium Demand (965 flights)**, and **High Demand (662 flights)**. 

To prove the efficiency of structural optimization, two configurations were evaluated side-by-side:
1.  **Baseline Model:** A simple, unweighted `Decision Tree Classifier` with a maximum depth of 5.
2.  **Optimized Model:** An `XGBoost Classifier` utilizing custom training row sample-weights computed dynamically to inversely scale class frequencies. This forced tree boosting passes to prioritize minority segment recall.

### 🏆 Predictive Performance Summary

| Core Evaluation Metric | Baseline Model (Decision Tree) | Optimized Model (XGBoost) |
| :--- | :---: | :---: |
| **Overall Model Accuracy** | 56.40% | **84.50%** |
| **Precision (Weighted)** | 52.10% | **83.90%** |
| **Recall (Weighted)** | 56.40% | **84.50%** |
| **F1-Score (Weighted)** | 51.30% | **84.10%** |

*The unweighted baseline frequently default-predicted 'Low' due to class bias. The optimized XGBoost model cleanly bypassed this limitation, showing exceptional stability and high predictive accuracy across minority 'Medium' and 'High' booking conditions.*

---

## 📊 Power BI Business Intelligence Interface
The final XGBoost model predictions were combined back with the clean, human-readable text layer to create a dual-perspective report tailored for business leaders.



### 2. Dashboard Layout Architecture
*   **Tab 1: Commercial Optimization View:** Displays executive KPI cards (`Total Flights`, `Avg Load Factor`) alongside a clustered bar chart of underperforming routes, a promotional donut chart highlighting passenger volume generation, and a dual-axis line chart matching monthly occupancy changes against volatile fuel pricing trends.
*   **Tab 2: Model Monitoring & Validation Panel:** Displays a prominent live model accuracy card. It also uses a Power BI Matrix Visual with actual `Demand` on the rows and `Predicted_Demand` on the columns. This recreates an interactive **Confusion Matrix** directly within the report interface, allowing business analysts to verify where the model is making errors.




