# Box Office Prediction of *Avatar: Fire & Ash* 

<img width="1392" height="600" alt="image" src="https://github.com/user-attachments/assets/b61350e4-b6c5-4247-9fa5-829854c7c7e5" />


---
> *"I don't look at the box office. I look at the seats. If people are filling the seats, the money takes care of itself."* — James Cameron

In the high-stakes world of modern cinema, producing a blockbuster is no longer just art; it is a billion-dollar gamble. With production budgets exceeding $250 million ($400 million in case of *Avatar: Fire and Ash*), studios cannot rely on gut feeling alone. They need data-driven certainty. However, predicting the opening weekend for a James Cameron film presents a unique paradox: his films historically open "slow" but have infinite "legs" (long-term endurance), defying the front-loaded trend of modern Marvel-style blockbusters.

This project leverages Machine Learning (XGBoost) to predict the Domestic Opening Week Gross for the upcoming movie Avatar: Fire and Ash (to be released on Dec 19, 2025). By analyzing over 20 years of historical movie metadata (2000–2025), the model attempts to identify key drivers of blockbuster success.

---
## Repository Contents

| File / Directory | Description & Role |
| :--- | :--- |
| **`01_Data_Acquisition_Feature_Engineering.ipynb`** | **The Engine:** A custom scraper that bypasses API limits using time-chunking. It includes Sequel Imputation Logic (fixing missing predecessor data for franchises) and generates historical features like Cumulative Cast Revenue ("Star Power") and Co-Star Collaboration Counts ("Actor Familiarity"). |
| **`02_EDA_Modeling_and_Final_Prediction.ipynb`** | **The Analysis:** Contains the full research pipeline—from EDA to benchmarking algorithms (DNN vs. XGBoost). Includes SHAP analysis for interpretability and the final inference logic for *Avatar: Fire & Ash*. |
| **`data/Avatar_Final_Dataset_Enhanced.csv`** | **The Final Dataset:** The fully processed dataset (5,649 rows) with imputed theater counts, inflation-adjusted financials, and encoded distributor tiers. Ready for training. |
| **`data/phase1_target_list.csv`** | **Raw Data (Phase 1):** The initial list of high-revenue movie IDs (19,170 rows) and release dates scraped from TMDb. |
| **`data/phase2_features.csv`** | **Raw Data (Phase 2):** Intermediate metadata containing uncleaned budgets, cast lists, and keyword tags before engineering. |
| **`models/avatar_model_final.pkl`** | The serialized **Weighted XGBoost** model, saved and ready for deployment or web-app integration. |
| **`models/avatar_scaler.pkl`** | **The Normalizer:** The fitted `StandardScaler` object required to preprocess new user inputs (e.g., Budgets) exactly as the model expects. |
| **`models/avatar_features.pkl`** | **The Map:** An ordered list of feature names. This ensures that the web app aligns the columns correctly (preventing a "Feature Mismatch Error") during prediction. |


## **Methodology:**

### **1. Data Source & Acquisition**
Due to the strict anti-scraping policies of Box Office Mojo and the lack of a public API for granular weekly data, the project utilized a Statistical Proxy Approach. It extracted verified Global Revenue figures from TMDb and applied a standard decay ratio (Derived from average blockbuster performance: 45% Domestic Share × 35% First Week Drop) to synthesize the target variable for training.

Historically, Hollywood blockbusters derive approximately **40%** of their total revenue from the Domestic (US/Canada) market and 60% form International markets. For wide-release films, the **Opening Week** (First 7 Days) typically accounts for **35%** of the total lifetime domestic gross due to marketing hype and front-loaded demand [Source: "Box Office Decay Curves", Industry Analysis].

### **2. Dataset Information**
Data was acquired via the **TMDb (The Movie Database) API** using a custom time-chunking scraper (`01_Data_Acquisition_Feature_Engineering.ipynb`) to bypass the 10,000-item API limit. The dataset spans releases from January 2000 to December 2025.


| **Data Set Characteristics:** | Multivariate, Time-Series | **Number of Instances:** | 5,649 |
| :--- | :--- | :--- | :--- |
| **Attribute Characteristics:** | Real, Integer, Categorical | **Number of Attributes:** | 17 |
| **Associated Tasks:** | Regression, Causal Inference | **Missing Values?** | Yes (Imputed) |
| **Area:** | Film Industry | **Date Created:** | December 2025 |


### **3. Final Dataset Overview**

| Attribute | Type | Description & Methodology |
| :--- | :--- | :--- |
| **`Log_Budget`** | Continuous | Natural log of production budget (Inflation Adjusted to 2025 USD). |
| **`Theater_Count`** | Integer | **Strongest Predictor (Kim et al., 2020):** Estimated screen count based on Distributor Tier. |
| **`Cast_Star_Power`** | Continuous | Cumulative inflation-adjusted gross of the top 3 cast members. **Validation:** A Two-Sample Kolmogorov-Smirnov (KS) Test confirmed ($p < 0.05$) that high-revenue films come from a statistically distinct "Star Power" distribution. |
| **`Director_Prev_Gross`**| Continuous | Cumulative inflation-adjusted gross of the director's previous filmography. |
| **`Prev_Movie_Gross`** | Continuous | Inflation-adjusted revenue of the specific franchise predecessor (e.g., *Avatar 1* for *Avatar 2*). Imputed using Budget-Ratios for older films. |
| **`Distributor_Tier`** | Ordinal | **Proxy for Marketing Power:** <br>• **Tier 1:** Disney, Warner, Universal, Sony (>4,000 screens)<br>• **Tier 2:** A24, Lionsgate, Netflix (~2,500 screens)<br>• **Tier 3:** Independent/Niche (<500 screens) |
| **`Competition_Score`** | Integer | Count of other "Major" releases opening within ±7 days of the target film. |
| **`Actor_Familiarity`** | Integer | Graph Theory metric counting prior collaborations between cast members. |
| **`Runtime`** | Integer | Duration in minutes. (Acts as a capacity constraint; longer movies = fewer daily shows). |
| **`MPAA_Rating`** | Ordinal | Age-restriction rating encoded as ordinal data (G=0, PG=1, PG-13=2, R=3, NC-17=4). |
| **`Release_Month`** | Categorical | Month of release (1–12) to capture seasonality (e.g., "Summer Blockbuster" or "Holiday Season" effects). |
| **`Has_IMAX_3D`** | Binary | 1 if the film supports Premium Large Formats (IMAX/3D), which command higher ticket prices. |
| **`Genre_SciFi`** | Binary | 1 if genre is Science Fiction (Target genre for *Avatar*). |
| **`Genre_Adventure`** | Binary | 1 if genre is Adventure. |
| **`Genre_Action`** | Binary | 1 if genre is Action. |
| **`Genre_Fantasy`** | Binary | 1 if genre is Fantasy. |
| **`Is_Sequel`** | Binary | 1 if the movie belongs to an existing franchise collection. |


### **3. Log-Transforming Data**
Box office revenue does not follow a normal Gaussian distribution; it follows a **Lévy (Power Law) Distribution**, where a tiny fraction of "Blockbusters" account for the vast majority of industry revenue (Sharma et al., 2021). To handle this extreme skew, all financial variables (Budget, Revenue) were Log-Transformed (`np.log1p`) during preprocessing. This neutralized the "heavy tail" effect, preventing high-magnitude outliers from destabilizing the gradient descent.

### **4. Handling Temporal Gaps (Sequel Imputation)**
To address missing predecessor revenue for sequels falling outside the dataset's timeframe (e.g., a 2008 sequel to a 1989 film), the project applied a Budget-Derived Heuristic. Missing Prev_Gross values were imputed using the dataset's median Predecessor-to-Budget Ratio, and ensured the model captured franchise momentum without learning erroneous zero-revenue correlations.

---

## Model Results & Benchmarking

The project benchmarked multiple architectures against a **Random Forest Baseline**.

| Model | R² Score | Outcome |
| :--- | :--- | :--- |
| **Weighted XGBoost** | **0.552** | **Selected for Inference (Best Stability)** |
| Deep Neural Network | 0.545 | Competitive, but prone to extrapolation hallucination. |
| Random Forest | 0.481 | Baseline Benchmark. |
| QR50 (Robust) | 0.507 | Robust Baseline. Optimized for Median (MAE); ignores outliers but explains less total variance. |
| SVM (Geometric) | 0.465 | Failed to capture high-dimensional non-linearities. |

The project chose **Weighted XGBoost** with Cost-Sensitive Learning (`sample_weight=Budget**2`) to force the model to prioritize accuracy on high-budget blockbusters over low-budget indie films.

**Final Prediction:** The model predicts a Domestic Opening Week of **$114,799,392** for *Avatar: Fire and Ash*.

---

## Relevant Papers & Citations
1.  Sharma, A. S., Roy, T., Rifat, S. A., & Mridul, M. A. (2021). Presenting a larger up-to-date movie dataset and investigating the effects of pre-released attributes on gross revenue. Journal of Computer Science, 17(10), 870–888. https://doi.org/10.3844/jcssp.2021.870.888.
2.  Kim, J.-M., Xia, L., Kim, I., Lee, S., & Lee, K.-H. (2020). Finding Nemo: Predicting movie performances by machine learning methods. Journal of Risk and Financial Management, 13(5), 93. https://doi.org/10.3390/jrfm13050093
3.  Chao, Q., Kim, E., & Li, B. (2023). Movie Box Office Prediction With Self-Supervised and Visually Grounded Pretraining. Nanyang Technological University; Alibaba Group. https://github.com/jdsannchao/MOVIE-BOX-OFFICE-PREDICTION
