# Avatar: Fire & Ash Box Office Prediction Dataset

| **Data Set Characteristics:** | Multivariate, Time-Series | **Number of Instances:** | 20,400 |
| :--- | :--- | :--- | :--- |
| **Attribute Characteristics:** | Real, Integer, Categorical | **Number of Attributes:** | 14 |
| **Associated Tasks:** | Regression, Causal Inference | **Missing Values?** | Yes (Imputed) |
| **Area:** | Business / Arts | **Date Donated:** | December 2025 |

---

## Abstract
This dataset was scraped and engineered to predict the **Domestic Opening Week Gross** for the upcoming film *Avatar: Fire and Ash* (2025). It leverages 25 years of historical metadata (2000–2025) to model the non-linear economic drivers of the modern film industry. The project utilizes a **Weighted XGBoost** architecture, selected after benchmarking against a Random Forest baseline [Source 171].



---

## Dataset Information

### **1. Data Source & Acquisition**
Data was acquired via the **TMDb (The Movie Database) API** using a custom time-chunking scraper (`01_Data_Acquisition_Feature_Engineering.ipynb`) to bypass the 10,000-item API limit. The dataset spans releases from January 2000 to December 2025.

### **2. Statistical Characteristics (The "Lévy Flight" Problem)**
Box office revenue does not follow a normal Gaussian distribution; it follows a **Lévy (Power Law) Distribution**, where a tiny fraction of "Blockbusters" account for the vast majority of industry revenue [Source 213].
* **Methodology:** To handle this extreme skew, all financial variables (Budget, Revenue) were Log-Transformed (`np.log1p`) during preprocessing. This neutralized the "heavy tail" effect, preventing high-magnitude outliers from destabilizing the gradient descent.

### **3. Target Variable Engineering (The Proxy Logic)**
**Challenge:** Explicit "Domestic Opening Week" data is often missing for mid-tier or older films.
**Solution:** We engineered a proxy target using industry-standard decay curves:
$$Target = Global Revenue \times 0.40 \text{ (Domestic Share)} \times 0.35 \text{ (Front-Load Factor)}$$
* **Justification:** Wide-release films typically derive ~35% of their total domestic gross in the first 7 days.

---

## Attribute Information (Features)

| Attribute | Type | Description & Methodology |
| :--- | :--- | :--- |
| **`Log_Budget`** | Continuous | Natural log of production budget (Inflation Adjusted to 2025 USD). |
| **`Theater_Count`** | Integer | **Strongest Predictor:** Estimated screen count based on Distributor Tier. Confirmed by *The Economics of Theatrical Distribution* [Source 2] as the primary driver of Opening Weekend ROI. |
| **`Cast_Star_Power`** | Continuous | **Validated Feature:** Cumulative inflation-adjusted gross of the top 3 cast members. **Validation:** A Two-Sample Kolmogorov-Smirnov (KS) Test confirmed ($p < 0.05$) that high-revenue films come from a statistically distinct "Star Power" distribution. |
| **`Director_Prev_Gross`**| Continuous | Cumulative inflation-adjusted gross of the director's previous filmography. |
| **`Distributor_Tier`** | Ordinal | **Proxy for Marketing Power:** <br>• **Tier 1:** Disney, Warner, Universal, Sony (>4,000 screens)<br>• **Tier 2:** A24, Lionsgate, Netflix (~2,500 screens)<br>• **Tier 3:** Independent/Niche (<500 screens) |
| **`Competition_Score`** | Integer | Count of other "Major" releases opening within ±7 days of the target film. |
| **`Actor_Familiarity`** | Integer | Graph Theory metric counting prior collaborations between cast members. |
| **`Runtime`** | Integer | Duration in minutes. (Acts as a capacity constraint; longer movies = fewer daily shows). |
| **`Genre_SciFi`** | Binary | 1 if genre is Science Fiction (Target genre for *Avatar*). |
| **`Is_Sequel`** | Binary | 1 if the movie belongs to an existing franchise collection. |

---

## Model Results & Benchmarking

The project benchmarked multiple architectures against a **Random Forest Baseline**, established as the standard numerical classifier for tabular data [Source 171].

| Model | R² Score | Outcome |
| :--- | :--- | :--- |
| **Weighted XGBoost** | **0.552** | **Selected for Inference (Best Stability)** |
| Deep Neural Network | 0.545 | Competitive, but prone to extrapolation hallucination. |
| Random Forest | 0.481 | Baseline Benchmark. |
| SVM (Geometric) | 0.465 | Failed to capture non-linear interactions. |

**Final Prediction:** The model predicts a Domestic Opening Week of **$114,799,392** for *Avatar: Fire and Ash*.

---

## Relevant Papers & Citations
1.  **Lévy Distribution in Economics:** *Modeling the Motion Picture Industry: The "Lévy Flight" of Box Office Revenue* (Source 213).
2.  **Feature Importance:** *The Economics of Theatrical Distribution: Screen Count vs. ROI* (Source 2).
3.  **Baseline Standards:** *Comparative Analysis of Ensemble Methods in Tabular Data* (Source 171).
