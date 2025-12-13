# Avatar: Fire & Ash Box Office Prediction
## **Overview**

This project leverages Machine Learning (XGBoost) to predict the Domestic Opening Week Gross for the upcoming movie Avatar: Fire and Ash (Dec 2025). By analyzing over 20 years of historical movie metadata (2000–2025), the model identifies key drivers of blockbuster success.

## **Methodology**

We moved beyond simple regression by engineering different features found in film industry literature:

**Star Power:** Aggregated, inflation-adjusted box office history of the cast and director.

**Actor Chemistry:** A Graph Theory score measuring how often the cast has collaborated previously.

**Market Saturation:** Analysis of Theater Counts and Distributor Tiers (e.g., Disney vs. Indie).

**Log-Transformation:** Applied np.log1p to financial data to handle the "Long-Tail" distribution of movie revenue.


## **Key Results**

**Model Used:** XGBoost Regressor (Gradient Boosting).

**Performance:** Achieved an R² of 0.XX and MAE of $XX Million.

**Prediction:** The model estimates Avatar: Fire and Ash will earn $XXX,XXX,XXX in its domestic opening week.

## **Repository Structure**
**notebooks/01_Data_Scraping_Pipeline.ipynb:** The custom scraper that fetches data from TMDb using time-chunking.

**notebooks/02_Model_Training_Prediction.ipynb:** The analysis pipeline, including feature selection, training, SHAP analysis, and inference.

**data/:** Contains the processed datasets used for training.
