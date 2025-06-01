# Sales Forecasting
This project applies time series modeling techniques to forecast weekly sales for a major retailer, using the [Walmart Recruiting: Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data) dataset from Kaggle.

## Objective
To build and compare forecasting models (SARIMA and Prophet) that predict weekly sales by department and store. The goal is to provide insights into model accuracy, holiday effects, and overall forecasting performance.

## Dataset

- **Source**: [Kaggle - Walmart Recruiting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data)
- **Files Used**:
  - `train.csv`: Weekly sales data across 45 Walmart stores and 99 departments.
  - `features.csv`: Contains additional features like temperature, fuel prices, CPI, and holidays.
  - `stores.csv`: Metadata about each store's type and size.

> **Note**: Due to file size limitations on GitHub, only a sample or smaller version of the dataset may be included in the repo. Please download the full dataset from Kaggle if you wish to run the notebooks end-to-end.

## Project Structure
```
sales-forecasting/
│
├── notebooks/
│ ├── 01_data_cleaning.ipynb # Merges and processes raw data
│ ├── 02_model_comparison.ipynb # LassoCV modeling and residual diagnostics
│ ├── 03_forecasting.ipynb # SARIMA vs Prophet forecasting and evaluation
│
├── visuals/
│ └── prophet_forecast_plot.png # Example forecast output
│
├── data
│ └── sample_merged_sales_data.csv # Truncated version of main dataset for quick testing
├── README.md # Project overview and instructions
├── requirements.txt # Python dependencies
```

## Key Tools & Libraries
- **Modeling**: `statsmodels`, `prophet`, `scikit-learn`
- **Data Wrangling**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
  
## Results Summary
- Prophet and SARIMA were evaluated on five (store, department) pairs with strong predictive power.
- **SARIMA** performed better on departments with seasonal patterns and less holiday influence.
- **Prophet** handled holiday effects more explicitly and flexibly.
- The best model varied by department; metrics like MAE and RMSE were used to compare.

## How to Use
1. Clone the repo locally:
   ```
   git clone https://github.com/maekala/sales-forecasting.git
   ```
2. (Optional) Create a virtual environment:
   ```
   conda create -n sales_forecasting python=3.10
   conda activate sales_forecasting
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download full dataset from Kaggle and place it in the project root or a `data/` folder.

5. Run the Jupyter notebooks in order (starting with `eda.ipynb`).

## Future Work

- Extend forecasting to include all departments using automated hyperparameter tuning.
- Build an interactive dashboard to explore forecasts and model metrics.
- Explore deep learning methods like LSTMs for further accuracy gains.
