# Agricultural Dataset Analysis - Comprehensive Report

## Overview

This repository contains the Python code and analysis performed on an agricultural dataset. The primary goal is to explore and understand various aspects of the data, including commodity prices, market trends, and state-wise distribution. The analysis includes data cleaning, visualization, and predictive modeling.

## Dataset

The dataset used for this analysis is Agriculture dataset. The specific columns and their meanings depend on the original dataset.

## Technologies

-   **Python:** The primary programming language used for data manipulation and analysis.
-   **Pandas:** Used for data manipulation, cleaning, and transformation.
-   **Matplotlib:** Used for creating static, interactive, and animated visualizations.
-   **Seaborn:** Used for data visualization based on matplotlib, providing a higher-level interface for statistical graphics.
-   **Scikit-learn (sklearn):** Used for machine learning tasks, specifically linear regression.

## Code Structure

The main script, `main.py`, performs the following tasks:

1.  **Data Loading and Cleaning:**
    -   Loads the dataset from a CSV file.
    -   Cleans column names by removing leading/trailing spaces.
    -   Renames specific columns for clarity.
    -   Converts the `Arrival_Date` column to datetime objects.
    -   Handles missing values using imputation (mean, forward fill) and dropping duplicates.
    -   Normalizes `MaxPrice` and `ModalPrice` using `MinMaxScaler`.

2.  **Data Exploration and Visualization (Objectives 1-9):**
    -   **Objective 1:** Visualizes the distribution of the top 10 commodities using a pie chart.
    -   **Objective 2:** Plots the top 10 markets by average modal price using a bar plot.
    -   **Objective 3:** Creates a correlation heatmap of price-related columns (`MinPrice`, `MaxPrice`, `ModalPrice`).
    -   **Objective 4:** Generates a histogram to visualize the distribution of modal prices for the top 5 commodities.
    -   **Objective 5:** Plots the record count per state using a bar plot.
    -   **Objective 6:** Creates a pairplot to visualize pairwise relationships between price variables.
    -   **Objective 7:** Visualizes average Modal Price for each Commodity using barplot.
    -   **Objective 8:** Analyzes how prices differ by Market using boxplot.
    -   **Objective 9:** Implements and evaluates a linear regression model to predict `ModalPrice` from `MaxPrice`.  Includes model training, prediction, visualization of the regression line, and calculation of the Mean Squared Error (MSE).

3.  **Model Evaluation (Objective 9):**
    -   Splits the data into training and testing sets.
    -   Trains a `LinearRegression` model.
    -   Makes predictions on the test set.
    -   Calculates the Mean Squared Error (MSE) to evaluate model performance.

## Running the Code

1.  **Prerequisites:**
    -   Python 3.x installed.
    -   Install the necessary libraries:
        ```bash
        pip install pandas matplotlib seaborn scikit-learn
        ```
2.  **Execution:**
    -   Save the Python code as a `.py` file (e.g., `main.py`).
    -   Make sure the `Raw data.csv` file is accessible at the path specified in the `pd.read_csv()` function (or change the path to match the actual location).
    -   Run the script from the command line:
        ```bash
        python main.py
        ```
    -   The code will generate various plots and print the prediction and MSE to the console.


## Output

The script will generate several visualizations, including:

-   Pie chart showing the distribution of top 10 commodities.
-   Bar plot of top 10 markets by average modal price.
-   Heatmap of correlations between price variables.
-   Histograms showing the distribution of modal prices for the top 5 commodities.
-   Bar plot of the record count per state.
-   Pairplot of the price variables.
-   Bar plot of average Modal Price by commodity
-   Boxplot of Modal Price distribution by Market.
-   Scatter plot with regression line showing the prediction from LinearRegression
-   The predicted value using regression model
-   The Mean Squared Error (MSE) of the linear regression model.

## Further Enhancements

-   **Advanced Data Cleaning:**  Handle outliers in price columns.
-   **Feature Engineering:**  Create new features such as price fluctuations over time or market-specific seasonality.
-   **More Sophisticated Modeling:**  Try other regression models (e.g., Random Forest, Gradient Boosting) or time series analysis techniques.
-   **Interactive Visualizations:**  Use libraries like Plotly or Bokeh for interactive plots.
-   **Reporting:**  Generate a comprehensive report using tools like Jupyter Notebook or a dedicated reporting library.
-   **Automated Data Loading:** If the data is updated frequently, automate the data loading process using a scheduled task or API integration.
