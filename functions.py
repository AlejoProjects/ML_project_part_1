from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.stattools import durbin_watson
#from pyexpat import model
from matplotlib import pyplot as plt
import statsmodels.stats.api as sms
import statsmodels.api as sm
import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns  



def eda_info(column,column_name):
    '''
    Perform Exploratory Data Analysis (EDA) on a given column of a DataFrame.
    '''
    print("\n--- Statistical Summary of " + column_name + " ---")
    print("\nSkewness:",column.skew())
    print("Kurtosis:", column.kurtosis())
    plt.figure(figsize=(10, 6))
    # Histogram and KDE (Kernel Density Estimate)
    sns.histplot(column, bins=50, kde=True, color='purple', edgecolor='black')
    plt.title(column_name, fontsize=16)
    plt.xlabel('values', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

def total_eda_info(df):
    '''
    Perform Exploratory Data Analysis (EDA) on the entire DataFrame, including visualizations and statistical summaries for each column.
    param df: The DataFrame to analyze.'''
    print("--- Statistical Summary of critical_temp ---")
    columns = np.size(df.columns)
    column_names = df.columns.tolist() 
    for i in range(columns):
        column = df.iloc[:, i]
        eda_info(column,column_names[i]) 

def plot_values(x,y,text,color_used='blue'):
    '''
    Plot the relationship between two variables with a line plot.
    param x: The independent variable (x-axis).
    param y: The dependent variable (y-axis).
    param text: A list containing labels for x, y, and the title.
    param color_used: The color to use for the plot.
    '''
    plt.xlabel(text[0])
    plt.ylabel(text[1])
    plt.title(text[2])
    print("columns: ", np.size(x))
    plt.plot(x, y, marker='o', label=text[3], color=color_used)
    plt.legend()

def sort_values(sk,ku):
    skewness_sort = sk.sort_values(ascending=False)
    kurtosis_sort = ku.sort_values(ascending=False)
    return skewness_sort, kurtosis_sort
def classify_kurtosis(ku_sort,epsilon = 0.5):
    '''
    Classify columns based on their kurtosis values into mesokurtic, leptokurtic, and platykurtic categories.
     param ku_sort: A sorted Series of kurtosis values for each column.
     param epsilon: A small threshold value to determine the classification boundaries (default is 0.
    
    '''
    #mesokurtic stands for values closest 0 
    mesokurtic_condition = (-epsilon <= ku_sort ) & (ku_sort <= epsilon)
    mesokurtic = pd.DataFrame(ku_sort[mesokurtic_condition])
    print("Mesokurtic columns:")
    print(f' {mesokurtic.shape[0]} columns')
    #Leptokurtic stands for values higher than 0
    leptokurtic_condition = ku_sort > epsilon
    leptokurtic = pd.DataFrame(ku_sort[leptokurtic_condition])
    print("Leptokurtic columns:")
    print(f' {leptokurtic.shape[0]} columns')
    #Platykurtic stands for values lower than 0
    platykurtic_condition = (ku_sort < -epsilon) 
    platykurtic =pd.DataFrame( ku_sort[platykurtic_condition])
    print("Platykurtic columns:")
    print(f' {platykurtic.shape[0]} columns')
    return mesokurtic, leptokurtic, platykurtic


def classify_skewness(sk_sort, epsilon=0.5):
    '''
    Classify columns based on their skewness values into left-skewed, right-skewed, and approximately symmetric categories.
    params sk_sort: A sorted Series of skewness values for each column.
    params epsilon: A small threshold value to determine the classification boundaries (default is 0.5).
    '''
    # Left-skewed (negatively skewed): Skewness < -0.5
    left_skewed_condition = sk_sort < -epsilon
    left_skewed = pd.DataFrame(sk_sort[left_skewed_condition])
    print("Left-skewed columns:")
    print(f' {left_skewed.shape[0]} columns')
    # Right-skewed (positively skewed): Skewness > 0.5
    right_skewed_condition = sk_sort > epsilon
    right_skewed = pd.DataFrame(sk_sort[right_skewed_condition])
    print("Right-skewed columns:")
    print(f' {right_skewed.shape[0]} columns')
    # Approximately symmetric: Skewness between -0.5 y 0.5 (INCLUSIVO)
    symmetric_condition = (sk_sort >= -epsilon) & (sk_sort <= epsilon)
    symmetric = pd.DataFrame(sk_sort[symmetric_condition])
    print("Approximately symmetric columns:")
    print(f' {symmetric.shape[0]} columns')
    
    return left_skewed, right_skewed, symmetric
def mean_imputation(df, columns):
    df[columns] = df[columns].fillna(df[columns].mean())
    return df
def median_imputation(df, columns):
    df[columns] = df[columns].fillna(df[columns].median())
    return df
def evaluate_model(model, X_test, Y_test, model_name):
    '''
    Evaluate the performance of a regression model using RMSE, MAE, and R^2 metrics.
    param model: The fitted regression model to evaluate.
    param X_test: The test set features.
    param Y_test: The test set target variable.
    param model_name: The name of the model for display purposes.
    '''
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, preds))
    mae = mean_absolute_error(Y_test, preds)
    r2 = r2_score(Y_test, preds)
    print(f"{model_name} RMSE: {rmse:.3f}")
    print(f"{model_name} MAE:  {mae:.3f}")
    print(f"{model_name} R^2:  {r2:.3f}\n")
    return rmse, mae, r2

def linearity_check(u_model, X_test, Y_test):
    '''
    Check for linearity and homoscedasticity using a residuals vs. fitted values plot.
    param u_model: The fitted OLS regression model.
    param X_test: The test set features.
    param Y_test: The test set target variable.
    '''
    # --- Linearity & Homoscedasticity (Visual Check) ---
    print("Checking for Linearity (Visual Check)...")
    fitted = u_model.predict(X_test) # FIXED: Replaced .fittedvalues with .predict()
    residuals = Y_test - fitted
    plt.figure(figsize=(8, 5))
    sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 2})
    plt.title('Residuals vs. Fitted Values')
    plt.xlabel('Fitted Values (Predictions)')
    plt.ylabel('Residuals')
    plt.show()

def residual_independence_test(u_model, X_test, Y_test):
    '''
    check for independence of residuals using the Durbin-Watson test, which tests for autocorrelation in the residuals.
    param u_model: The fitted OLS regression model.
    param X_test: The test set features.
    param Y_test: The test set target variable.
    '''
    # --- Independence of Residuals ---
    print("Checking for Independence of Residuals (Durbin-Watson Test)...")
    residuals = Y_test - u_model.predict(X_test)
    dw_stat = durbin_watson(residuals)
    print(f"Durbin-Watson statistic: {dw_stat:.2f}")
    if 1.5 < dw_stat < 2.5:
        print("Result: No significant autocorrelation (Good).")
    else:
        print("Result: Potential autocorrelation detected (Bad).")
    print("\n" + "="*80 + "\n")

def homoscedasticity_test(u_model, X_test, Y_test):
    '''
    check for homoscedasticity using the Breusch-Pagan test, which tests whether the variance of the residuals is constant across all levels of the independent variables.
    param u_model: The fitted OLS regression model.
    param X_test: The test set features.
    param Y_test: The test set target variable.
    '''
    # --- Homoscedasticity ---
    print("Checking for Homoscedasticity (Breusch-Pagan Test)...")
    residuals = Y_test - u_model.predict(X_test)
    X_test_with_const = sm.add_constant(X_test) 
    
    bp_test = sms.het_breuschpagan(residuals, X_test_with_const)
    print(f"Breusch-Pagan Test p-value: {bp_test[1]:.4f}")
    if bp_test[1] > 0.05:
        print("Result: No evidence of heteroscedasticity (Good).")
    else:
        print("Result: Evidence of heteroscedasticity found (Bad).")
    print("\n" + "="*80 + "\n")

def normality_test(u_model, X_test, Y_test):
    '''
    check for normality of residuals using a Q-Q plot and the Jarque-Bera test.
    param u_model: The fitted OLS regression model.
    param X_test: The test set features.
    param Y_test: The test set target variable.
    '''
    # --- Normality of Residuals ---
    print("Checking for Normality of Residuals (Q-Q Plot and Jarque-Bera)...")
    residuals = Y_test - u_model.predict(X_test)
    fig = sm.qqplot(residuals, line='45', fit=True)
    plt.title("Q-Q Plot of Residuals")
    plt.show()
    # FIXED: Used scipy.stats instead of u_model.summary2()
    jb_stat, jb_prob = stats.jarque_bera(residuals)
    print(f"Jarque-Bera test probability (p-value): {jb_prob:.4f}")
    if jb_prob > 0.05:
        print("Result: Residuals appear to be normally distributed (Good).")
    else:
        print("Result: Residuals may not be normally distributed (Bad).")
    print("\n" + "="*80 + "\n")

def multicollinearity_test(X):
    '''
    check for multicollinearity using Variance Inflation Factor (VIF).
    param X: The feature matrix (DataFrame) used in the regression model.
    '''

    print("Checking for Multicollinearity (VIF)...")
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)   
    X_no_const = X.drop('const', axis=1, errors='ignore') # errors='ignore' prevents crash if 'const' isn't there
    vif_data = pd.DataFrame() # FIXED: Changed data.DataFrame() to pd.DataFrame()
    vif_data["feature"] = X_no_const.columns
    # Ensure data is numeric for VIF calculation
    X_values = X_no_const.dropna().values 
    vif_data["VIF"] = [variance_inflation_factor(X_values, i) for i in range(X_no_const.shape[1])]
    print(vif_data)
    if all(vif_data["VIF"] < 5):
        print("\nResult: No significant multicollinearity detected (Good).")
    else:
        print("\nResult: Potential multicollinearity detected (Bad).")
    print("\n" + "="*80 + "\n")

def tests_check(u_model, X_test, Y_test):
    '''
    Run all diagnostic tests for the OLS regression model.
    param u_model: The fitted OLS regression model.
    param X_test: The test set features.
    param Y_test: The test set target variable. 
    '''
    linearity_check(u_model, X_test, Y_test)
    residual_independence_test(u_model, X_test, Y_test)
    homoscedasticity_test(u_model, X_test, Y_test)
    normality_test(u_model, X_test, Y_test)
    # Note: multicollinearity_test requires the X dataframe, so call it separately or pass X_test to it.
    multicollinearity_test(X_test)
def check_regularized_residuals(model, X_test, Y_test, model_name="Model"):
    '''
    Check residuals for regularized regression models (Ridge and Lasso) using visual diagnostics.
     param model: The fitted regularized regression model (Ridge or Lasso).
    param X_test: The test set features.
    param Y_test: The test set target variable.
    param model_name: A string to identify the model in the plots (default is "Model").
     This function generates two plots:
    1. Residuals vs. Fitted Values: To visually assess linearity and homoscedasticity.
    2. Q-Q Plot of Residuals: To visually assess the normality of
    
    '''
    print(f"--- Visual Residual Diagnostics for {model_name} ---")
    fitted = model.predict(X_test)
    residuals = Y_test - fitted
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    # 1. Linearity & Homoscedasticity Visual Check
    sns.residplot(x=fitted, y=residuals, lowess=True, 
                  line_kws={'color': 'red', 'lw': 2}, ax=ax[0])
    ax[0].set_title(f'Residuals vs. Fitted Values ({model_name})')
    ax[0].set_xlabel('Predicted Values')
    ax[0].set_ylabel('Residuals')
    # 2. Normality Visual Check
    sm.qqplot(residuals, line='45', fit=True, ax=ax[1])
    ax[1].set_title(f"Q-Q Plot of Residuals ({model_name})")
    
    plt.tight_layout()
    plt.show()