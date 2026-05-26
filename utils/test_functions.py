from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from .data_visualization import linearity_check
from .data_visualization import model_testing_visualization
from statsmodels.stats.stattools import durbin_watson
from sklearn.neural_network import MLPRegressor
from sklearn.impute import IterativeImputer
from matplotlib import pyplot as plt
import statsmodels.stats.api as sms
import statsmodels.api as sm
import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns  
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

def top_coefficients(features,models):

# Unir coeficientes en un DataFrame para compararlos
    coef_df = pd.DataFrame({
        'Feature': features,
        'OLS_Coef': models['u_model'].coef_,
        'Ridge_Coef': models['r_model'].coef_,
        'Lasso_Coef': models['l_model'].coef_
    })
   

    coef_df['OLS_abs'] = coef_df['OLS_Coef'].abs()
    coef_df['ridge_abs'] = coef_df['Ridge_Coef'].abs()
    coef_df['lasso_abs'] = coef_df['Lasso_Coef'].abs()
    top_5_ols = coef_df.sort_values(by='OLS_abs', ascending=False).head(5)
    top_5_ridge = coef_df.sort_values(by='ridge_abs', ascending=False).head(5)
    top_5_lasso = coef_df.sort_values(by='lasso_abs', ascending=False).head(5)

    print("Top 5 largest coefficients (Unregularized OLS):")
    print(top_5_ols[['Feature', 'OLS_Coef']])

    print("Top 5 largest coefficients (Ridge):")
    print(top_5_ridge[['Feature', 'Ridge_Coef']])

    print("Top 5 largest coefficients (Lasso):")
    print(top_5_lasso[['Feature', 'Lasso_Coef']])

    print("--- 5. Feature Selection (Lasso) ---")
    dropped_features = coef_df[coef_df['Lasso_Coef'] == 0]['Feature'].tolist()
    print(f"Lasso dropped {len(dropped_features)} features.")
    for feature in features:
        if feature in coef_df['Feature'].values:
            val = coef_df.loc[coef_df['Feature'] == feature, 'Lasso_Coef'].values[0]
            if val == 0:
                print(f"-> {feature} fue eliminada (coef = 0). Físicamente, esto sugiere que no tiene poder predictivo o es redundante para predecir la temperatura crítica.")
            else:
                print(f"-> {feature} se mantuvo con un coeficiente de {val:.4f}.") 
    for feature in ['ril', 'msqv']:
        if feature in coef_df['Feature'].values:
            val = coef_df.loc[coef_df['Feature'] == feature, 'Lasso_Coef'].values[0]
            if val == 0:
                print(f"-> {feature} fue eliminada (coef = 0). Físicamente, esto sugiere que no tiene poder predictivo o es redundante para predecir la temperatura crítica.")
            else:
                print(f"-> {feature} se mantuvo con un coeficiente de {val:.4f}.") 
def simple_testing(model, X_test_scaled, Y_test,cu='teal'):
    '''
    Función para realizar un test simple de predicciones vs valores reales y graficar los resultados.
    params model: modelo entrenado (debe tener el método .predict)
    params X_test_scaled: conjunto de características de prueba ya escalado
    params Y_test: valores reales del conjunto de prueba
    '''
    model_test = model.predict(X_test_scaled)
    model_ev = r2_score(Y_test, model_test)
    model_testing_visualization(model, X_test_scaled, Y_test,color_used = cu)
    print(f"R² del modelo: {model_ev:.4f}")

def test_mlp_regressor(data,hidden_layers = (512,256,128,64),learning_rate=0.001):
        X_train_scaled, Y_train, X_test_scaled, Y_test = data
        nn = MLPRegressor(
        hidden_layer_sizes=hidden_layers, # 3 capas ocultas
        activation='relu',                 # función de activación
        solver='adam',                     # algoritmo de optimización
        alpha=0.01,                       # regularización L2
        learning_rate_init=learning_rate,          # tasa de aprendizaje inicial
        max_iter=500,                      # máximo de epoch
        early_stopping=True,               # parar si no mejora
        validation_fraction=0.15,          # % de train para validación
        n_iter_no_change=20,               # epoch sin mejora para parar
        random_state=42,
        verbose=True                       # ver el progreso
        )
        nn.fit(X_train_scaled, Y_train)
        nn_preds = nn.predict(X_test_scaled)
        nn_mse, nn_mae, nn_r2 = evaluate_model(nn, X_test_scaled, Y_test, "MLP Regressor")
        print(f"MLP Regressor - MSE: {nn_mse:.4f}, MAE: {nn_mae:.4f}, R²: {nn_r2:.4f}")
        return nn,nn_preds
