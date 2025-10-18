import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Load the CSV file
raw_data = pd.read_csv("Housing.csv")

yes_no_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
raw_data[yes_no_columns] = raw_data[yes_no_columns].apply(lambda col: col.map({'yes': 1, 'no': 0}).astype('int64'))

# spliting furnishingstatus into boolean values based off states
raw_data = pd.get_dummies(raw_data, columns=['furnishingstatus'], drop_first=True) # drop_first = true, drops the 1st furnishing state to prevent redundant info

# Select only the bedrooms column
bedrooms = raw_data[['bedrooms']]

# Generate polynomial features (degree=2 for quadratic)
poly = PolynomialFeatures(degree=3, include_bias=False)
bedrooms_poly = poly.fit_transform(bedrooms)

# Add the quadratic term back to the DataFrame
print("")
print("bedrooms poly: ")
print(bedrooms_poly)
print("")
raw_data['bedrooms_squared'] = bedrooms_poly[:, 1]  # Extract the squared term
raw_data['bedrooms_cubed'] = bedrooms_poly[:, 2]  # Extract the cubed term
# including squared and cubed terms do not improve r2 value 

# Possible interactions 
raw_data['area_bedrooms'] = raw_data['area'] * raw_data['bedrooms'] 
raw_data['price_per_sqft'] = raw_data['price'] / raw_data['area']
raw_data['price_per_room'] = raw_data['price'] / raw_data['bedrooms']

# Define independent variables (features)
X = raw_data[['area_bedrooms', 'price_per_sqft', 'price_per_room', 'stories', 'mainroad', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']]

# scale independent values 
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add an intercept for VIF calculation
X_with_intercept = pd.DataFrame(X)
X_with_intercept['intercept'] = 1

# Calculate VIF
vif = pd.DataFrame()
vif["Feature"] = X_with_intercept.columns
vif["VIF"] = [variance_inflation_factor(X_with_intercept.values, i) for i in range(X_with_intercept.shape[1])]
print("")
print(vif)
print("")

# Define the dependent variable (target)
y = raw_data['price']

# Split into training and testing sets (50% train, 50% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=42)

# Ridge regression  
def ridge_regression (X, y, λ): 
    # Add a bias term (intercept) to the feature matrix
    X = np.c_[np.ones(X.shape[0]), X]  # Adds a column of ones to X for the intercept

    # Identity matrix (excluding the bias term)
    I = np.eye(X.shape[1])
    I[0, 0] = 0  # Do not regularize the bias term

    # Compute A and b
    A = X.T @ X + λ * np.eye(X.shape[1])  # Add regularization term (λ * I)
    A[0, 0] -= λ  # Do not regularize the intercept
    b = X.T @ y

    # Solve A @ β = b
    β = np.linalg.solve(A, b)
    return β
    

def predict(X, β):
    X = np.c_[np.ones(X.shape[0]), X]  # Add the bias term to X
    return X @ β

# list of alpha values to test
alphas = [0.01, 0.1, 1, 10, 100]

# Loop through each alpha and plot results
for α in alphas:
    # Compute Ridge Regression coefficients
    β = ridge_regression(X_train, y_train, α)
    
    # Make predictions
    y_train_pred = predict(X_train, β)
    y_test_pred = predict(X_test, β)
    
    # Evaluate the model
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Print R² scores for training and testing
    print("")
    print(f"Alpha: {α}")
    print(f"Training R²: {train_r2:.4f}, Testing R²: {test_r2:.4f}")
    
    # Plot actual vs predicted values for this alpha
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test, color='blue', label='Actual Values', alpha=0.6)  # Actual values
    plt.scatter(y_test, y_test_pred, color='orange', label='Predicted Values', alpha=0.6)  # Predicted values
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs Predicted Values (Alpha = {α})")
    plt.legend(loc='upper left')
    plt.show()



# finding trends between datasets manually 
compared = raw_data['price_per_room']
price = raw_data['price']

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(compared, price, alpha=0.7, color='blue')
plt.ylabel("Price")
plt.grid(True)
plt.show()



