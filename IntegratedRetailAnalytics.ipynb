import pandas as pd # Import pandas for data manipulation and analysis
import numpy as np # Import numpy for numerical operations
import matplotlib.pyplot as plt # Import matplotlib for basic plotting
import seaborn as sns # Import seaborn for statistical data visualization
from sklearn.cluster import KMeans # Import KMeans for store segmentation
from sklearn.preprocessing import StandardScaler # Import scaler to normalize data for clustering
from sklearn.ensemble import IsolationForest # Import Isolation Forest for anomaly detection
from xgboost import XGBRegressor # Import XGBoost for advanced demand forecasting
from sklearn.metrics import mean_squared_error, silhouette_score # Import metrics to evaluate models
import warnings # Import warnings to manage library alerts

# Initial Configuration
warnings.filterwarnings('ignore') # Suppress warnings to keep the output clean

def load_data_robustly(url):
    """Utility function to fetch Google Sheets directly into a DataFrame."""
    try:
        sheet_id = url.split('/d/')[1].split('/')[0] # Extract the unique ID from the Google Sheet URL
        csv_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv' # Construct direct CSV export link
        df = pd.read_csv(csv_url) # Read the CSV data from the constructed URL
        df.columns = df.columns.str.strip() # Remove any accidental hidden leading/trailing spaces in headers
        return df # Return the loaded dataframe
    except Exception as e:
        print(f"Error loading from {url}: {e}") # Print error message if download fails
        return None # Return None to trigger error handling

# 1. DATA LOADING
print("Step 1: Fetching datasets...")
features_url = "https://docs.google.com/spreadsheets/d/1UZY23n6Ef7ambSBW2Qlvcb9swDKHs0cwwouSz814wX4/edit"
sales_url = "https://docs.google.com/spreadsheets/d/1IQ3VZ4MW5UEsRe4jYoNlf1Bv6kJHltSVxm65FsHXC8g/edit"
stores_url = "https://docs.google.com/spreadsheets/d/1PPKc_ZibOBn6_ofSlMXOKBEvJ63uT-3slDFqsW5DBms/edit"

# Load each dataset using the robust utility function
features = load_data_robustly(features_url) # Load regional features (CPI, Fuel Price, etc.)
sales = load_data_robustly(sales_url) # Load historical weekly sales data
stores = load_data_robustly(stores_url) # Load store metadata (Type, Size)

if features is None or sales is None or stores is None:
    print("\n[!] Failed to load. Check Google Sheet sharing permissions.") # Error check for restricted links
else:
    print("✅ All datasets loaded successfully.")

    # 2. STANDARDIZING COLUMN NAMES
    # This section prevents KeyErrors caused by inconsistent casing in datasets
    for dataframe in [features, sales, stores]:
        dataframe.columns = dataframe.columns.str.replace(' ', '_').str.title() # Convert 'Weekly Sales' to 'Weekly_Sales'
        dataframe.rename(columns={'Isholiday': 'IsHoliday', 'Cpi': 'CPI'}, inplace=True) # Manually fix specific acronyms

    # 3. MERGING
    print("Step 2: Merging Data...")
    df = pd.merge(sales, features, on=['Store', 'Date'], how='left') # Join sales with regional features on Store and Date
    
    # Check if 'IsHoliday' was duplicated during the merge and consolidate
    if 'IsHoliday_x' in df.columns:
        df['IsHoliday'] = df['IsHoliday_x'] # Keep the consolidated version
        df.drop(['IsHoliday_x', 'IsHoliday_y'], axis=1, inplace=True) # Drop the redundant duplicate columns
    
    df = pd.merge(df, stores, on='Store', how='left') # Final merge to include Store metadata (Type/Size)

    # 4. PREPROCESSING & DATA CLEANING
    print("Step 3: Preprocessing...")
    # Fix date formatting: Handles DD/MM/YYYY and Mixed formats found in international retail data
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce') 
    
    # Fill missing values in Markdowns with 0 (assuming no markdown was active)
    md_cols = ['Markdown1', 'Markdown2', 'Markdown3', 'Markdown4', 'Markdown5']
    for col in md_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Convert to numbers and fill nulls
            
    # Impute missing economic indicators using the median value (robust to outliers)
    df['CPI'] = df['CPI'].fillna(df['CPI'].median()) # Handle missing Consumer Price Index values
    df['Unemployment'] = df['Unemployment'].fillna(df['Unemployment'].median()) # Handle missing Unemployment data

    # 5. ANOMALY DETECTION (Component 1)
    print("Step 4: Running Anomaly Detection...")
    target_col = 'Weekly_Sales' if 'Weekly_Sales' in df.columns else 'Weekly_sales' # Identify target column name
    
    # Isolation Forest isolates observations by randomly selecting a feature and split value
    iso = IsolationForest(contamination=0.01, random_state=42) # Set 1% as the expected outlier/anomaly rate
    df['Anomaly'] = iso.fit_predict(df[[target_col]].fillna(0)) # Label each row as 1 (normal) or -1 (anomaly)
    df_clean = df[df['Anomaly'] == 1].reset_index(drop=True) # Filter out the detected anomalies for a cleaner model

    # 6. STORE SEGMENTATION (Component 2)
    print("Step 5: Store Segmentation...")
    # Group data by store to calculate average characteristics for clustering
    store_stats = df_clean.groupby('Store').agg({
        target_col: 'mean', # Average sales per store
        'Size': 'first', # Physical size of the store
        'Temperature': 'mean' # Average regional temperature
    }).reset_index()

    scaler = StandardScaler() # Initialize standard scaler
    scaled = scaler.fit_transform(store_stats.drop('Store', axis=1)) # Scale features to have mean=0 and variance=1
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # Initialize K-Means to find 3 store segments
    store_stats['Cluster'] = kmeans.fit_predict(scaled) # Assign each store to a specific Cluster ID

    # 7. DEMAND FORECASTING (Component 3)
    print("Step 6: Training Forecasting Model...")
    # Map the Cluster labels back to the main transactional dataset
    df_final = pd.merge(df_clean, store_stats[['Store', 'Cluster']], on='Store', how='left') 
    
    # Convert categorical 'Type' into binary columns (One-Hot Encoding)
    if 'Type' in df_final.columns:
        df_final = pd.get_dummies(df_final, columns=['Type'])

    # Extract temporal features to help the model learn seasonality
    df_final['Week'] = df_final['Date'].dt.isocalendar().week.astype(int) # Extract week of the year
    df_final['Year'] = df_final['Date'].dt.year # Extract the year

    # Define the final list of input features for the Machine Learning model
    features_list = ['Store', 'Dept', 'CPI', 'Unemployment', 'Fuel_Price', 'Size', 'Cluster', 'Week', 'Year']
    features_list += [c for c in df_final.columns if 'Type_' in c] # Dynamically add the Type_A, Type_B columns
    
    X = df_final[features_list].fillna(0) # Prepare feature matrix (input)
    y = df_final[target_col] # Prepare target vector (sales to be predicted)

    # Split data chronologically (80% for training, 20% for testing performance)
    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    # Initialize and train the XGBoost Regressor (Gradient Boosted Decision Trees)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train) # Fit the model to the training data

    # 8. OUTPUT & EVALUATION
    print("\n" + "="*45)
    print("FINAL ANALYSIS COMPLETED")
    print("="*45)
    # Calculate Silhouette Score: Measures how similar a store is to its own cluster compared to others
    print(f"Clustering Quality (Silhouette): {silhouette_score(scaled, store_stats['Cluster']):.4f}")
    # Calculate RMSE: Measures the average magnitude of the prediction error
    print(f"Forecasting Accuracy (RMSE):     {np.sqrt(mean_squared_error(y_test, model.predict(X_test))):.2f}")
    print("="*45)

# --- VISUALIZATION ADD-ON ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=store_stats, x='Size', y=target_col, hue='Cluster', palette='viridis', s=100)
plt.title('Store Segments: Size vs Average Sales')
plt.xlabel('Store Size')
plt.ylabel('Average Weekly Sales')
plt.legend(title='Cluster')
plt.show()
