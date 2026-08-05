import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ===================== 1. CREATE SAMPLE DATASET =====================

def create_sample_dataset():
    """Create a sample dataset with outliers"""
    np.random.seed(42)
    
    # Normal data
    normal_data = np.random.normal(50, 15, 200)
    
    # Adding outliers
    outliers = np.array([2, 95, 110, 120, 130, 5, 8, 0, 140, 150])
    
    # Combine data
    data = np.concatenate([normal_data, outliers])
    np.random.shuffle(data)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': data.astype(int),
        'Salary': data * 1000 + np.random.normal(0, 5000, len(data))
    })
    
    # Add some categorical data
    df['Department'] = np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], len(data))
    
    return df

df = create_sample_dataset()
print("Dataset Info:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nBasic Statistics:")
print(df.describe())

# ===================== 2. VISUALIZATION METHODS =====================

def visualize_outliers(df, column='Age'):
    """Create various visualizations to detect outliers"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Box Plot
    axes[0, 0].boxplot(df[column])
    axes[0, 0].set_title('Box Plot', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel(column)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Scatter Plot
    axes[0, 1].scatter(range(len(df)), df[column], alpha=0.6)
    axes[0, 1].axhline(y=df[column].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 1].set_title('Scatter Plot', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel(column)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Histogram
    axes[0, 2].hist(df[column], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(x=df[column].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 2].axvline(x=df[column].median(), color='green', linestyle='--', label='Median')
    axes[0, 2].set_title('Histogram', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel(column)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Violin Plot
    axes[1, 0].violinplot(df[column], showmeans=True, showmedians=True)
    axes[1, 0].set_title('Violin Plot', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel(column)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Density Plot (KDE)
    sns.kdeplot(data=df[column], ax=axes[1, 1], fill=True)
    axes[1, 1].axvline(x=df[column].mean(), color='red', linestyle='--', label='Mean')
    axes[1, 1].axvline(x=df[column].median(), color='green', linestyle='--', label='Median')
    axes[1, 1].set_title('Density Plot (KDE)', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. QQ Plot
    stats.probplot(df[column], dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('QQ Plot', fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Outlier Detection Visualizations for {column}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

visualize_outliers(df, 'Age')

# ===================== 3. STATISTICAL DETECTION METHODS =====================

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    print(f"\nIQR Method Results for {column}:")
    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower Bound: {lower_bound:.2f}")
    print(f"Upper Bound: {upper_bound:.2f}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Outlier values: {sorted(outliers[column].tolist())}")
    
    return outliers

def detect_outliers_zscore(df, column, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(df[column]))
    outliers = df[z_scores > threshold]
    
    print(f"\nZ-Score Method Results for {column} (threshold={threshold}):")
    print(f"Mean: {df[column].mean():.2f}")
    print(f"Std Dev: {df[column].std():.2f}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Outlier values: {sorted(outliers[column].tolist())}")
    
    return outliers

def detect_outliers_percentile(df, column, lower_percentile=1, upper_percentile=99):
    """Detect outliers using percentile method"""
    lower_bound = np.percentile(df[column], lower_percentile)
    upper_bound = np.percentile(df[column], upper_percentile)
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    print(f"\nPercentile Method Results for {column}:")
    print(f"Lower Bound ({lower_percentile}th percentile): {lower_bound:.2f}")
    print(f"Upper Bound ({upper_percentile}th percentile): {upper_bound:.2f}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Outlier values: {sorted(outliers[column].tolist())}")
    
    return outliers

# Apply detection methods
print("=" * 60)
print("OUTLIER DETECTION METHODS")
print("=" * 60)

outliers_iqr = detect_outliers_iqr(df, 'Age')
outliers_zscore = detect_outliers_zscore(df, 'Age')
outliers_percentile = detect_outliers_percentile(df, 'Age')

# ===================== 4. MACHINE LEARNING DETECTION METHODS =====================

def detect_outliers_ml(df, column):
    """Detect outliers using ML methods"""
    
    X = df[[column]].values
    
    # 1. Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_labels = iso_forest.fit_predict(X)
    iso_outliers = df[iso_labels == -1]
    
    # 2. Local Outlier Factor
    lof = LocalOutlierFactor(contamination=0.1)
    lof_labels = lof.fit_predict(X)
    lof_outliers = df[lof_labels == -1]
    
    # 3. One-Class SVM
    svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
    svm_labels = svm.fit_predict(X)
    svm_outliers = df[svm_labels == -1]
    
    print(f"\nMachine Learning Methods Results for {column}:")
    print(f"Isolation Forest outliers: {len(iso_outliers)}")
    print(f"LOF outliers: {len(lof_outliers)}")
    print(f"One-Class SVM outliers: {len(svm_outliers)}")
    
    return iso_outliers, lof_outliers, svm_outliers

iso_outliers, lof_outliers, svm_outliers = detect_outliers_ml(df, 'Age')

# ===================== 5. OUTLIER HANDLING METHODS =====================

def handle_outliers_remove(df, column):
    """Method 1: Remove outliers"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    cleaned_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    print(f"\nRemoval Method for {column}:")
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Removed {len(df) - len(cleaned_df)} rows")
    
    return cleaned_df

def handle_outliers_replace(df, column, method='median'):
    """Method 2: Replace with mean or median"""
    if method == 'mean':
        replacement = df[column].mean()
    else:  # median
        replacement = df[column].median()
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    cleaned_df = df.copy()
    cleaned_df.loc[(cleaned_df[column] < lower_bound) | (cleaned_df[column] > upper_bound), column] = replacement
    
    print(f"\nReplacement Method for {column} (using {method}):")
    print(f"Replacement value: {replacement:.2f}")
    print(f"Modified {len(df) - len(cleaned_df)} outliers")
    
    return cleaned_df

def handle_outliers_winsorize(df, column, limits=(0.01, 0.99)):
    """Method 3: Winsorization/Capping"""
    lower_limit = np.percentile(df[column], limits[0] * 100)
    upper_limit = np.percentile(df[column], limits[1] * 100)
    
    cleaned_df = df.copy()
    cleaned_df[column] = cleaned_df[column].clip(lower=lower_limit, upper=upper_limit)
    
    print(f"\nWinsorization Method for {column}:")
    print(f"Lower limit: {lower_limit:.2f}")
    print(f"Upper limit: {upper_limit:.2f}")
    print(f"Capped values at {limits[0]*100}% and {limits[1]*100}% percentiles")
    
    # Show changed values
    changed = df[(df[column] < lower_limit) | (df[column] > upper_limit)]
    if len(changed) > 0:
        print(f"Modified {len(changed)} outliers")
    
    return cleaned_df

def handle_outliers_log_transform(df, column):
    """Method 4: Log transformation"""
    # Add small constant to handle zeros
    min_val = df[column].min()
    if min_val <= 0:
        df_clean = df.copy()
        df_clean[column] = df_clean[column] - min_val + 1
    
    df_clean = df.copy()
    df_clean[f'{column}_log'] = np.log1p(df_clean[column])
    
    print(f"\nLog Transformation for {column}:")
    print(f"Original range: [{df[column].min():.2f}, {df[column].max():.2f}]")
    print(f"Transformed range: [{df_clean[f'{column}_log'].min():.2f}, {df_clean[f'{column}_log'].max():.2f}]")
    
    return df_clean

def handle_outliers_robust_scale(df, column):
    """Method 5: Robust scaling"""
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df[[column]])
    
    df_scaled = df.copy()
    df_scaled[f'{column}_robust_scaled'] = scaled_data
    
    print(f"\nRobust Scaling for {column}:")
    print(f"Median: {np.median(df[column]):.2f}")
    print(f"IQR: {stats.iqr(df[column]):.2f}")
    print(f"Scaled range: [{df_scaled[f'{column}_robust_scaled'].min():.2f}, {df_scaled[f'{column}_robust_scaled'].max():.2f}]")
    
    return df_scaled

# Apply handling methods
print("\n" + "=" * 60)
print("OUTLIER HANDLING METHODS")
print("=" * 60)

# 1. Remove outliers
df_removed = handle_outliers_remove(df, 'Age')

# 2. Replace with median
df_replaced = handle_outliers_replace(df, 'Age', method='median')

# 3. Winsorization
df_winsorized = handle_outliers_winsorize(df, 'Age', limits=(0.01, 0.99))

# 4. Log transformation
df_log = handle_outliers_log_transform(df, 'Age')

# 5. Robust scaling
df_robust = handle_outliers_robust_scale(df, 'Age')

# ===================== 6. COMPARISON VISUALIZATION =====================

def compare_outlier_handling(df, column='Age'):
    """Compare different outlier handling methods visually"""
    
    # Prepare data for comparison
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Apply different methods
    df_removed = handle_outliers_remove(df, column)
    df_replaced = handle_outliers_replace(df, column)
    df_winsorized = handle_outliers_winsorize(df, column)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original data
    axes[0, 0].boxplot(df[column])
    axes[0, 0].set_title('Original Data', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel(column)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.5, 0.95, f'Outliers: {len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])}', 
                   transform=axes[0, 0].transAxes, ha='center', fontsize=10)
    
    # After removal
    axes[0, 1].boxplot(df_removed[column])
    axes[0, 1].set_title('After Removal', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel(column)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(0.5, 0.95, f'Removed: {len(df) - len(df_removed)}', 
                   transform=axes[0, 1].transAxes, ha='center', fontsize=10)
    
    # After replacement
    axes[0, 2].boxplot(df_replaced[column])
    axes[0, 2].set_title('After Replacement', fontsize=12, fontweight='bold')
    axes[0, 2].set_ylabel(column)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].text(0.5, 0.95, f'Replaced with median', 
                   transform=axes[0, 2].transAxes, ha='center', fontsize=10)
    
    # After winsorization
    axes[1, 0].boxplot(df_winsorized[column])
    axes[1, 0].set_title('After Winsorization', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel(column)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(0.5, 0.95, f'Capped at 1% and 99%', 
                   transform=axes[1, 0].transAxes, ha='center', fontsize=10)
    
    # Comparison of statistics
    stats_data = {
        'Original': [df[column].mean(), df[column].std(), len(df)],
        'Removal': [df_removed[column].mean(), df_removed[column].std(), len(df_removed)],
        'Replacement': [df_replaced[column].mean(), df_replaced[column].std(), len(df_replaced)],
        'Winsorized': [df_winsorized[column].mean(), df_winsorized[column].std(), len(df_winsorized)]
    }
    
    # Show statistics table
    stats_df = pd.DataFrame(stats_data, index=['Mean', 'Std Dev', 'Count']).round(2)
    axes[1, 1].axis('off')
    axes[1, 1].table(cellText=stats_df.values, 
                     rowLabels=stats_df.index,
                     colLabels=stats_df.columns,
                     cellLoc='center',
                     loc='center')
    axes[1, 1].set_title('Statistics Comparison', fontsize=12, fontweight='bold')
    
    # Side-by-side comparison histogram
    axes[1, 2].hist(df[column], bins=20, alpha=0.3, label='Original', density=True)
    axes[1, 2].hist(df_removed[column], bins=20, alpha=0.3, label='Removal', density=True)
    axes[1, 2].hist(df_replaced[column], bins=20, alpha=0.3, label='Replacement', density=True)
    axes[1, 2].hist(df_winsorized[column], bins=20, alpha=0.3, label='Winsorized', density=True)
    axes[1, 2].set_title('Distribution Comparison', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel(column)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Outlier Handling Methods Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Compare methods
compare_outlier_handling(df, 'Age')

# ===================== 7. COMPLETE PIPELINE =====================

class OutlierHandler:
    """Complete outlier handling pipeline"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.results = {}
    
    def detect_all_methods(self, columns):
        """Detect outliers using all methods"""
        for col in columns:
            print(f"\n{'='*50}")
            print(f"Analyzing column: {col}")
            print(f"{'='*50}")
            
            self.results[col] = {}
            self.results[col]['iqr'] = detect_outliers_iqr(self.df, col)
            self.results[col]['zscore'] = detect_outliers_zscore(self.df, col)
            self.results[col]['percentile'] = detect_outliers_percentile(self.df, col)
    
    def handle_outliers(self, columns, method='winsorize'):
        """Handle outliers with specified method"""
        for col in columns:
            if method == 'remove':
                self.df = handle_outliers_remove(self.df, col)
            elif method == 'replace':
                self.df = handle_outliers_replace(self.df, col)
            elif method == 'winsorize':
                self.df = handle_outliers_winsorize(self.df, col)
            elif method == 'log':
                self.df = handle_outliers_log_transform(self.df, col)
            elif method == 'robust':
                self.df = handle_outliers_robust_scale(self.df, col)
        
        return self.df
    
    def visualize_all(self, columns):
        """Visualize outlier detection for multiple columns"""
        for col in columns:
            visualize_outliers(self.df, col)

# Example usage of the complete pipeline
handler = OutlierHandler(df)
handler.detect_all_methods(['Age', 'Salary'])
handler.visualize_all(['Age'])

# Handle outliers
cleaned_df = handler.handle_outliers(['Age'], method='winsorize')
print(f"\nFinal cleaned dataset shape: {cleaned_df.shape}")

print("\n" + "=" * 60)
print("OUTLIER HANDLING COMPLETE!")
print("=" * 60)