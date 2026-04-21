# Jason Luu
# 3/3/26

##READ BEFORE PROCEEDING
    #kept having issues with reading the data from Arek's attached raw data
        #make sure that if you download it, put it in the same folder as your python file.
            #when downloaded from github, it downloads as an .xlsx file but has the .csv file format at the end
                #this is still considered an excel file, which stumped me yesterday (3/3/26)
                    #just tried to implement a file checker (with the help of AI).
    #shopping_trends and shopping_trends_updated


#added extra imports from arek's code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
import os
import glob

#clustering imports
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

warnings.filterwarnings('ignore')

print("Shopwise - Data Preprocessing and Integration")

#File finder
    #kept having  issues with my file reading, so I got help from AI for this part


#check current directory first for the files
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

#list of all files in current directory
print("\nFiles in current directory:")
all_files = os.listdir('.')
csv_files = []
excel_files = []

for file in all_files:
    if file.endswith('.csv'):
        csv_files.append(file)
        print(f"CSV: {file}")
    elif file.endswith(('.xlsx', '.xls')):
        excel_files.append(file)
        print(f"Excel: {file}")

#search for shopping trends files specifically
print("\nSearching for shopping trends files...")
shopping_files = []


#recursive search (search in all subfolders)
print("\nSearching in all subfolders...")
for root, dirs, files in os.walk('.'):
    for file in files:
        if 'shopping' in file.lower() or 'trends' in file.lower():
            full_path = os.path.join(root, file)
            print(f"Found: {full_path}")
            if full_path not in shopping_files:
                shopping_files.append(full_path)

print("loading file...")

dataset = None
load_attempts = []

#try all found shopping files first
if shopping_files:
    for file_path in shopping_files:
        try:
            print(f"\nTrying: {file_path}")
            if file_path.endswith('.csv'):
                dataset = pd.read_csv(file_path)
                print(f"Successfully loaded CSV: {file_path}")
                break
            elif file_path.endswith(('.xlsx', '.xls')):
                dataset = pd.read_excel(file_path)
                print(f"Successfully loaded Excel: {file_path}")
                break
            else:
                #try as CSV first, then Excel
                try:
                    dataset = pd.read_csv(file_path)
                    print(f"Successfully loaded as CSV: {file_path}")
                    break
                except:
                    dataset = pd.read_excel(file_path)
                    print(f"Successfully loaded as Excel: {file_path}")
                    break
        except Exception as e:
            print(f" Failed: {e}")
            load_attempts.append(f"{file_path}: {e}")

#if no shopping files found, try common filenames
if dataset is None:
    print("\nTrying common filenames...")

    common_names = [
        'shopping_trends.csv',
        'shopping_trends.xlsx',
        'shopping_trends_updated.csv',
        'shopping_trends_updated.xlsx',
        'shopping_trends (1).csv',
        'shopping_trends (1).xlsx',
        'shopping.csv',
        'trends.csv',
        'data.csv'
    ]

    for filename in common_names:
        if os.path.exists(filename):
            try:
                print(f"\nTrying: {filename}")
                if filename.endswith('.csv'):
                    dataset = pd.read_csv(filename)
                else:
                    dataset = pd.read_excel(filename)
                print(f"Successfully loaded: {filename}")
                break
            except Exception as e:
                print(f"Failed: {e}")
                continue


#continue with preprocessing
print("Dataset successfully loaded")

print(f"Shape: {dataset.shape[0]} rows × {dataset.shape[1]} columns")
print(f"\nFirst 5 rows:")
print(dataset.head())
print(f"\nDataset Info:")
print(f"Total entries: {len(dataset)}")
print(f"Columns: {list(dataset.columns)}")



#Jason's integration and preprocessing


class ShopWisePreprocessor:
    #data cleaning, encoding, scaling, feature "engineering"

    def __init__(self, data):
        #first initialize preprocessor with loaded data set

        self.raw_data = data.copy()
        self.processed_data = None
        self.label_encoders = {}
        self.scalers = {}
        self.imputers = {}
        self.encoding_mappings = {}

        print("Shopwise preprocessing")
        print(f"Raw data shape: {self.raw_data.shape}")
        print(f"Total samples: {len(self.raw_data)}")
        print(f"Total features: {len(self.raw_data.columns)}")

    def analyze_data_quality(self):
        #data quality analysis

        print("Data quality analysis")

        quality_report = {}

        #check for missing values
        print("\nMissing Value Check")
        missing_values = self.raw_data.isnull().sum()
        missing_percentage = (missing_values / len(self.raw_data)) * 100

        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Count': missing_values.values,
            'Missing %': missing_percentage.values
        }).sort_values('Missing %', ascending=False)

        print(missing_df[missing_df['Missing Count'] > 0])

        #check data types
        print("\nData Types Analysis")
        dtypes_df = pd.DataFrame({
            'Column': self.raw_data.columns,
            'Data Type': self.raw_data.dtypes.values,
            'Non-Null Count': self.raw_data.count().values
        })
        print(dtypes_df)

        #check for duplicates
        duplicates = self.raw_data.duplicated().sum()
        print(f"\nDuplicate Analysis")
        print(f"Duplicate rows: {duplicates} ({duplicates / len(self.raw_data) * 100:.2f}%)")

        #check for outliers in numerical columns
        print("\nOutlier Analysis")
        numerical_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        outlier_report = []

        for col in numerical_cols:
            Q1 = self.raw_data[col].quantile(0.25)
            Q3 = self.raw_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.raw_data[(self.raw_data[col] < lower_bound) |
                                     (self.raw_data[col] > upper_bound)][col].count()
            outlier_report.append({
                'Column': col,
                'Outliers': outliers,
                'Outlier %': (outliers / len(self.raw_data)) * 100,
                'Min': self.raw_data[col].min(),
                'Max': self.raw_data[col].max(),
                'Mean': self.raw_data[col].mean(),
                'Std': self.raw_data[col].std()
            })

        outlier_df = pd.DataFrame(outlier_report)
        print(outlier_df)

        #check cardinality of categorical columns
        print("\nCategorical Columns Cardinality")
        categorical_cols = self.raw_data.select_dtypes(include=['object']).columns
        cardinality_report = []

        for col in categorical_cols:
            unique_values = self.raw_data[col].nunique()
            cardinality_report.append({
                'Column': col,
                'Unique Values': unique_values,
                'Sample Values': ', '.join(map(str, self.raw_data[col].value_counts().head(3).index.tolist()))
            })

        cardinality_df = pd.DataFrame(cardinality_report)
        print(cardinality_df)

        #store report for l8r
        quality_report['missing'] = missing_df
        quality_report['dtypes'] = dtypes_df
        quality_report['duplicates'] = duplicates
        quality_report['outliers'] = outlier_df
        quality_report['cardinality'] = cardinality_df

        return quality_report

    def handle_missing_values(self, strategy='auto'):
        #missing values in the dataset (specifically auto, mean, median, mode, or drop)
            #auto is a method to choose best option "automatically"

        print("Handling missing values")

        df = self.raw_data.copy()

        #check for missing values
        missing_before = df.isnull().sum().sum()
        if missing_before == 0:
            print("No missing values found in the dataset")
            self.processed_data = df
            return df

        print(f"\nMissing values before handling: {missing_before}")
        print(df.isnull().sum()[df.isnull().sum() > 0])

        #separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        print(f"\nNumerical columns: {numerical_cols}")
        print(f"Categorical columns: {categorical_cols}")

        if strategy == 'auto':
            #auto-select is the best strategy for each column
            for col in df.columns:
                missing_pct = df[col].isnull().sum() / len(df) * 100

                if missing_pct > 0:
                    if missing_pct > 50:
                        print(f"Column '{col}' has {missing_pct:.1f}% missing - consider dropping")

                    if col in numerical_cols:
                        #for numerical: use median (more robust to outliers)
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                        print(f"Filled '{col}' with median: {median_val:.2f}")

                    else:  #categorical
                        #for categorical: use mode (most frequent)
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                        df[col].fillna(mode_val, inplace=True)
                        print(f"Filled '{col}' with mode: '{mode_val}'")

        elif strategy == 'drop':
            #drop rows with any missing values
            rows_before = len(df)
            df.dropna(inplace=True)
            rows_after = len(df)
            print(f"Dropped {rows_before - rows_after} rows with missing values")

        else:
            #use of specified strategy for all columns
            if strategy in ['mean', 'median']:
                imputer = SimpleImputer(strategy=strategy)
                df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
                print(f"Applied {strategy} imputation to numerical columns")
            else:
                imputer = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
                print(f"Applied mode imputation to categorical columns")

        missing_after = df.isnull().sum().sum()
        print(f"\nMissing values after handling: {missing_after}")

        self.processed_data = df
        return df

    def encode_categorical_variables(self, method='label'):
        #need this to encode our categorical variables into numbers.
            #machine learning in scikit needs this for smoother data processing. (like clustering.....)

        print("Encoding categorical variables")

        if self.processed_data is None:
            self.processed_data = self.raw_data.copy()

        df = self.processed_data.copy()

        #identify categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        #remove ID columns from encoding ( should be kept as identifiers)
        id_cols = [col for col in categorical_cols if 'id' in col.lower() or 'ID' in col]
        categorical_cols = [col for col in categorical_cols if col not in id_cols]

        print(f"\nCategorical columns to encode: {categorical_cols}")

        if method == 'label':
            #label encoding
            for col in categorical_cols:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))

                #store mapping for reference in scikit
                unique_values = df[col].unique()
                mapping = dict(zip(
                    self.label_encoders[col].classes_,
                    range(len(self.label_encoders[col].classes_))
                ))
                self.encoding_mappings[col] = mapping

                print(f"Label encoded '{col}' -> '{col}_encoded'")
                print(f"Mapping sample: {dict(list(mapping.items())[:3])}")

        elif method == 'onehot':
            # one-hot encoding is for assigning the values a number if there is no ordering/ranking
            df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
            print(f"One-hot encoding new shape: {df.shape}")

        print(f"\nTotal encoded features created: {len(categorical_cols)}")

        self.processed_data = df
        return df

    def scale_numerical_features(self, method='standard'):
        #for upscaling numerical features so model can perform better

        print("Scaling numerical features")

        if self.processed_data is None:
            self.processed_data = self.raw_data.copy()

        df = self.processed_data.copy()

        #identify numerical columns to scale
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        #don't scale ID columns or encoded categorical variables
        cols_to_scale = []
        for col in numerical_cols:
            if ('id' not in col.lower() and 'ID' not in col and
                    '_encoded' not in col and 'Cluster' not in col):
                cols_to_scale.append(col)

        print(f"\nNumerical columns to scale: {cols_to_scale}")

        if cols_to_scale:
            if method == 'standard':
                scaler = StandardScaler()
                df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                self.scalers['standard'] = scaler
                print(f"Applied StandardScaler (mean=0, std=1)")

            elif method == 'minmax':
                scaler = MinMaxScaler()
                df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                self.scalers['minmax'] = scaler
                print(f"Applied MinMaxScaler (range: 0 to 1)")

            #show scaling results
            print(f"\nScaling results (first 5 rows):")
            print(df[cols_to_scale].head())

        self.processed_data = df
        return df

    def create_feature_engineering(self):
        #feature engineering for creating new features (a bit self explanatory)

        print("Feature engineering")

        if self.processed_data is None:
            self.processed_data = self.raw_data.copy()

        df = self.processed_data.copy()

        #customer value metrics
        if 'Purchase Amount (USD)' in df.columns:
            # high value customer flag
            purchase_mean = df['Purchase Amount (USD)'].mean()
            df['Is_High_Value'] = (df['Purchase Amount (USD)'] > purchase_mean).astype(int)
            print(f"Created 'Is_High_Value' (threshold: ${purchase_mean:.2f})")

        #purchase frequency scoring
        if 'Previous Purchases' in df.columns:
            freq_mean = df['Previous Purchases'].mean()
            df['Is_Frequent_Buyer'] = (df['Previous Purchases'] > freq_mean).astype(int)
            print(f"Created 'Is_Frequent_Buyer' (threshold: {freq_mean:.1f} purchases)")

        #customer loyalty scoring
        if 'Subscription Status' in df.columns:
            df['Has_Subscription'] = (df['Subscription Status'] == 'Yes').astype(int)
            print(f"Created 'Has_Subscription' from Subscription Status")

        # discount usage
        if 'Discount Applied' in df.columns:
            df['Used_Discount'] = (df['Discount Applied'] == 'Yes').astype(int)
            print(f"Created 'Used_Discount'")

        # promo Usage
        if 'Promo Code Used' in df.columns:
            df['Used_Promo'] = (df['Promo Code Used'] == 'Yes').astype(int)
            print(f"Created 'Used_Promo'")

        # combined customer score
        if all(col in df.columns for col in ['Used_Discount', 'Used_Promo', 'Has_Subscription']):
            df['Customer_Engagement_Score'] = (
                    df['Used_Discount'] + df['Used_Promo'] + df['Has_Subscription']
            )
            print(f"Created 'Customer_Engagement_Score' (0-3 scale)")

        #categorization for age groups
        if 'Age' in df.columns:
            bins = [0, 25, 35, 50, 65, 100]
            labels = ['Young Adult', 'Adult', 'Middle Age', 'Senior', 'Elder']
            df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
            print(f"Created 'Age_Group' from Age")

            #encode age group
            age_encoder = LabelEncoder()
            df['Age_Group_encoded'] = age_encoder.fit_transform(df['Age_Group'].astype(str))
            print(f"     Age groups: {df['Age_Group'].unique().tolist()}")

        #spending per purchase
        if 'Purchase Amount (USD)' in df.columns and 'Previous Purchases' in df.columns:
            #avoid division by zero
            df['Spending_Per_Purchase'] = df['Purchase Amount (USD)'] / (df['Previous Purchases'] + 1)
            print(f"Created 'Spending_Per_Purchase'")

        self.processed_data = df
        return df

    def remove_outliers(self, method='iqr', threshold=1.5):
        #removing outliers after generating numerical features (mainly z score and interquartile range)

        print("Handling outliers")

        if self.processed_data is None:
            self.processed_data = self.raw_data.copy()

        df = self.processed_data.copy()

        #identify numerical columns (excluding encoded ones)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_check = [col for col in numerical_cols if '_encoded' not in col]

        outliers_removed = 0

        for col in cols_to_check:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    #cap the outliers instead of removing to keep data
                    df[col] = df[col].clip(lower_bound, upper_bound)
                    outliers_removed += len(outliers)
                    print(f"Capped {len(outliers)} outliers in '{col}'")

            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > threshold]
                if len(outliers) > 0:
                    #keep but cap at threshold
                    mean = df[col].mean()
                    std = df[col].std()
                    df[col] = df[col].clip(mean - threshold * std, mean + threshold * std)
                    outliers_removed += len(outliers)
                    print(f"Capped {len(outliers)} outliers in '{col}'")

        print(f"\nTotal outliers handled: {outliers_removed}")

        self.processed_data = df
        return df

    def validate_preprocessing(self):
        #validation and report builing

        print("Preprocessing validation")

        if self.processed_data is None:
            print("No processed data found. Run preprocessing steps first.")
            return

        #check for missing values
        missing = self.processed_data.isnull().sum().sum()
        print(f"\nMissing Values Check:")
        print(f"   Total missing values: {missing}")

        #check data types
        print(f"\nData types report:")
        dtypes_summary = self.processed_data.dtypes.value_counts()
        for dtype, count in dtypes_summary.items():
            print(f"   {dtype}: {count} columns")

        #check shape
        print(f"\nDataset shape:")
        print(f"   Rows: {self.processed_data.shape[0]}")
        print(f"   Columns: {self.processed_data.shape[1]}")
        print(f"   New features created: {self.processed_data.shape[1] - self.raw_data.shape[1]}")

        #check numerical ranges
        print(f"\nNumerical Features Range:")
        numerical_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols[:5]:  # Show first 5
            print(f"   ✓ {col}: [{self.processed_data[col].min():.2f}, {self.processed_data[col].max():.2f}]")

        #summary statistics
        print(f"\nSummary Statistics (numerical features):")
        print(self.processed_data.describe())

        #save preprocessing report
        report = {
            'original_shape': self.raw_data.shape,
            'processed_shape': self.processed_data.shape,
            'new_features': self.processed_data.shape[1] - self.raw_data.shape[1],
            'missing_values': missing,
            'encoded_columns': list(self.label_encoders.keys()),
            'scaled_columns': list(self.scalers.keys()) if self.scalers else []
        }

        print(f"\nPreprocessing validation complete!")
        return report

    def get_preprocessed_data(self):
        #output of full preprocessed dataset
        if self.processed_data is None:
            print("⚠No processed data found. Run preprocessing steps first.")
            return self.raw_data
        return self.processed_data

    def save_preprocessed_data(self, filename='preprocessed_shopping_trends.csv'):
        #save preprocessed data as csv file

        if self.processed_data is not None:
            self.processed_data.to_csv(filename, index=False)
            print(f"\nPreprocessed data saved to '{filename}'")
        else:
            print("No processed data to save.")


                                                    #running complete preprocessing pipeline

def run_complete_preprocessing_pipeline(data):
    #run entire pipeline with all steps

    print("Shopwise - complete preprocessing pipeline")


    #initialize preprocessor
    preprocessor = ShopWisePreprocessor(data)

    #data quality analysis
    quality_report = preprocessor.analyze_data_quality()

    #handle missing values
    preprocessor.handle_missing_values(strategy='auto')

    #encode categorical variables
    preprocessor.encode_categorical_variables(method='label')

    #scale numerical features
    preprocessor.scale_numerical_features(method='standard')

    #feature engineering
    preprocessor.create_feature_engineering()

    #handle outliers
    preprocessor.remove_outliers(method='iqr', threshold=1.5)

    #preprocessing validation
    validation_report = preprocessor.validate_preprocessing()

    #save processed data
    preprocessor.save_preprocessed_data('preprocessed_shopping_trends.csv')

    return preprocessor


                                                                    #clustering implemenetation

class ShopWiseClustering:

    def __init__(self, data):
        self.data = data
        self.cluster_labels = None
        self.kmeans_model = None
        self.cluster_features = None
        print("\nClustering Initialized")

    def prepare_clustering_data(self, feature_cols=None):
       #prepare data for clustering using relevant info
        print("\nPreparing data for clustering")

        #default features if none provided
        if feature_cols is None:
            #select numerical columns that would be good for clustering
            potential_features = ['Age', 'Purchase Amount (USD)', 'Review Rating',
                                  'Previous Purchases', 'Is_High_Value', 'Is_Frequent_Buyer',
                                  'Has_Subscription', 'Used_Discount', 'Used_Promo',
                                  'Customer_Engagement_Score', 'Spending_Per_Purchase']

            #only keep features that exist in the data
            feature_cols = [col for col in potential_features if col in self.data.columns]

            #add encoded categorical features (limit it)
            encoded_features = [col for col in self.data.columns if '_encoded' in col]
            feature_cols.extend(encoded_features[:5])  #take first 5 encoded features to prevent too many dimensions, prevent higher computational complexity

        self.cluster_features = feature_cols
        print(f"Selected features for clustering: {feature_cols}")

        #extracting features
        X = self.data[feature_cols].copy()

        #handle any remaining NaN values
        X = X.fillna(X.mean())

        return X

    def find_optimal_clusters(self, X, max_clusters=10):
        #optimal clusters using silhouette score (the -1 to 0 to 1 scoring of relations to other clusters
            #and davie-bouldin index, which averages the similarity between each cluster and its MOST similar one. (0 is best score for davies bouldin, approaching infinity is no good)

        print("\nFinding optimal number of clusters")

        silhouette_scores = []
        davies_bouldin_scores = []
        K_range = range(2, min(max_clusters + 1, len(X)))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            silhouette_avg = silhouette_score(X, labels)
            davies_bouldin_avg = davies_bouldin_score(X, labels)

            silhouette_scores.append(silhouette_avg)
            davies_bouldin_scores.append(davies_bouldin_avg)
            print(f"  k={k}: Silhouette={silhouette_avg:.4f}, Davies-Bouldin={davies_bouldin_avg:.4f}")

        #plot the scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(K_range, silhouette_scores, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Silhouette Score by k')
        ax1.grid(True)

        ax2.plot(K_range, davies_bouldin_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Davies-Bouldin Index')
        ax2.set_title('Davies-Bouldin Index by k')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        #return optimal k (highest silhouette or lowest davies-nouldin)
        optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
        optimal_k_db = K_range[np.argmin(davies_bouldin_scores)]

        print(f"\nOptimal k by Silhouette Score: {optimal_k_silhouette}")
        print(f"Optimal k by Davies-Bouldin Index: {optimal_k_db}")

        return optimal_k_silhouette, silhouette_scores, davies_bouldin_scores

    def perform_clustering(self, n_clusters=4, random_state=42):
        #now perfrming clustering with a specific number of clusters

        print(f"\nPerforming K-Means clustering with k={n_clusters}")

        #prepare data for clustering, easier call this way
        X = self.prepare_clustering_data()

        #perform clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_labels = self.kmeans_model.fit_predict(X)

        #add cluster labels to data
        self.data['Cluster'] = self.cluster_labels

        #clustering evaluation
        silhouette_avg = silhouette_score(X, self.cluster_labels)
        davies_bouldin_avg = davies_bouldin_score(X, self.cluster_labels)

        print(f"\nClustering Results:")
        print(f"  Silhouette Score: {silhouette_avg:.4f}")
        print(f"  Davies-Bouldin Index: {davies_bouldin_avg:.4f}")

        #cluster distribution
        cluster_dist = self.data['Cluster'].value_counts().sort_index()
        print(f"\nCluster Distribution:")
        for cluster, count in cluster_dist.items():
            percentage = (count / len(self.data)) * 100
            print(f"  Cluster {cluster}: {count} customers ({percentage:.1f}%)")

        return self.cluster_labels

    def analyze_clusters(self):
        #characteristics of clusters
        print("\nAnalyzing Cluster Characteristics")

        #selecting columns for analysis
        analysis_cols = ['Age', 'Purchase Amount (USD)', 'Review Rating',
                         'Previous Purchases', 'Is_High_Value', 'Is_Frequent_Buyer']
        analysis_cols = [col for col in analysis_cols if col in self.data.columns]

        #group by cluster and calculate means
        cluster_profile = self.data.groupby('Cluster')[analysis_cols].mean()
        print("\nCluster Profiles (mean values):")
        print(cluster_profile)

        #create customer personas (general identifiers for them)
        print("\nCustomer Personas by Cluster:")
        for cluster in range(len(cluster_profile)):
            profile = cluster_profile.loc[cluster]

            #determine characteristics
            if 'Purchase Amount (USD)' in profile:
                if profile['Purchase Amount (USD)'] > self.data['Purchase Amount (USD)'].mean():
                    spending = "High spender"
                else:
                    spending = "Budget conscious"

            if 'Review Rating' in profile:
                if profile['Review Rating'] > 4.0:
                    satisfaction = "Highly satisfied"
                elif profile['Review Rating'] > 3.0:
                    satisfaction = "Moderately satisfied"
                else:
                    satisfaction = "At-risk"

            if 'Previous Purchases' in profile:
                if profile['Previous Purchases'] > self.data['Previous Purchases'].mean():
                    loyalty = "Loyal"
                else:
                    loyalty = "Occasional"

            print(f"\n  Cluster {cluster}: {satisfaction} {spending} {loyalty}")
            for col in analysis_cols:
                if col in profile:
                    print(f"    • {col}: {profile[col]:.2f}")

        #visualize clusters
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        #age distribution by cluster
        if 'Age' in self.data.columns:
            self.data.boxplot(column='Age', by='Cluster', ax=axes[0, 0])
            axes[0, 0].set_title('Age Distribution by Cluster')

        #purchase amount by cluster
        if 'Purchase Amount (USD)' in self.data.columns:
            self.data.boxplot(column='Purchase Amount (USD)', by='Cluster', ax=axes[0, 1])
            axes[0, 1].set_title('Purchase Amount by Cluster')

        #review rating by cluster
        if 'Review Rating' in self.data.columns:
            self.data.boxplot(column='Review Rating', by='Cluster', ax=axes[1, 0])
            axes[1, 0].set_title('Review Rating by Cluster')

        #previous purchases by cluster
        if 'Previous Purchases' in self.data.columns:
            self.data.boxplot(column='Previous Purchases', by='Cluster', ax=axes[1, 1])
            axes[1, 1].set_title('Previous Purchases by Cluster')

        plt.suptitle('Customer Segments Analysis')
        plt.tight_layout()
        plt.show()

        return cluster_profile

    def get_clustered_data(self):
       #cluster labels
        return self.data


#main method

if __name__ == "__main__":
    #run complete preprocessing pipeline
    preprocessor = run_complete_preprocessing_pipeline(dataset)

    #get the preprocessed data
    processed_data = preprocessor.get_preprocessed_data()

    #display final results
    print("\nCompleted preprocessing - Summary")
    print(f"\nOriginal dataset shape: {dataset.shape}")
    print(f"Processed dataset shape: {processed_data.shape}")
    print(f"New features created: {processed_data.shape[1] - dataset.shape[1]}")

    # Group columns by type
    numerical_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = processed_data.select_dtypes(include=['object']).columns.tolist()

    print(f"\nColumn Summary:")
    print(f"Numerical columns: {len(numerical_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")

    #sample of processed data
    print("\nSample of preprocessed data (first 5 rows):")
    print(processed_data.head())

    print("\nThe preprocessed data is ready for clustering and classification!")

                                                                                    #clustering portion of main method 3/4/26

    print("Run Clustering Analysis")

    #initialize clustering
    clustering = ShopWiseClustering(processed_data)

    #prepare data and find optimal clusters
    X = clustering.prepare_clustering_data()
    optimal_k, sil_scores, db_scores = clustering.find_optimal_clusters(X, max_clusters=8)

    #perform clustering with optimal k (or use 4 as default since it is balanced for shopping trends, per google...)
    n_clusters = optimal_k if optimal_k else 4
    clustering.perform_clustering(n_clusters=n_clusters)

    #analyze clusters
    cluster_profiles = clustering.analyze_clusters()

    #get data with cluster labels
    clustered_data = clustering.get_clustered_data()

    #save clustered data
    clustered_data.to_csv('clustered_shopping_trends.csv', index=False)
    print("\nClustered data saved to 'clustered_shopping_trends.csv'")

    print("\nClustering pipeline completed")