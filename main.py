import eel
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Must be BEFORE importing pyplot — suppresses all popup windows
import matplotlib.pyplot as plt
import base64
from io import StringIO, BytesIO
from io import BytesIO as BIO

# Prevent project.py main block from executing
class DummyMain:
    pass

sys.modules['__main__'] = DummyMain()

from project import ShopWisePreprocessor, ShopWiseClustering

eel.init('web')

processed_data = None
clustered_data = None
original_data = None


# -----------------------------
# PREPROCESS DATA
# -----------------------------
@eel.expose
def preprocess(file_content, file_name):

    global processed_data
    global original_data

    try:

        df = pd.read_csv(StringIO(file_content))
        original_data = df.copy()

        prep = ShopWisePreprocessor(df)

        prep.handle_missing_values('auto')
        prep.encode_categorical_variables('label')
        prep.scale_numerical_features('standard')
        prep.create_feature_engineering()
        prep.remove_outliers('iqr')

        processed_data = prep.get_preprocessed_data()

        return f"Success! {len(processed_data)} rows processed."

    except Exception as e:

        import traceback
        print(traceback.format_exc())

        return f"Error: {str(e)}"


# -----------------------------
# RUN CLUSTERING
# -----------------------------
@eel.expose
def cluster():

    global clustered_data
    global processed_data
    global original_data

    if processed_data is None:
        return "Error: preprocess dataset first"

    try:

        clust = ShopWiseClustering(processed_data)

        X = clust.prepare_clustering_data()

        optimal_k, _, _ = clust.find_optimal_clusters(X, max_clusters=8)

        n_clusters = optimal_k if optimal_k else 4

        clust.perform_clustering(n_clusters=n_clusters)

        clustered_data = clust.get_clustered_data()

        # attach clusters to original data for display
        original_data['Cluster'] = clustered_data['Cluster']

        return f"Clustering done. Optimal clusters: {n_clusters}"

    except Exception as e:

        import traceback
        print(traceback.format_exc())

        return f"Error: {str(e)}"


# -----------------------------
# GET RESULTS (REAL VALUES)
# -----------------------------
@eel.expose
def get_results():

    global original_data

    if original_data is None:
        return []

    df = original_data.copy()

    return df.head(50)[[
        'Customer ID',
        'Age',
        'Gender',
        'Purchase Amount (USD)',
        'Review Rating',
        'Cluster'
    ]].to_dict('records')


# -----------------------------
# SORT RESULTS
# -----------------------------
@eel.expose
def get_sorted_results(column):

    global original_data

    if original_data is None:
        return []

    df = original_data.sort_values(by=column)

    return df.head(50)[[
        'Customer ID',
        'Age',
        'Gender',
        'Purchase Amount (USD)',
        'Review Rating',
        'Cluster'
    ]].to_dict('records')


# -----------------------------
# DATASET SUMMARY
# -----------------------------
@eel.expose
def get_dataset_info():

    global original_data

    if original_data is None:
        return {"loaded": False}

    return {
        "loaded": True,
        "rows": len(original_data),
        "columns": len(original_data.columns),
        "column_names": list(original_data.columns)
    }


# -----------------------------
# CLUSTER DISTRIBUTION
# -----------------------------
@eel.expose
def get_cluster_stats():

    global original_data

    if original_data is None:
        return {}

    counts = original_data['Cluster'].value_counts().sort_index()

    return counts.to_dict()


# -----------------------------
# CLUSTER COUNT (for dashboard card)
# -----------------------------
@eel.expose
def get_cluster_count():

    global original_data

    if original_data is None or 'Cluster' not in original_data.columns:
        return 0

    return int(original_data['Cluster'].nunique())


# -----------------------------
# AVERAGE REVIEW RATING (for dashboard card)
# -----------------------------
@eel.expose
def get_avg_rating():

    global original_data

    if original_data is None or 'Review Rating' not in original_data.columns:
        return 0.0

    return round(float(original_data['Review Rating'].mean()), 2)


# -----------------------------
# CLUSTER PROFILES (for Customer Segments page)
# Returns per-cluster aggregated stats for rich persona cards
# -----------------------------
@eel.expose
def get_cluster_profiles():

    global original_data

    if original_data is None or 'Cluster' not in original_data.columns:
        return []

    profiles = []
    for cluster_id, group in original_data.groupby('Cluster'):
        profile = {
            "cluster": int(cluster_id),
            "count": int(len(group)),
            "avg_age": round(float(group['Age'].mean()), 1) if 'Age' in group.columns else None,
            "avg_spend": round(float(group['Purchase Amount (USD)'].mean()), 2) if 'Purchase Amount (USD)' in group.columns else None,
            "avg_rating": round(float(group['Review Rating'].mean()), 2) if 'Review Rating' in group.columns else None,
            "top_gender": group['Gender'].mode()[0] if 'Gender' in group.columns and len(group) > 0 else None,
        }
        profiles.append(profile)

    return profiles


# -----------------------------
# GENERATE GRAPH (kept for legacy; not used by frontend charts anymore)
# -----------------------------
@eel.expose
def get_cluster_graph():

    global original_data

    if original_data is None:
        return ""

    plt.figure(figsize=(6, 4))

    original_data['Cluster'].value_counts().sort_index().plot(kind='bar')

    plt.title("Customers per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Customers")

    buffer = BIO()
    plt.savefig(buffer, format="png")
    plt.close()

    buffer.seek(0)

    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return img_base64


# -----------------------------
# START APP
# -----------------------------
eel.start('index.html', size=(1300, 850), port=0)