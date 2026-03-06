import eel
import sys
import os
import pandas as pd
from io import StringIO, BytesIO

# Prevent project.py from running its global code when imported
class DummyMain:
    def __init__(self):
        pass

sys.modules['__main__'] = DummyMain()  # Trick: fake __name__ == '__main__' so if __name__ block skips

# Now safe to import
from project import ShopWisePreprocessor, ShopWiseClustering

eel.init('web')

processed_data = None
clustered_data = None

@eel.expose
def preprocess(file_content, file_name):
    global processed_data
    try:
        if file_name.lower().endswith('.csv'):
            df = pd.read_csv(StringIO(file_content))
        else:
            df = pd.read_excel(BytesIO(file_content.encode()))

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

@eel.expose
def cluster():
    global clustered_data
    if processed_data is None:
        return "Error: Preprocess first"
    try:
        clust = ShopWiseClustering(processed_data)
        X = clust.prepare_clustering_data()
        optimal_k, _, _ = clust.find_optimal_clusters(X, max_clusters=8)
        n = optimal_k if optimal_k else 4
        clust.perform_clustering(n_clusters=n)
        clustered_data = clust.get_clustered_data()
        return f"Clustering done. Optimal: {n}"
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"Error: {str(e)}"

@eel.expose
def get_results():
    if clustered_data is None:
        return []
    return clustered_data.head(15)[['Customer ID', 'Age', 'Gender', 'Purchase Amount (USD)', 'Review Rating', 'Cluster']].to_dict('records')

@eel.expose
def get_dataset_info():
    if processed_data is None:
        return {"loaded": False}

    info = {
        "loaded": True,
        "rows": len(processed_data),
        "columns": len(processed_data.columns),
        "column_names": list(processed_data.columns),
        "first_5_rows": processed_data.head(5).to_dict('records')
    }

    return info

eel.start('index.html', size=(1300, 850), port=0)