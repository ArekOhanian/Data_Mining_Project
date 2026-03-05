import eel

# Import your code from project.py
from project import ShopWisePreprocessor, ShopWiseClustering

eel.init('web')   # looks in the web folder for html/css/js

processed = None   # will hold preprocessed data
clustered = None   # will hold clustered data

# This function runs when frontend button is clicked
@eel.expose
def start_preprocess(file_text, file_name):
    global processed

    try:
        # very simple - assume csv for now
        from io import StringIO
        import pandas as pd
        data = pd.read_csv(StringIO(file_text))

        # use your class
        prep = ShopWisePreprocessor(data)
        prep.handle_missing_values('auto')
        prep.encode_categorical_variables('label')
        prep.scale_numerical_features('standard')
        prep.create_feature_engineering()
        prep.remove_outliers('iqr')

        processed = prep.get_preprocessed_data()

        return "Preprocessing done! Rows: " + str(len(processed))

    except Exception as e:
        return "Error: " + str(e)


# Simple clustering button
@eel.expose
def start_clustering():
    global clustered

    if processed is None:
        return "Error: First preprocess the data"

    try:
        clust = ShopWiseClustering(processed)
        clust.perform_clustering(n_clusters=4)   # fixed 4 clusters for simplicity
        clustered = clust.get_clustered_data()

        return "Clustering done! Check console for details"

    except Exception as e:
        return "Error: " + str(e)


# Show some data
@eel.expose
def show_some_rows():
    if clustered is None:
        return []

    # return first 10 rows as list of dicts
    rows = clustered.head(10).to_dict('records')
    return rows


# Start the webpage
eel.start('index.html', size=(900, 600))