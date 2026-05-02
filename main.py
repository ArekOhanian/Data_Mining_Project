import eel
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import StringIO, BytesIO
import os

# Prevent project.py main block from executing
class DummyMain:
    pass

sys.modules['__main__'] = DummyMain()

from project import ShopWisePreprocessor, ShopWiseClustering, shopWiseAssociation

#groq

from groq import Groq

GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

eel.init('web')

processed_data = None
clustered_data = None
original_data = None
association_results = None
assocition_rules_df = None


#groq integration

@eel.expose
def ask_ai(query: str, cluster_id: int = None):
    global original_data
    try:
        context = ""

        if original_data is not None and 'Cluster' in original_data.columns:
            if cluster_id is not None:
                cluster_df = original_data[original_data['Cluster'] == cluster_id]
                if not cluster_df.empty:
                    # Build a richer context for the AI
                    context = f"""
=== CLUSTER {cluster_id} PROFILE ===
- Customers in this cluster: {len(cluster_df)}
- Average Age: {cluster_df['Age'].mean():.1f} years
- Average Purchase Amount: ${cluster_df['Purchase Amount (USD)'].mean():.2f}
- Average Review Rating: {cluster_df['Review Rating'].mean():.2f} / 5.0
- Most Common Gender: {cluster_df['Gender'].mode()[0] if not cluster_df['Gender'].mode().empty else 'N/A'}
- Min Purchase: ${cluster_df['Purchase Amount (USD)'].min():.2f}
- Max Purchase: ${cluster_df['Purchase Amount (USD)'].max():.2f}
- Age Range: {cluster_df['Age'].min()} – {cluster_df['Age'].max()}
"""
            else:
                n_clusters = original_data['Cluster'].nunique() if 'Cluster' in original_data.columns else 0
                avg_rating_str = f"{original_data['Review Rating'].mean():.2f}" if 'Review Rating' in original_data.columns else 'N/A'
                avg_spend_str = f"${original_data['Purchase Amount (USD)'].mean():.2f}" if 'Purchase Amount (USD)' in original_data.columns else 'N/A'
                age_range_str = f"{original_data['Age'].min()} – {original_data['Age'].max()}" if 'Age' in original_data.columns else 'N/A'
                context = f"""
=== DATASET OVERVIEW ===
- Total Customers: {len(original_data)}
- Number of Clusters: {n_clusters}
- Average Review Rating: {avg_rating_str}
- Average Purchase Amount: {avg_spend_str}
- Age Range: {age_range_str}
"""

        system_prompt = """You are a world-class retail analytics and marketing expert with deep expertise in customer segmentation.
Your role is to analyze customer clusters and provide:
- Clear, actionable business insights
- Specific marketing strategies with channel recommendations
- Personalized product and pricing recommendations
- Risk identification and mitigation strategies

Be concise but thorough. Use bullet points for clarity. Lead with the most impactful insight first.
Always ground your recommendations in the data provided."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context + "\n\nQuestion: " + query}
            ],
            temperature=0.7,
            max_tokens=900
        )
        return response.choices[0].message.content

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"AI Error: {str(e)}"


#preprocess

@eel.expose
def preprocess(file_content, file_name):
    global processed_data, original_data
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


#clustering

@eel.expose
def cluster():
    global clustered_data, processed_data, original_data
    if processed_data is None:
        return "Error: preprocess dataset first"

    try:
        clust = ShopWiseClustering(processed_data)
        X = clust.prepare_clustering_data()
        optimal_k, _, _ = clust.find_optimal_clusters(X, max_clusters=8)
        n_clusters = optimal_k if optimal_k else 4

        clust.perform_clustering(n_clusters=n_clusters)
        clustered_data = clust.get_clustered_data()

        original_data['Cluster'] = clustered_data['Cluster']
        return f"Clustering done. Optimal clusters: {n_clusters}"
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"Error: {str(e)}"

#association
@eel.expose
def run_assocaition(min_support=0.05, min_confidence=0.5):
    global processed_data, association_results, association_rules_df

    if processed_data is None:
        return "Error: preprocess first"
    
    try:
        assoc = shopWiseAssociation(processed_data)

        assoc.prepare_transactions()
        assoc.run_apriori(min_support=min_support)
        assoc.generate_rules(min_confidence=min_confidence)

        association_results = assoc
        association_rules_df = assoc.rules

        return f"Association complete: {len(association_rules_df)} rules found"
    
    except Exception as e:
        return f"Error: {str(e)}"
    
@eel.expose
def get_association_rules():
    global association_rules_df

    if association_rules_df is None or association_rules_df.empty:
        return []
    
    rules = association_rules_df.copy()
    
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]\
        .head(50)\
        .to_dict('records')


@eel.expose
def get_cluster_association(cluster_id, min_support=0.05, min_confidence=0.5):
    global original_data

    if original_data is None or 'Cluster' not in original_data.columns:
        return []

    cluster_df = original_data[original_data['Cluster'] == cluster_id]

    if cluster_df.empty:
        return []

    assoc = shopWiseAssociation(cluster_df)

    assoc.prepare_transactions()
    assoc.run_apriori(min_support=min_support)
    assoc.generate_rules(min_confidence=min_confidence)

    rules = assoc.rules.copy()

    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]\
        .head(20)\
        .to_dict('records')


#data accessors

@eel.expose
def get_results():
    global original_data
    if original_data is None:
        return []
    df = original_data.copy()
    cols = ['Customer ID', 'Age', 'Gender', 'Purchase Amount (USD)', 'Review Rating', 'Cluster']
    available_cols = [col for col in cols if col in df.columns]
    return df.head(100)[available_cols].to_dict('records')


@eel.expose
def get_sorted_results(column):
    global original_data
    if original_data is None:
        return []
    df = original_data.copy()
    if column in df.columns:
        df = df.sort_values(by=column)
    cols = ['Customer ID', 'Age', 'Gender', 'Purchase Amount (USD)', 'Review Rating', 'Cluster']
    available_cols = [col for col in cols if col in df.columns]
    return df.head(100)[available_cols].to_dict('records')


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


@eel.expose
def get_cluster_stats():
    global original_data
    if original_data is None or 'Cluster' not in original_data.columns:
        return {}
    counts = original_data['Cluster'].value_counts().sort_index()
    return counts.to_dict()


@eel.expose
def get_cluster_count():
    global original_data
    if original_data is None or 'Cluster' not in original_data.columns:
        return 0
    return int(original_data['Cluster'].nunique())


@eel.expose
def get_avg_rating():
    global original_data
    if original_data is None or 'Review Rating' not in original_data.columns:
        return 0.0
    return round(float(original_data['Review Rating'].mean()), 2)


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
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64


#start app

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # ← ADD THIS LINE
    print("Working directory:", os.getcwd())
    eel.init('web')
    eel.start('index.html', size=(1300, 850), port=0)