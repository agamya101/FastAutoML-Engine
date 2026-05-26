import streamlit as st
import pandas as pd
import numpy as np
import time
import itertools
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

# --- UI Setup ---
st.set_page_config(page_title="FastAutoML Engine", layout="wide")
st.title("⚡ FastAutoML: Successive Quartering Feature Selector")

# --- Introduction & Methodology Dropdown ---
with st.expander("ℹ️ Click here to see how it works & why it's faster"):
    st.markdown("""
    ### What is this project?
    When building machine learning models, finding the right combination of data features is critical. However, testing every possible combination (**Brute-Force Grid Search**) is incredibly slow because its time complexity scales exponentially at $O(2^N)$. If you have 10 features, that's 1,023 combinations to test!
    
    ### Why is this engine faster?
    This application uses a custom **Successive Quartering Selection** algorithm designed to dramatically cut down search times:
    
    1. **Data Budgeting:** Instead of testing all feature combinations on 100% of your data, the engine starts by testing them on a tiny sliver of the dataset (e.g., 15%). 
    2. **Aggressive Pruning (Tournament Style):** At the end of each round, the engine evaluates the models using a custom performance score and **instantly eliminates the bottom 75%** of feature combinations.
    3. **Resource Scaling:** For the next round, only the surviving top 25% of combinations move forward, and the engine doubles the data budget to test them more rigorously. 
    
    ### The Mathematical Guard (MAE + Variance)
    Aggressive elimination risks throwing away a great feature group early just because it got "unlucky" on a small slice of data. To fix this, our custom formula combines **Mean Absolute Error (MAE)** with **Statistical Variance**:
    """)
    
    st.latex(r"\text{Custom Score} = \text{Mean MAE} + \left(1.96 \times \sqrt{\text{Variance}}\right)")
    
    st.markdown("""
    By adding a penalty for high variance across cross-validation folds, the engine ensures that only stable, truly high-performing feature groups survive the early cuts.
    """)

# --- File Uploader ---
st.write("---")
uploaded_file = st.file_uploader("Upload your CSV file to begin", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head(3))
    
    # Let user pick the target variable
    all_columns = list(df.columns)
    target = st.selectbox("Select the column to predict (Target Variable)", all_columns)
    
    if st.button("Run Fast Optimization"):
        start_time = time.time()
        
        # --- Preprocessing ---
        df = df.dropna(subset=[target])
        y = df[target]
        X = df.drop(columns=[target])
        
        # Handle missing data and non-numeric columns simply
        X = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))
        features_list = list(X.columns)
        
        if len(features_list) > 10:
            st.warning("⚠️ High number of features! Limiting to the first 10 for demonstration speed.")
            features_list = features_list[:10]
            X = X[features_list]

        # --- Generate Power Set of Features ---
        feature_combinations = []
        for r in range(1, len(features_list) + 1):
            for combo in itertools.combinations(features_list, r):
                feature_combinations.append(list(combo))
                
        st.write(f"Testing a total of **{len(feature_combinations)}** unique feature combinations...")
        
        # --- Successive Quartering Engine ---
        current_pool = feature_combinations
        
        # DYNAMIC BUDGET: If data is small (< 100 rows), skip slicing and use 100% data from the start
        if len(X) < 100:
            data_budget_pct = 1.0
            st.info(f"ℹ️ Small dataset detected ({len(X)} rows). Running optimization on full data to ensure stable Cross-Validation.")
        else:
            data_budget_pct = 0.15  # Start with 15% for massive datasets
            
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        round_num = 1
        
        # Keep looping until 1 best combination remains or we hit 100% data budget
        while len(current_pool) > 1 and data_budget_pct <= 1.0:
            current_rows = int(len(X) * data_budget_pct)
            
            # Absolute hard-floor safety check for cross-validation math
            if current_rows < 5:
                current_rows = len(X)
                
            X_slice = X.iloc[:current_rows]
            y_slice = y.iloc[:current_rows]
            
            round_results = []
            st.write(f"🔄 **Round {round_num}:** Testing {len(current_pool)} combinations on {current_rows} rows of data...")
            
            for combo in current_pool:
                mae_scores = []
                X_combo = X_slice[combo]
                
                # 5-Part Split Testing
                for train_idx, val_idx in kf.split(X_combo):
                    model = DecisionTreeRegressor(random_state=42)
                    model.fit(X_combo.iloc[train_idx], y_slice.iloc[train_idx])
                    preds = model.predict(X_combo.iloc[val_idx])
                    mae_scores.append(mean_absolute_error(y_slice.iloc[val_idx], preds))
                
                # Mathematical Guard: Combine Mean MAE and Variance
                avg_mae = np.mean(mae_scores)
                variance_mae = np.var(mae_scores)
                custom_score = avg_mae + (1.96 * np.sqrt(variance_mae))
                
                round_results.append({'features': combo, 'score': custom_score, 'mae': avg_mae})
            
            # Sort and keep top 25% (eliminate 75%)
            round_results = sorted(round_results, key=lambda x: x['score'])
            keep_count = max(1, len(current_pool) // 4)
            current_pool = [r['features'] for r in round_results[:keep_count]]
            
            # If we used 100% data budget but still have multiple combos, pick the absolute best one
            if data_budget_pct == 1.0 and len(current_pool) > 1:
                current_pool = [round_results[0]['features']]
                break
                
            # Scale up data size for large datasets
            data_budget_pct = min(1.0, data_budget_pct * 2)
            round_num += 1
        
        # --- Results Output ---
        end_time = time.time()
        your_engine_time = end_time - start_time
        best_features = current_pool[0]
        
        st.success("🎉 Optimization Complete!")
        
        # --- BENCHMARK: Run Standard Full Search (Brute-Force) ---
        st.write("---")
        st.write("### ⏱️ Running Industry Benchmark (Full Brute-Force Grid Search)...")
        
        benchmark_start = time.time()
        best_benchmark_score = float('inf')
        best_benchmark_features = None
        
        for combo in feature_combinations:
            mae_scores = []
            X_combo = X[combo]
            for train_idx, val_idx in kf.split(X_combo):
                model = DecisionTreeRegressor(random_state=42)
                model.fit(X_combo.iloc[train_idx], y.iloc[train_idx])
                preds = model.predict(X_combo.iloc[val_idx])
                mae_scores.append(mean_absolute_error(y.iloc[val_idx], preds))
            
            avg_mae = np.mean(mae_scores)
            if avg_mae < best_benchmark_score:
                best_benchmark_score = avg_mae
                best_benchmark_features = combo
                
        benchmark_time = time.time() - benchmark_start
        
        # --- Display Comparison Results ---
        st.write("### 📊 Performance Comparison Report")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="⚡ Your Custom Engine Time", value=f"{your_engine_time:.4f} sec")
            st.write("**Your Chosen Features:**")
            st.code(f"{best_features}")
            
        with col2:
            st.metric(label="🐢 Brute-Force Grid Search Time", value=f"{benchmark_time:.4f} sec", delta=f"{((benchmark_time - your_engine_time)/benchmark_time)*100:.1f}% Slower", delta_color="inverse")
            st.write("**True Optimal Features:**")
            st.code(f"{best_benchmark_features}")
            
        # Success check
        if set(best_features) == set(best_benchmark_features):
            st.balloons()
            st.success("✅ Perfect Match! Your aggressive elimination engine found the mathematically correct features in a fraction of the time!")
        else:
            st.warning("⚠️ Approximation Made: Your engine found a slightly different sub-optimal feature group, but saved significant compute time.")