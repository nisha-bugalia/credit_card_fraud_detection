import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import time
import os

# Set Streamlit Page Configuration
st.set_page_config(page_title="Real-Time Fraud Detection", page_icon="🚨", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Premium Design ---
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .metric-card {
        background: #1E2127;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin: 10px;
        border-left: 4px solid #4CAF50;
    }
    .metric-card.high-risk {
        border-left-color: #FF5252;
    }
    .metric-card.medium-risk {
        border-left-color: #FFC107;
    }
    .metric-title {
        font-size: 14px;
        color: #A0AAB4;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #FFFFFF;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        background: -webkit-linear-gradient(45deg, #FF5252, #FF9800);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .alert-row {
        background-color: rgba(255, 82, 82, 0.1);
        border: 1px solid #FF5252;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

DB_PATH = "data/fraud_db.sqlite"

def load_live_data(limit=1000):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f"SELECT * FROM alerts ORDER BY Timestamp DESC LIMIT {limit}", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error reading from DB: {e}")
        return pd.DataFrame()

def main():
    st.markdown('<p class="main-header">Real-Time Fraud Monitor</p>', unsafe_allow_html=True)
    st.markdown("Streaming data ingestion via **Apache Kafka** and real-time inference via **PySpark**.")
    
    st.sidebar.header("Controls")
    auto_refresh = st.sidebar.checkbox("Auto-Refresh (Live)", value=True)
    
    st.sidebar.markdown("---")
    menu = st.sidebar.radio("Navigation", ["Live Inference Data", "Training Offline Metrics"])

    if menu == "Live Inference Data":
        df = load_live_data(limit=5000)
        
        if df.empty:
            st.warning("No live data found. Make sure Kafka Producer and PySpark Streaming pipeline are running!")
        else:
            total_processed = len(df)
            high_risk = len(df[df['RiskLevel'] == 'HIGH RISK'])
            medium_risk = len(df[df['RiskLevel'] == 'MEDIUM RISK'])
            
            # Metrics Row
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Transactions Processed (Latest)</div><div class="metric-value">{total_processed}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card high-risk"><div class="metric-title">High Risk Alerts</div><div class="metric-value">{high_risk}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card medium-risk"><div class="metric-title">Medium Risk Alerts</div><div class="metric-value">{medium_risk}</div></div>', unsafe_allow_html=True)

            st.markdown("### 🚨 Latest High Risk Transactions")
            high_risk_df = df[df['RiskLevel'] == 'HIGH RISK'].head(10)
            if high_risk_df.empty:
                st.success("No High Risk transactions recently!")
            else:
                st.dataframe(high_risk_df[['Timestamp', 'Amount', 'Probability', 'RiskLevel', 'ActualClass']], use_container_width=True)
            
            st.markdown("### Transaction Probability Stream")
            # Simple line chart
            plot_df = df.head(100).sort_values("Timestamp")
            if len(plot_df) > 0:
                fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0E1117')
                ax.set_facecolor('#0E1117')
                sns.lineplot(x=range(len(plot_df)), y='Probability', data=plot_df, color='#FF9800', ax=ax)
                ax.axhline(0.8, color='red', linestyle='--', label='High Risk Threshold')
                ax.axhline(0.5, color='yellow', linestyle='--', label='Medium Risk Threshold')
                ax.set_title("Fraud Probability (Last 100 Txs)", color='white')
                ax.legend()
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('grey')
                st.pyplot(fig)
                
            st.markdown("### Raw Stream Feed")
            st.dataframe(df.head(20), use_container_width=True)

        if auto_refresh:
            time.sleep(2)
            st.rerun()

    elif menu == "Training Offline Metrics":
        st.subheader("Offline Model Training Results")
        
        metrics_path = "output/metrics.json"
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
                
            metrics_table = []
            for m in metrics_data:
                metrics_table.append({
                    "Model": m["model_name"],
                    "Accuracy": f"{m['accuracy']:.4f}",
                    "F1 Score": f"{m['f1_score']:.4f}",
                    "Precision": f"{m['precision']:.4f}",
                    "Recall": f"{m['recall']:.4f}",
                    "ROC AUC": f"{m['roc_auc']:.4f}"
                })
            st.table(metrics_table)
        else:
            st.warning("No offline metrics found. Run offline pipeline training first.")

if __name__ == "__main__":
    main()
