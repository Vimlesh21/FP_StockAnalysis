"""
Streamlit Dashboard for TCS Stock Forecast
Real-time predictions, pipeline monitoring, and analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime, timedelta
import sys
import requests
from typing import Dict, Any, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import logger
from src.api.predict_simple import predict_short, predict_long, get_last_feature_row, FEATURE_COLS
from mlops.pipeline import MLOpsPipeline, PipelineStatus

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="TCS Stock Forecast - MLOps Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success { color: #09ab3b; font-weight: bold; }
    .warning { color: #ff9900; font-weight: bold; }
    .error { color: #ff2b2b; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_predictions(horizon: str) -> Optional[pd.DataFrame]:
    """Load predictions CSV"""
    pred_file = Path(f"predictions/{horizon}_predictions.csv")
    if pred_file.exists():
        return pd.read_csv(pred_file)
    return None


@st.cache_data
def load_evaluation_summary() -> Optional[pd.DataFrame]:
    """Load evaluation metrics"""
    eval_file = Path("reports/evaluation_summary.csv")
    if eval_file.exists():
        return pd.read_csv(eval_file)
    return None


def get_pipeline_runs() -> list:
    """Get list of recent pipeline runs"""
    runs_dir = Path("mlops/runs")
    if not runs_dir.exists():
        return []
    
    runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
    return runs[:10]  # Last 10 runs


def load_pipeline_results(run_id: str) -> Dict[str, Any]:
    """Load pipeline results from a specific run"""
    results_file = Path(f"mlops/runs/{run_id}/results.json")
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return {}


def load_pipeline_metrics(run_id: str) -> list:
    """Load pipeline metrics from a specific run"""
    metrics_file = Path(f"mlops/runs/{run_id}/metrics.json")
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return []


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    st.title("📈 TCS Stock Forecast - MLOps Dashboard")
    st.markdown("Real-time predictions, pipeline monitoring, and performance analytics")
    
    # Sidebar Navigation
    sidebar = st.sidebar
    sidebar.markdown("### 🎯 Navigation")
    
    page = sidebar.radio(
        "Select a page:",
        ["🏠 Home", "🔮 Predictions", "📊 Analytics", "⚙️ Pipeline Control", "📋 Monitoring"],
        label_visibility="collapsed"
    )
    
    if page == "🏠 Home":
        show_home()
    elif page == "🔮 Predictions":
        show_predictions()
    elif page == "📊 Analytics":
        show_analytics()
    elif page == "⚙️ Pipeline Control":
        show_pipeline_control()
    elif page == "📋 Monitoring":
        show_monitoring()


# ============================================================================
# PAGE: HOME
# ============================================================================

def show_home():
    st.markdown("## Welcome to the TCS Stock Forecast Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    # Get latest predictions
    short_pred = load_predictions("short")
    long_pred = load_predictions("long")
    
    with col1:
        st.markdown("### 📌 Short-term Forecast")
        if short_pred is not None and len(short_pred) > 0:
            latest_short = short_pred.iloc[-1]
            st.metric(
                "Next Day Return",
                f"{latest_short['pred']:.4f}",
                f"{latest_short['pred']*100:.2f}%"
            )
        else:
            st.warning("No short-term predictions available")
    
    with col2:
        st.markdown("### 📌 Long-term Forecast")
        if long_pred is not None and len(long_pred) > 0:
            latest_long = long_pred.iloc[-1]
            st.metric(
                "63-Day Return",
                f"{latest_long['pred']:.4f}",
                f"{latest_long['pred']*100:.2f}%"
            )
        else:
            st.warning("No long-term predictions available")
    
    with col3:
        st.markdown("### 📌 System Status")
        eval_summary = load_evaluation_summary()
        if eval_summary is not None and len(eval_summary) > 0:
            short_r2 = eval_summary[eval_summary['horizon'] == 'short']['model_r2'].values
            if len(short_r2) > 0:
                st.metric("Model R² Score", f"{short_r2[0]:.4f}")
            else:
                st.info("No evaluation metrics")
        else:
            st.info("No evaluation metrics")
    
    # Quick Info
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📂 Quick Links")
        st.markdown("""
        - **[API Endpoints](http://localhost:8000/docs)** - FastAPI documentation
        - **[Logs](logs/tcs-stock.log)** - Pipeline logs
        - **[Reports](reports/)** - Evaluation reports
        """)
    
    with col2:
        st.markdown("### 📚 What's New")
        st.markdown("""
        - ✅ Dual horizon forecasting (short & long-term)
        - ✅ Real-time predictions via FastAPI
        - ✅ MLOps pipeline with monitoring
        - ✅ Comprehensive analytics dashboard
        - 🔄 Coming: Model retraining automation
        """)


# ============================================================================
# PAGE: PREDICTIONS
# ============================================================================

def show_predictions():
    st.markdown("## 🔮 Live Predictions")
    
    tab1, tab2, tab3 = st.tabs(["Short-term (1-day)", "Long-term (63-day)", "Comparison"])
    
    # SHORT-TERM TAB
    with tab1:
        st.markdown("### Next Business Day Prediction")
        
        try:
            result = predict_short()
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Predicted Return",
                    f"{result['pred_return']:.6f}",
                    f"{result['pct_return']:.3f}%",
                    delta_color="inverse"
                )
            with col2:
                st.metric(
                    "Current Price",
                    f"₹{result['today_rate']:.2f}" if result['today_rate'] else "N/A"
                )
            with col3:
                est_price = result['est_next_close']
                if est_price:
                    price_change = est_price - result['today_rate']
                    st.metric(
                        "Est. Next Close",
                        f"₹{est_price:.2f}",
                        f"₹{price_change:.2f}"
                    )
            with col4:
                feature_date = result.get('feature_date', 'N/A')
                predict_date = result.get('predict_date_from_feature', 'N/A')
                st.metric("Feature Date", feature_date)
            
            # Prediction Details
            st.markdown("#### 📋 Prediction Details")
            details_df = pd.DataFrame({
                'Metric': [
                    'Prediction Date',
                    'System Date',
                    'Model Horizon',
                    'Pred Return (frac)',
                    'Pred Return (%)',
                ],
                'Value': [
                    result.get('predict_date_from_feature', 'N/A'),
                    result.get('system_date', 'N/A'),
                    result.get('horizon', 'N/A'),
                    f"{result['pred_return']:.6f}",
                    f"{result['pct_return']:.3f}%",
                ]
            })
            st.dataframe(details_df, use_container_width=True)
            
            # Note
            if result.get('note'):
                st.warning(f"⚠️ {result['note']}")
        
        except Exception as e:
            st.error(f"Error fetching short-term prediction: {e}")
    
    # LONG-TERM TAB
    with tab2:
        st.markdown("### 63-Day Forecast")
        
        try:
            result = predict_long()
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Predicted 63D Return",
                    f"{result['pred_return_63']:.6f}",
                    f"{result['pct_return_63']:.3f}%",
                    delta_color="inverse"
                )
            with col2:
                st.metric(
                    "Current Price",
                    f"₹{result['today_rate']:.2f}" if result['today_rate'] else "N/A"
                )
            with col3:
                est_price = result['est_close_63bd']
                if est_price:
                    price_change = est_price - result['today_rate']
                    st.metric(
                        "Est. 63D Close",
                        f"₹{est_price:.2f}",
                        f"₹{price_change:.2f}"
                    )
            with col4:
                feature_date = result.get('feature_date', 'N/A')
                st.metric("Feature Date", feature_date)
            
            # Prediction Details
            st.markdown("#### 📋 Prediction Details")
            details_df = pd.DataFrame({
                'Metric': [
                    'Prediction Date (from feature)',
                    'Prediction Date (from system)',
                    'System Date',
                    'Pred Return (frac)',
                    'Pred Return (%)',
                ],
                'Value': [
                    result.get('predict_date_from_feature_63bd', 'N/A'),
                    result.get('predict_date_from_system_63bd', 'N/A'),
                    result.get('system_date', 'N/A'),
                    f"{result['pred_return_63']:.6f}",
                    f"{result['pct_return_63']:.3f}%",
                ]
            })
            st.dataframe(details_df, use_container_width=True)
            
            # Note
            if result.get('note'):
                st.warning(f"⚠️ {result['note']}")
        
        except Exception as e:
            st.error(f"Error fetching long-term prediction: {e}")
    
    # COMPARISON TAB
    with tab3:
        st.markdown("### Forecast Comparison")
        
        try:
            short_result = predict_short()
            long_result = predict_long()
            
            comparison_df = pd.DataFrame({
                'Aspect': [
                    'Horizon',
                    'Predicted Return (frac)',
                    'Predicted Return (%)',
                    'Current Price',
                    'Estimated Future Price',
                    'Price Change',
                ],
                'Short-term (1D)': [
                    '1 business day',
                    f"{short_result['pred_return']:.6f}",
                    f"{short_result['pct_return']:.3f}%",
                    f"₹{short_result['today_rate']:.2f}" if short_result['today_rate'] else 'N/A',
                    f"₹{short_result['est_next_close']:.2f}" if short_result['est_next_close'] else 'N/A',
                    f"₹{short_result['est_next_close'] - short_result['today_rate']:.2f}" if short_result['est_next_close'] and short_result['today_rate'] else 'N/A',
                ],
                'Long-term (63D)': [
                    '~63 business days',
                    f"{long_result['pred_return_63']:.6f}",
                    f"{long_result['pct_return_63']:.3f}%",
                    f"₹{long_result['today_rate']:.2f}" if long_result['today_rate'] else 'N/A',
                    f"₹{long_result['est_close_63bd']:.2f}" if long_result['est_close_63bd'] else 'N/A',
                    f"₹{long_result['est_close_63bd'] - long_result['today_rate']:.2f}" if long_result['est_close_63bd'] and long_result['today_rate'] else 'N/A',
                ]
            })
            
            st.dataframe(comparison_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error creating comparison: {e}")


# ============================================================================
# PAGE: ANALYTICS
# ============================================================================

def show_analytics():
    st.markdown("## 📊 Model Analytics & Performance")
    
    tab1, tab2, tab3 = st.tabs(["Model Performance", "Predictions History", "Feature Analysis"])
    
    with tab1:
        st.markdown("### 📈 Model Evaluation Metrics")
        
        eval_summary = load_evaluation_summary()
        if eval_summary is not None:
            # Display metrics table
            st.dataframe(eval_summary, use_container_width=True)
            
            # Visualize metrics
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[
                    go.Bar(name='MAE', x=eval_summary['horizon'], y=eval_summary['model_mae']),
                ])
                fig.update_layout(title="Mean Absolute Error (Lower is Better)", 
                                xaxis_title="Horizon", yaxis_title="MAE")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(data=[
                    go.Bar(name='RMSE', x=eval_summary['horizon'], y=eval_summary['model_rmse']),
                ])
                fig.update_layout(title="Root Mean Squared Error (Lower is Better)",
                                xaxis_title="Horizon", yaxis_title="RMSE")
                st.plotly_chart(fig, use_container_width=True)
            
            # R² Score
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(data=[
                    go.Bar(name='Model R²', x=eval_summary['horizon'], y=eval_summary['model_r2'], marker_color='lightgreen'),
                    go.Bar(name='Baseline R²', x=eval_summary['horizon'], y=eval_summary['baseline_r2'], marker_color='lightcoral'),
                ])
                fig.update_layout(title="R² Score Comparison (Higher is Better)",
                                xaxis_title="Horizon", yaxis_title="R² Score", barmode='group')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No evaluation summary available")
    
    with tab2:
        st.markdown("### 📉 Predictions History")
        
        horizon = st.selectbox("Select Horizon", ["short", "long"])
        pred_df = load_predictions(horizon)
        
        if pred_df is not None:
            st.write(f"Total predictions: {len(pred_df)}")
            
            # Display data
            st.dataframe(pred_df.tail(20), use_container_width=True)
            
            # Plot
            if len(pred_df) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(pred_df.index if pred_df.index.name else range(len(pred_df))),
                    y=pred_df['actual'] if 'actual' in pred_df.columns else pred_df.iloc[:, 0],
                    mode='lines',
                    name='Actual'
                ))
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(pred_df.index if pred_df.index.name else range(len(pred_df))),
                    y=pred_df['pred'] if 'pred' in pred_df.columns else pred_df.iloc[:, 1],
                    mode='lines',
                    name='Predicted'
                ))
                fig.update_layout(
                    title=f"{horizon.title()}-term: Actual vs Predicted",
                    xaxis_title="Date",
                    yaxis_title="Return",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No {horizon}-term predictions available")
    
    with tab3:
        st.markdown("### 🔍 Feature Information")
        
        st.write("**Feature Columns Used in Model:**")
        features_df = pd.DataFrame({
            'Feature': FEATURE_COLS,
            'Type': [
                'Lag Return 1D', 'Lag Return 2D', 'Lag Return 3D', 'Lag Return 4D', 'Lag Return 5D',
                'SMA 7D', 'SMA 21D', 'Momentum 7D', 'Volatility 14D', 'Volume'
            ]
        })
        st.dataframe(features_df, use_container_width=True)


# ============================================================================
# PAGE: PIPELINE CONTROL
# ============================================================================

def show_pipeline_control():
    st.markdown("## ⚙️ MLOps Pipeline Control")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🚀 Pipeline Execution")
        st.markdown("Execute the complete MLOps pipeline for data refresh and model retraining.")
        
        ticker = st.text_input("Ticker Symbol", value="TCS.NS")
        skip_training = st.checkbox("Skip Training Stage", value=False)
        
        if st.button("🟢 Run Pipeline", use_container_width=True):
            with st.spinner("Running pipeline..."):
                try:
                    pipeline = MLOpsPipeline(ticker=ticker)
                    success, results = pipeline.run_full_pipeline(skip_training=skip_training)
                    
                    if success:
                        st.success("✅ Pipeline completed successfully!")
                        
                        # Display results
                        st.json(results)
                    else:
                        st.error("❌ Pipeline failed!")
                        st.json(results)
                
                except Exception as e:
                    st.error(f"Pipeline execution error: {e}")
                    logger.exception("Pipeline execution failed")
    
    with col2:
        st.markdown("### 📊 Recent Runs")
        runs = get_pipeline_runs()
        
        if runs:
            run_ids = [r.name for r in runs]
            selected_run = st.selectbox("Select Run", run_ids)
            
            if selected_run:
                results = load_pipeline_results(selected_run)
                metrics = load_pipeline_metrics(selected_run)
                
                st.markdown(f"#### Run: {selected_run}")
                
                if results:
                    st.metric("Status", results.get('status', 'N/A'))
                    st.metric("Duration (s)", f"{results.get('total_duration_seconds', 0):.2f}")
                
                if metrics:
                    st.markdown("**Stages:**")
                    for metric in metrics:
                        status_color = "🟢" if metric['status'] == 'success' else "🔴"
                        st.write(f"{status_color} {metric['stage']}: {metric['duration_seconds']:.2f}s ({metric['rows_processed']} rows)")
        else:
            st.info("No pipeline runs found")


# ============================================================================
# PAGE: MONITORING
# ============================================================================

def show_monitoring():
    st.markdown("## 📋 System Monitoring & Logs")
    
    tab1, tab2, tab3 = st.tabs(["Pipeline Runs", "System Health", "Logs"])
    
    with tab1:
        st.markdown("### 📊 Pipeline Run History")
        
        runs = get_pipeline_runs()
        
        if runs:
            # Create summary dataframe
            run_data = []
            for run_dir in runs:
                run_id = run_dir.name
                results = load_pipeline_results(run_id)
                metrics = load_pipeline_metrics(run_id)
                
                run_data.append({
                    'Run ID': run_id,
                    'Status': results.get('status', 'unknown'),
                    'Duration (s)': f"{results.get('total_duration_seconds', 0):.2f}",
                    'Stages': len(metrics),
                    'Timestamp': results.get('timestamp', 'N/A'),
                })
            
            runs_df = pd.DataFrame(run_data)
            st.dataframe(runs_df, use_container_width=True)
        else:
            st.info("No pipeline runs available")
    
    with tab2:
        st.markdown("### 🏥 System Health")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Database", "✅ Connected", help="PostgreSQL database status")
        
        with col2:
            models_exist = Path("models/short_model.pkl").exists() and Path("models/long_model.pkl").exists()
            status = "✅ Ready" if models_exist else "⚠️ Missing"
            st.metric("Models", status)
        
        with col3:
            features_file = Path("reports/evaluation_summary.csv")
            status = "✅ Available" if features_file.exists() else "⚠️ Missing"
            st.metric("Evaluation", status)
        
        st.markdown("---")
        st.markdown("### 📁 Directory Structure")
        
        dirs = {
            'Data': 'predictions/',
            'Models': 'models/',
            'Reports': 'reports/',
            'Logs': 'logs/',
            'Pipeline': 'mlops/runs/',
        }
        
        dir_status = []
        for name, path in dirs.items():
            exists = Path(path).exists()
            status = "✅" if exists else "⚠️"
            dir_status.append({
                'Directory': name,
                'Path': path,
                'Status': status,
            })
        
        st.dataframe(pd.DataFrame(dir_status), use_container_width=True)
    
    with tab3:
        st.markdown("### 📝 Application Logs")
        
        log_file = Path("logs/tcs-stock.log")
        if log_file.exists():
            with open(log_file) as f:
                logs = f.readlines()
            
            # Show last N lines
            n_lines = st.slider("Show last N lines", 10, 500, 100)
            
            log_text = ''.join(logs[-n_lines:])
            st.text_area("Logs", log_text, height=400, disabled=True)
            
            # Download button
            st.download_button(
                label="📥 Download Full Log",
                data=log_text,
                file_name=f"tcs-stock-{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                mime="text/plain"
            )
        else:
            st.info("Log file not found")


if __name__ == "__main__":
    main()
