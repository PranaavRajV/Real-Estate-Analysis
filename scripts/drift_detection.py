import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import logging
import os

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_drift_analysis():
    logger.info("Starting Drift Analysis...")
    
    # reference data (Training)
    ref_df = pd.read_csv("data/processed/clean_data.csv")
    
    # current data (Simulating new drifts - e.g. prices suddenly increase)
    cur_df = ref_df.copy()
    cur_df['sale_price'] = cur_df['sale_price'] * 1.2 # 20% market jump
    cur_df['sqft'] = cur_df['sqft'] + 100 # Change in architecture trends
    
    # Create Report
    drift_report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])
    
    drift_report.run(reference_data=ref_df, current_data=cur_df)
    
    # Save Report
    report_path = "reports/figures/data_drift_report.html"
    drift_report.save_html(report_path)
    logger.info(f"Drift report generated at {report_path}")

if __name__ == "__main__":
    run_drift_analysis()
