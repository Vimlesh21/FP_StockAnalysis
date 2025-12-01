"""
Run:
    python tools/create_final_report_nb.py

This script creates a clean notebook with:
- Overview section
- Data summary
- Model results (short + long)
- Predictions preview
- Plots (if figures exist)
"""

import os
from pathlib import Path
import json
import pandas as pd
import nbformat as nbf


# ------------------------------------------------------------
# 1. Helper: safe file loading
# ------------------------------------------------------------

def safe_read_csv(path):
    """Read CSV only if file exists."""
    if Path(path).exists():
        return pd.read_csv(path)
    return None


def safe_read_json(path):
    """Read JSON only if file exists."""
    if Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


# ------------------------------------------------------------
# 2. Create notebook
# ------------------------------------------------------------

def create_notebook():
    """
    Build a simple, clean Jupyter notebook file with markdown + code.

    Output:
        reports/final_report.ipynb
    """
    # Output folder
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)

    nb = nbf.v4.new_notebook()
    cells = []

    # --------------------------------------------------------
    # CELL 1 — Title
    # --------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
        "# 📘 Final Forecasting Report\n"
        "This notebook summarizes the entire forecasting pipeline:\n"
        "- Data preparation\n"
        "- Feature engineering\n"
        "- Short-term model results\n"
        "- Long-term model results\n"
        "- Predictions preview\n"
        "- Evaluation summary\n"
        "- Plots\n\n"
        "*Generated automatically using `create_final_report_nb.py`.*"
    ))

    # --------------------------------------------------------
    # CELL 2 — Imports
    # --------------------------------------------------------
    cells.append(nbf.v4.new_code_cell(
        "# Basic imports\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from pathlib import Path\n\n"
        "pd.set_option('display.max_rows', 10)\n"
        "pd.set_option('display.max_columns', None)"
    ))

    # --------------------------------------------------------
    # CELL 3 — Load predictions
    # --------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
        "## 📂 Load Predictions\n"
        "Short-term and long-term predictions saved earlier."
    ))

    cells.append(nbf.v4.new_code_cell(
        "short_pred_path = Path('predictions/short_predictions.csv')\n"
        "long_pred_path  = Path('predictions/long_predictions.csv')\n\n"
        "short_df = pd.read_csv(short_pred_path) if short_pred_path.exists() else None\n"
        "long_df  = pd.read_csv(long_pred_path)  if long_pred_path.exists()  else None\n\n"
        "short_df, long_df"
    ))

    # --------------------------------------------------------
    # CELL 4 — Load evaluation summary
    # --------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
        "## 📊 Evaluation Summary\n"
        "Aggregated metrics for both models."
    ))

    cells.append(nbf.v4.new_code_cell(
        "summary_path = Path('reports/evaluation_summary.csv')\n"
        "summary_df = pd.read_csv(summary_path) if summary_path.exists() else None\n"
        "summary_df"
    ))

    # --------------------------------------------------------
    # CELL 5 — Simple plots
    # --------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
        "## 📈 Prediction vs Actual (Short-term)\n"
        "Shows first 200 points if available."
    ))

    cells.append(nbf.v4.new_code_cell(
        "if short_df is not None:\n"
        "    short_df2 = short_df.head(200)\n"
        "    plt.figure(figsize=(12,4))\n"
        "    plt.plot(short_df2['actual'], label='Actual')\n"
        "    plt.plot(short_df2['pred'], label='Predicted')\n"
        "    plt.title('Short-term: Predicted vs Actual')\n"
        "    plt.legend()\n"
        "    plt.show()\n"
        "else:\n"
        "    print('Short-term predictions not found.')"
    ))

    # --------------------------------------------------------
    # CELL 6 — Long-term plot
    # --------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
        "## 📈 Prediction vs Actual (Long-term)\n"
        "Shows first 200 points if available."
    ))

    cells.append(nbf.v4.new_code_cell(
        "if long_df is not None:\n"
        "    long_df2 = long_df.head(200)\n"
        "    plt.figure(figsize=(12,4))\n"
        "    plt.plot(long_df2['actual'], label='Actual')\n"
        "    plt.plot(long_df2['pred'], label='Predicted')\n"
        "    plt.title('Long-term: Predicted vs Actual')\n"
        "    plt.legend()\n"
        "    plt.show()\n"
        "else:\n"
        "    print('Long-term predictions not found.')"
    ))

    # --------------------------------------------------------
    # CELL 7 — Final comments
    # --------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
        "## ✅ Summary\n"
        "- Models trained successfully.\n"
        "- Predictions generated.\n"
        "- Evaluation summary loaded.\n"
        "- Plots visualized.\n\n"
        "You can now use this notebook to add explanations, insights, visuals, and final storytelling for the project."
    ))

    # Add all cells
    nb["cells"] = cells

    # Save notebook
    out_path = out_dir / "final_report.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"\n📘 Notebook created at: {out_path}\n")



# ------------------------------------------------------------
# 3. Run when file is executed
# ------------------------------------------------------------

if __name__ == "__main__":
    create_notebook()
