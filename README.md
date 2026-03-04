# 🧠 NeuroCore - Perceptron Analytics Dashboard

A futuristic, interactive gaming-themed dashboard for Perceptron Neural Network analysis and visualization. Built with Python, Streamlit, and Plotly.

## ✨ Features

- **📂 Multi-format Data Support**: Load CSV and ARFF files directly.
- **🧹 Auto Data Cleaning**: Automatic missing value detection and removal.
- **🔗 Smart Label Encoding**: Automatic categorical data (text) mapping to numeric labels.
- **⚖️ Precise Parameter Tuning**: Free decimal entry for initial weights ($W$), threshold ($\theta$), learning rate ($\alpha$), and gain ($a$).
- **🔥 Sequential Learning (Slide 16 & 18)**: Implements "Immediate Update" logic—weights are updated after _every single row_ for maximum accuracy.
- **📋 Detailed Training Log**: Step-by-step calculation table following **Slide 19** standard (n, Epoch, inputs, desired, predicted, error, weight deltas, updated weights).
- **📈 Error Convergence Chart**: Real-time visualization of Total Absolute Error per Epoch.
- **🎮 Manual Prediction Box**: Live testing with dropdowns for categorical features and confidence strength indicator.
- **📥 Data Export**: Download the full training log as a CSV for reporting.

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9 or higher

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd perceptron_dashboard

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Dashboard

```bash
streamlit run app.py
```

## 🛠️ Tech Stack

- **Framework**: [Streamlit](https://streamlit.io/)
- **Visuals**: [Plotly](https://plotly.com/)
- **Logic**: [NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/)
- **Animations**: [Streamlit Lottie](https://github.com/andfanilo/streamlit-lottie)

---

_Created with ❤️ for Perceptron Neural Network Analysis._
