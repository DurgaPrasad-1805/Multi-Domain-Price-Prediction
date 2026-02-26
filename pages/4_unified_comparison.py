import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Unified Model Comparison")

# --------------------------------------------------
# MODEL RESULTS (Grouped Order)
# --------------------------------------------------

results_data = [
    ["House Model (Dataset A - California)", 0.8169],
    ["House Model (Dataset B - Bengaluru)", 0.6664],
    ["Car Model (Dataset A - India)", 0.8836],
    ["Car Model (Dataset B - Germany)", 0.7544],
    ["Gold Model (Time-Series Linear)", 0.9607],
]

results_df = pd.DataFrame(results_data, columns=["Model", "R² Score"])

# --------------------------------------------------
# DISPLAY TABLE
# --------------------------------------------------

st.subheader("Model Performance Table")
st.dataframe(results_df, use_container_width=True)

# --------------------------------------------------
# BAR CHART VISUALIZATION
# --------------------------------------------------

st.subheader("Performance Comparison (R² Score)")

plt.figure(figsize=(10,6))
plt.barh(results_df["Model"], results_df["R² Score"])
plt.xlabel("R² Score")
plt.gca().invert_yaxis()
st.pyplot(plt.gcf())
plt.clf()

# --------------------------------------------------
# BEST MODEL
# --------------------------------------------------

best_index = results_df["R² Score"].idxmax()
best_model = results_df.loc[best_index, "Model"]
best_score = results_df.loc[best_index, "R² Score"]

st.success(f"🏆 Best Performing Model: {best_model} (R² = {best_score})")