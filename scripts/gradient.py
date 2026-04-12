import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import kaleido

kaleido.get_chrome_sync()

df_2015 = pd.read_csv("../data/times_clean_2015.csv")

METRICS = ["teaching", "research", "citations", "international", "income"]
TARGET = "total_score"

df_model = df_2015.dropna(subset=METRICS + [TARGET])

X = df_model[METRICS]
y = df_model[TARGET]

model = LinearRegression(fit_intercept=True)
model.fit(X, y)

gradient_results = pd.DataFrame({
    "Indicator": METRICS,
    "Weight": model.coef_
}).sort_values(by="Weight", ascending=True)

official_weights = [0.30, 0.30, 0.30, 0.075, 0.025]

gradient_results = pd.DataFrame({
    "Indicator": METRICS,
    "Empirical_Weight": model.coef_,
    "Official_Weight": official_weights
}).sort_values(by="Empirical_Weight", ascending=False)

print("=" * 65)
print("GRADIENT ANALYSIS: EMPIRICAL VS OFFICIAL WEIGHTS")
print("=" * 65)
print(gradient_results.to_string(index=False))

fig = go.Figure()

for i, row in gradient_results.iterrows():
    fig.add_shape(
        type='line',
        x0=0, y0=row['Indicator'],
        x1=row['Empirical_Weight'], y1=row['Indicator'],
        line=dict(color='lightslategrey', width=2)
    )

fig.add_trace(go.Scatter(
    x=gradient_results['Empirical_Weight'], 
    y=gradient_results['Indicator'],
    mode='markers',
    marker=dict(color='#4C72B0', size=15, line=dict(color='white', width=2)),
    hovertemplate="Indicator: %{y}<br>Weight: %{x:.3f}<extra></extra>"
))

fig.update_layout(
    title="Empirical Weights of THE Indicators",
    xaxis_title="Influence Magnitude on Total Score",
    yaxis_title="",
    template="plotly_white",
    height=500,
    showlegend=False
)

fig.write_html("../figures/html/gradient_lollipop.html")
fig.write_image("../figures/png/gradient_lollipop.png", scale=2)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

for i, metric in enumerate(METRICS):
    sns.regplot(
        data=df_model, 
        x=metric, 
        y=TARGET, 
        ax=axes[i],
        color=colors[i],
        line_kws={"color": "#C44E52", "lw": 2.5}
    )
    
    axes[i].set_title(f"{metric.capitalize()} vs Total Score", fontsize=13, fontweight="bold")
    axes[i].set_xlabel("Indicator Score", fontsize=10)
    axes[i].set_ylabel("Total Score", fontsize=10)
    axes[i].grid(True, linestyle='--', alpha=0.5)

axes[5].set_visible(False)

plt.suptitle("Linearity Verification: All 5 THE Indicators vs Total Score (2015)", 
             fontsize=17, fontweight="bold", y=0.98)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig("../figures/png/linearity_check.png", dpi=200, bbox_inches="tight")