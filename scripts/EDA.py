import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import kaleido

kaleido.get_chrome_sync()

df_2015 = pd.read_csv("../data/times_clean_2015.csv")
df_all = pd.read_csv("../data/times_clean_all.csv")

METRICS = ["teaching", "research", "citations", "international", "income"]

# 1 - GEOGRAPHY

country_counts = (
    df_2015["country"]
    .value_counts()
    .head(15)
    .reset_index()
)
country_counts.columns = ["country", "count"]

fig = px.bar(
    country_counts,
    x="count",
    y="country",
    orientation="h",
    color="count",
    color_continuous_scale="Blues",
    title="Top 15 Countries by Number of Universities (2015)",
    labels={"count": "Number of Universities", "country": ""}
)

fig.update_layout(
    yaxis=dict(autorange="reversed"),
    coloraxis_showscale=False,
    height=500
)

fig.write_html("../figures/html/geography.html")
fig.write_image("../figures/png/geography.png", scale=2)

# 2 - HISTOGRAMS

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

for i, metric in enumerate(METRICS):
    sns.histplot(
        df_2015[metric].dropna(),
        ax=axes[i],
        kde=True,
        color=colors[i],
        bins=20,
        edgecolor="white"
    )
    axes[i].set_title(metric.capitalize(), fontsize=13, fontweight="bold")
    axes[i].set_xlabel("Score")
    axes[i].set_ylabel("Count")

axes[5].set_visible(False)

plt.suptitle("Distribution of THE Indicators (2015)", 
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("../figures/png/histograms.png", dpi=150, bbox_inches="tight")

# 3 - BOX PLOTS

df_melted = df_2015[METRICS].melt(
    var_name="metric", 
    value_name="score"
)

fig = px.box(
    df_melted,
    x="metric",
    y="score",
    color="metric",
    title="Distribution of THE Indicators (2015)",
    labels={"metric": "Indicator", "score": "Score"},
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig.update_layout(showlegend=False, height=500)
fig.write_html("../figures/html/box_plots.html")
fig.write_image("../figures/png/box_plots.png", scale=2)

# 4 - DESCRIPTIVE

stats = df_2015[METRICS].describe().round(2).T
stats = stats[["mean", "std", "min", "25%", "50%", "75%", "max"]]
stats.columns = ["Mean", "Std", "Min", "25%", "Median", "75%", "Max"]

print("=" * 65)
print("DESCRIPTIVE STATISTICS — df_2015")
print("=" * 65)
print(stats.to_string())

# 5 - TOP-10

top10 = (
    df_2015
    .nsmallest(10, "world_rank")
    [["world_rank", "university_name", "country", "total_score"] + METRICS]
    .reset_index(drop=True)
)

fig = px.bar(
    top10,
    x="total_score",
    y="university_name",
    orientation="h",
    color="total_score",
    color_continuous_scale="Blues",
    title="Top 10 Universities by Total Score (2015)",
    labels={"total_score": "Total Score", "university_name": ""}
)

fig.update_layout(
    yaxis=dict(autorange="reversed"),
    coloraxis_showscale=False,
    height=450
)

fig.write_html("../figures/html/top_10.html")
fig.write_image("../figures/png/top_10.png", scale=2)

# 6 - SCATTER 

country_stats = (
    df_2015
    .groupby("country")
    .agg(
        count=("university_name", "count"),
        avg_score=("total_score", "mean"),
        total_score=("total_score", "sum")
    )
    .reset_index()
    .round(2)
)

fig = px.scatter(
    country_stats,
    x="count",
    y="avg_score",
    size="total_score",
    text="country",
    title="Number of Universities vs Average Score by Country (2015)",
    labels={
        "count": "Number of Universities",
        "avg_score": "Average Total Score",
    },
    color="avg_score",
    color_continuous_scale="Blues"
)

fig.update_traces(textposition="top center", textfont_size=9)
fig.update_layout(height=550, coloraxis_showscale=False)
fig.write_html("../figures/html/scatter.html")
fig.write_image("../figures/png/scatter.png", scale=2)


# 7 - TOTAL SCORE PER YEARS

yearly_avg = (
    df_all
    .groupby("year")["total_score"]
    .mean()
    .reset_index()
    .round(2)
)

fig = px.line(
    yearly_avg,
    x="year",
    y="total_score",
    markers=True,
    title="Average Total Score Across All Universities (2011-2016)",
    labels={"total_score": "Average Total Score", "year": "Year"}
)

fig.update_traces(line_color="#4C72B0", line_width=2.5, marker_size=8)
fig.update_layout(height=400)
fig.write_html("../figures/html/total_score_per_year.html")
fig.write_image("../figures/png/total_score_per_year.png", scale=2)

# 8 - BUMP CHART

top_countries = (
    df_all["country"]
    .value_counts()
    .head(7)
    .index
    .tolist()
)

bump_data = (
    df_all[df_all["country"].isin(top_countries)]
    .groupby(["year", "country"])["world_rank"]
    .mean()
    .reset_index()
    .round(1)
)

fig = px.line(
    bump_data,
    x="year",
    y="world_rank",
    color="country",
    markers=True,
    title="Average World Rank by Country (2011-2016)",
    labels={"world_rank": "Average Rank", "year": "Year"}
)

fig.update_layout(
    yaxis=dict(autorange="reversed"),
    height=500,
    legend_title="Country"
)

fig.write_html("../figures/html/bump_charts.html")
fig.write_image("../figures/png/bump_charts.png", scale=2)
