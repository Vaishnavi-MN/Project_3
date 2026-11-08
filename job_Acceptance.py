import pandas as pd

import streamlit as st

cleaned_data = pd.read_csv(r"C:\Users\mnvij\Python_sample\cleaned_data.csv")
results_df = pd.read_csv(r"C:\Users\mnvij\Python_sample\results_df.csv")
feature_scores = pd.read_csv(r"C:\Users\mnvij\Python_sample\feature_scores.csv")

data = cleaned_data.copy()
print(data.shape)
print(data["status"].unique())
total_candidates = len(data)

placed_candidates = (data["status"] == 1).sum()

drop_out_candidates = (data["status"] == 0).sum()

placement_rate = (
    placed_candidates / total_candidates * 100
)

job_acceptance_rate = (
    placed_candidates / total_candidates * 100
)

average_interview_score = (
    data["interview_performance_score"].mean()
)

average_skills_match = (
    data["skills_match_percentage"].mean()
)

offer_dropout_rate = (
    (drop_out_candidates/total_candidates) * 100
)

data["high_risk"] = (
    (data["skills_match_percentage"] < 50)
    |
    (data["technical_score"] < 50)
    |
    (data["communication_score"] < 50)
    ).astype(int)

high_risk_percentage = (
    data["high_risk"].mean() * 100
)


st.title("Job Acceptance Analytics Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Candidates", total_candidates)

with col2:
    st.metric("Placement Rate", f"{placement_rate:.2f}%")

with col3:
    st.metric("job Acceptance Rate", f"{job_acceptance_rate:.2f}%")

with col4:
    st.metric("Average Interview Score",
              f"{average_interview_score:.2f}")

col5, col6, col7 = st.columns(3)

with col5:
    st.metric("Average Skills Match",
              f"{average_skills_match:.2f}%")

with col6:
    st.metric("Offer Dropout Rate",
              f"{offer_dropout_rate:.2f}%")
    
with col7:
    st.metric("High Risk Percentage",
              f"{high_risk_percentage:.2f}%")
    
st.header("📊 Candidate Placement Analysis")

st.subheader("Candidate Placement Distribution")

placement_counts = cleaned_data["status"].value_counts()

placement_counts.index = placement_counts.index.map({
    1: "Placed",
    0: "Not Placed"
})

st.bar_chart(placement_counts)

placement_percentage = (
    cleaned_data["status"]
    .value_counts(normalize=True) * 100
)

placement_percentage.index = placement_percentage.index.map({
    1: "Placed",
    0: "Not Placed"
})

st.subheader("Placement Percentage")

st.bar_chart(placement_percentage)

placement_summary = cleaned_data["status"].value_counts().reset_index()

placement_summary.columns = ["Placement Status", "Candidates"]

placement_summary["Placement Status"] = (
    placement_summary["Placement Status"].map({
        1: "Placed",
        0: "Not Placed"
    })
)

st.dataframe(placement_summary, use_container_width=True)

st.header("📈 Candidate Performance Analysis")

st.subheader("Academic Scores vs Placement")

academic_columns = [
    "ssc_percentage",
    "hsc_percentage",
    "degree_percentage"
]

academic_analysis = cleaned_data.groupby("status")[academic_columns].mean()

academic_analysis.index = academic_analysis.index.map({
    1: "Placed",
    0: "Not Placed"
})

st.bar_chart(academic_analysis)

st.subheader("Technical Score vs Placement")

technical_analysis = cleaned_data.groupby("status")["technical_score"].mean()

technical_analysis.index = technical_analysis.index.map({
    1: "Placed",
    0: "Not Placed"
})

st.bar_chart(technical_analysis)

st.subheader("Skills Match vs Placement")

skills_analysis = cleaned_data.groupby("status")["skills_match_percentage"].mean()

skills_analysis.index = skills_analysis.index.map({
    1: "Placed",
    0: "Not Placed"
})

st.bar_chart(skills_analysis)

st.subheader("Interview Performance vs Placement")

interview_analysis = cleaned_data.groupby("status")[
    "interview_performance_score"
].mean()

interview_analysis.index = interview_analysis.index.map({
    1: "Placed",
    0: "Not Placed"
})

st.bar_chart(interview_analysis)

st.header("💼 Experience & Job Analysis")

st.subheader("Experience vs Placement Rate")

experience_analysis = (
    cleaned_data.groupby("years_of_experience")["status"]
    .mean() * 100
)

st.bar_chart(experience_analysis)

st.caption("Placement success percentage by years of experience")

st.subheader("Company Tier vs Placement Rate")

company_analysis = (
    cleaned_data.groupby("company_tier")["status"]
    .mean() * 100
)

st.bar_chart(company_analysis)

st.header("🤖 Machine Learning Model Performance")

best_model_row = results_df.loc[
    results_df["F1 Score"].idxmax()
]

st.success(
    f"Best Model: {best_model_row['Model']}"
)

st.metric(
    "Best Model Accuracy",
    f"{best_model_row['Accuracy'] * 100:.2f}%"
)

st.metric(
    "Best Model F1 Score",
    f"{best_model_row['F1 Score']:.4f}"
)

