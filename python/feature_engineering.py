import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned/cleaned_job_acceptance.csv")

# 1. Experience Category
def experience_category(years):
    if years == 0:
        return "Fresher"
    elif years <= 2:
        return "Junior"
    else:
        return "Senior"

df["experience_category"] = df["years_of_experience"].apply(
    experience_category
)

# 2. Academic Performance Band
def academic_band(score):
    if score >= 75:
        return "High"
    elif score >= 60:
        return "Medium"
    else:
        return "Low"

df["academic_performance_band"] = df["degree_percentage"].apply(
    academic_band
)

# 3. Skills Match Level
def skills_level(score):
    if score < 60:
        return "Low"
    elif score < 80:
        return "Medium"
    else:
        return "High"

df["skills_match_level"] = df["skills_match_percentage"].apply(
    skills_level
)

# 4. Interview Performance Category
def interview_category(score):
    if score >= 75:
        return "Excellent"
    elif score >= 60:
        return "Good"
    else:
        return "Needs Improvement"

df["interview_performance_category"] = df["communication_score"].apply(
    interview_category
)

# 5. Placement Probability Score
df["placement_probability_score"] = (
    df["technical_score"] * 0.30
    + df["aptitude_score"] * 0.25
    + df["communication_score"] * 0.20
    + df["skills_match_percentage"] * 0.25
)

# Display new features
print("New Feature Columns:")
print(df[[
    "experience_category",
    "academic_performance_band",
    "skills_match_level",
    "interview_performance_category",
    "placement_probability_score"
]].head())

# Save dataset
df.to_csv("feature_engineered_job_acceptance.csv", index=False)

print("\nFeature Engineering Completed!")
print("Final Dataset Shape:", df.shape)
print("Saved as feature_engineered_job_acceptance.csv")