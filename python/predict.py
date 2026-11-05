import pandas as pd
import joblib

# Load the trained model
model = joblib.load("models/final_model.pkl")

print("=== Job Acceptance Prediction System ===")

# Get candidate details
age_years = int(input("Enter age: "))
gender = input("Enter gender (Male/Female): ")

ssc_percentage = float(input("Enter SSC percentage: "))
hsc_percentage = float(input("Enter HSC percentage: "))
degree_percentage = float(input("Enter degree percentage: "))

degree_specialization = input("Enter degree specialization: ")

technical_score = float(input("Enter technical score: "))
aptitude_score = float(input("Enter aptitude score: "))
communication_score = float(input("Enter communication score: "))

skills_match_percentage = float(input("Enter skills match percentage: "))
certifications_count = int(input("Enter certifications count: "))

internship_experience = input("Internship experience (Yes/No): ")

years_of_experience = float(input("Enter years of experience: "))

career_switch_willingness = input("Career switch willingness (Yes/No): ")
relevant_experience = input("Relevant experience (Yes/No): ")

previous_ctc_lpa = float(input("Enter previous CTC (LPA): "))
expected_ctc_lpa = float(input("Enter expected CTC (LPA): "))

company_tier = input("Enter company tier: ")
job_role_match = input("Enter job role match: ")

competition_level = input("Enter competition level: ")
bond_requirement = input("Bond requirement (Yes/No): ")

notice_period_days = int(input("Enter notice period days: "))
layoff_history = input("Layoff history (Yes/No): ")

employment_gap_months = int(input("Enter employment gap months: "))
relocation_willingness = input("Relocation willingness (Yes/No): ")

# Feature Engineering

if years_of_experience == 0:
    experience_category = "Fresher"
elif years_of_experience <= 2:
    experience_category = "Junior"
else:
    experience_category = "Senior"

if degree_percentage >= 75:
    academic_performance_band = "High"
elif degree_percentage >= 60:
    academic_performance_band = "Medium"
else:
    academic_performance_band = "Low"

if skills_match_percentage < 60:
    skills_match_level = "Low"
elif skills_match_percentage < 80:
    skills_match_level = "Medium"
else:
    skills_match_level = "High"

if communication_score >= 75:
    interview_performance_category = "Excellent"
elif communication_score >= 60:
    interview_performance_category = "Good"
else:
    interview_performance_category = "Needs Improvement"

placement_probability_score = (
    technical_score * 0.30
    + aptitude_score * 0.25
    + communication_score * 0.20
    + skills_match_percentage * 0.25
)

# Create candidate DataFrame
candidate = pd.DataFrame([{
    "age_years": age_years,
    "gender": gender,
    "ssc_percentage": ssc_percentage,
    "hsc_percentage": hsc_percentage,
    "degree_percentage": degree_percentage,
    "degree_specialization": degree_specialization,
    "technical_score": technical_score,
    "aptitude_score": aptitude_score,
    "communication_score": communication_score,
    "skills_match_percentage": skills_match_percentage,
    "certifications_count": certifications_count,
    "internship_experience": internship_experience,
    "years_of_experience": years_of_experience,
    "career_switch_willingness": career_switch_willingness,
    "relevant_experience": relevant_experience,
    "previous_ctc_lpa": previous_ctc_lpa,
    "expected_ctc_lpa": expected_ctc_lpa,
    "company_tier": company_tier,
    "job_role_match": job_role_match,
    "competition_level": competition_level,
    "bond_requirement": bond_requirement,
    "notice_period_days": notice_period_days,
    "layoff_history": layoff_history,
    "employment_gap_months": employment_gap_months,
    "relocation_willingness": relocation_willingness,
    "experience_category": experience_category,
    "academic_performance_band": academic_performance_band,
    "skills_match_level": skills_match_level,
    "interview_performance_category": interview_performance_category,
    "placement_probability_score": placement_probability_score
}])

# Prediction
prediction = model.predict(candidate)

print("\n=== Prediction Result ===")

if prediction[0] == 1:
    print("Candidate Status: PLACED")
else:
    print("Candidate Status: NOT PLACED")