
import os
import joblib
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Job Acceptance Prediction System",
    page_icon="🎯",
    layout="wide"
)

# =========================================================
# PATH CONFIGURATION
# =========================================================

MODEL_PATH = "models/final_model.pkl"

HISTORY_DIR = "reports/prediction_history"
HISTORY_FILE = os.path.join(
    HISTORY_DIR,
    "prediction_history.csv"
)

os.makedirs(HISTORY_DIR, exist_ok=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🎯 Job Prediction System")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "About Project",
        "Prediction History",
        "Analytics Dashboard"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Predict whether a candidate is likely to get placed "
    "based on academic, technical, and career-related details."
)

# =========================================================
# HOME PAGE
# =========================================================

if page == "Home":

    st.title("🎯 Job Acceptance Prediction System")

    st.write(
        "Enter the candidate details below to predict "
        "whether the candidate is likely to get placed."
    )

    st.markdown("---")

    with st.form("prediction_form"):

        st.subheader("👤 Personal Information")

        col1, col2 = st.columns(2)

        with col1:
            age_years = st.number_input(
                "Age",
                min_value=18,
                max_value=60,
                value=22
            )

            gender = st.selectbox(
                "Gender",
                ["Male", "Female", "Other"]
            )

        with col2:
            degree_specialization = st.selectbox(
                "Degree Specialization",
                [
                    "Computer Science",
                    "Information Technology",
                    "Artificial Intelligence",
                    "Data Science",
                    "Electronics",
                    "Mechanical",
                    "Civil",
                    "Electrical",
                    "Other"
                ]
            )

        st.markdown("---")

        st.subheader("🎓 Academic Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            ssc_percentage = st.number_input(
                "SSC Percentage",
                min_value=0.0,
                max_value=100.0,
                value=80.0
            )

        with col2:
            hsc_percentage = st.number_input(
                "HSC Percentage",
                min_value=0.0,
                max_value=100.0,
                value=80.0
            )

        with col3:
            degree_percentage = st.number_input(
                "Degree Percentage",
                min_value=0.0,
                max_value=100.0,
                value=75.0
            )

        st.markdown("---")

        st.subheader("💻 Skills and Assessment")

        col1, col2 = st.columns(2)

        with col1:
            technical_score = st.number_input(
                "Technical Score",
                min_value=0.0,
                max_value=100.0,
                value=75.0
            )

            aptitude_score = st.number_input(
                "Aptitude Score",
                min_value=0.0,
                max_value=100.0,
                value=70.0
            )

            communication_score = st.number_input(
                "Communication Score",
                min_value=0.0,
                max_value=100.0,
                value=75.0
            )

        with col2:
            skills_match_percentage = st.number_input(
                "Skills Match Percentage",
                min_value=0.0,
                max_value=100.0,
                value=80.0
            )

            certifications_count = st.number_input(
                "Number of Certifications",
                min_value=0,
                max_value=50,
                value=2
            )

            internship_experience = st.selectbox(
                "Internship Experience",
                ["Yes", "No"]
            )

        st.markdown("---")

        st.subheader("💼 Career Information")

        col1, col2 = st.columns(2)

        with col1:
            years_of_experience = st.number_input(
                "Years of Experience",
                min_value=0.0,
                max_value=40.0,
                value=0.0
            )

            career_switch_willingness = st.selectbox(
                "Career Switch Willingness",
                ["Yes", "No"]
            )

            relevant_experience = st.selectbox(
                "Relevant Experience",
                ["Yes", "No"]
            )

            previous_ctc_lpa = st.number_input(
                "Previous CTC (LPA)",
                min_value=0.0,
                max_value=100.0,
                value=0.0
            )

            expected_ctc_lpa = st.number_input(
                "Expected CTC (LPA)",
                min_value=0.0,
                max_value=100.0,
                value=4.0
            )

        with col2:
            company_tier = st.selectbox(
                "Company Tier",
                ["Tier 1", "Tier 2", "Tier 3"]
            )

            job_role_match = st.selectbox(
                "Job Role Match",
                ["High", "Medium", "Low"]
            )

            competition_level = st.selectbox(
                "Competition Level",
                ["Low", "Medium", "High"]
            )

            bond_requirement = st.selectbox(
                "Bond Requirement",
                ["Yes", "No"]
            )

            notice_period_days = st.number_input(
                "Notice Period (Days)",
                min_value=0,
                max_value=365,
                value=0
            )

        st.markdown("---")

        st.subheader("📋 Additional Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            layoff_history = st.selectbox(
                "Layoff History",
                ["Yes", "No"]
            )

        with col2:
            employment_gap_months = st.number_input(
                "Employment Gap (Months)",
                min_value=0,
                max_value=120,
                value=0
            )

        with col3:
            relocation_willingness = st.selectbox(
                "Relocation Willingness",
                ["Yes", "No"]
            )

        st.markdown("---")

        submit_button = st.form_submit_button(
            "🔮 Predict Job Placement",
            use_container_width=True
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    if submit_button:

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

        # Create Input DataFrame

        input_data = {
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
        }

        input_df = pd.DataFrame([input_data])

        # Predict

        try:
            prediction = model.predict(input_df)[0]

            st.markdown("---")
            st.subheader("📊 Prediction Result")

            if prediction == "PLACED":

                st.success(
                    "🎉 Congratulations! The candidate is predicted to get placed."
                )

            else:

                st.warning(
                    "⚠️ The candidate is predicted to be not placed."
                )

            # Probability Score Display

            st.subheader("Placement Probability Score")

            probability_score = placement_probability_score

            st.progress(
                min(
                    max(int(probability_score), 0),
                    100
                )
            )

            st.metric(
                "Placement Probability Score",
                f"{probability_score:.1f}%"
            )

            st.caption(
                "This score is calculated from the candidate's "
                "technical, aptitude, communication, and skills-match scores. "
                "It is not the model's calibrated probability."
            )

            # Save Prediction History

            prediction_record = {
                "prediction_date": pd.Timestamp.now(),
                "prediction_result": prediction,
                "placement_probability_score": probability_score,
                "age_years": age_years,
                "degree_percentage": degree_percentage,
                "technical_score": technical_score,
                "aptitude_score": aptitude_score,
                "communication_score": communication_score,
                "skills_match_percentage": skills_match_percentage
            }

            history_df = pd.DataFrame([prediction_record])

            if os.path.exists(HISTORY_FILE):

                history_df.to_csv(
                    HISTORY_FILE,
                    mode="a",
                    header=False,
                    index=False
                )

            else:

                history_df.to_csv(
                    HISTORY_FILE,
                    index=False
                )

            st.info("Prediction saved to history.")

        except Exception as e:

            st.error(f"Prediction Error: {e}")

# =========================================================
# ABOUT PROJECT PAGE
# =========================================================

elif page == "About Project":

    st.title("📘 About Job Acceptance Prediction System")

    st.markdown("""
    ### Project Overview

    The Job Acceptance Prediction System is a machine learning
    application that predicts whether a candidate is likely to
    get placed based on academic performance, technical skills,
    aptitude, communication, and career-related details.

    ### Technologies Used

    - Python
    - Pandas
    - Scikit-learn
    - Machine Learning
    - Streamlit
    - Joblib

    ### Machine Learning Model

    The trained machine learning model is loaded from:

    `models/final_model.pkl`

    ### Features

    - Candidate information input
    - Job placement prediction
    - Placement score calculation
    - Prediction history
    - Analytics dashboard
    - CSV history storage

    ### Prediction Classes

    - PLACED
    - NOT PLACED

    ### Disclaimer

    This application provides a machine learning prediction
    and should not be considered a guarantee of employment.
    """)

# =========================================================
# PREDICTION HISTORY PAGE
# =========================================================

elif page == "Prediction History":

    st.title("📜 Prediction History")

    if os.path.exists(HISTORY_FILE):

        history = pd.read_csv(HISTORY_FILE)

        if not history.empty:

            st.dataframe(
                history,
                use_container_width=True
            )

            st.subheader("Download Prediction History")

            csv_data = history.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name="prediction_history.csv",
                mime="text/csv"
            )

        else:

            st.info("No prediction history available.")

    else:

        st.info(
            "No prediction history found. "
            "Make a prediction first."
        )

# =========================================================
# ANALYTICS DASHBOARD PAGE
# =========================================================

elif page == "Analytics Dashboard":

    st.title("📈 Prediction Analytics Dashboard")

    if os.path.exists(HISTORY_FILE):

        history = pd.read_csv(HISTORY_FILE)

        if not history.empty:

            total_predictions = len(history)

            placed_count = (
                history["prediction_result"] == "PLACED"
            ).sum()

            not_placed_count = (
                history["prediction_result"] == "NOT PLACED"
            ).sum()

            placement_rate = (
                placed_count / total_predictions
            ) * 100

            # KPI Cards

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Predictions",
                total_predictions
            )

            col2.metric(
                "Placed",
                placed_count
            )

            col3.metric(
                "Not Placed",
                not_placed_count
            )

            col4.metric(
                "Placement Rate",
                f"{placement_rate:.1f}%"
            )

            # Prediction Distribution

            st.markdown("---")

            st.subheader("📊 Prediction Distribution")

            chart_data = history["prediction_result"].value_counts()

            st.bar_chart(chart_data)

            # Predictions Over Time

            if "prediction_date" in history.columns:

                st.markdown("---")

                st.subheader("📅 Predictions Over Time")

                history["prediction_date"] = pd.to_datetime(
                    history["prediction_date"],
                    errors="coerce"
                )

                history["prediction_day"] = (
                    history["prediction_date"].dt.strftime("%Y-%m-%d")
                )

                daily_predictions = (
                    history.dropna(subset=["prediction_day"])
                    .groupby("prediction_day")
                    .size()
                )

                st.line_chart(daily_predictions)

            # Average Score

            if "placement_probability_score" in history.columns:

                st.markdown("---")

                st.subheader("🎯 Average Placement Score")

                average_score = history[
                    "placement_probability_score"
                ].mean()

                st.metric(
                    "Average Placement Score",
                    f"{average_score:.2f}%"
                )

        else:

            st.info("No predictions available yet.")

    else:

        st.info(
            "No prediction history found. "
            "Make a prediction first."
        )
