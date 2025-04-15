import os
import json
import math
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import pgeocode

# --- Load environment ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# --- Load HBCU data ---
DATA_PATH = "/home/cary/test_api/synthetic_hbcu_dataset.json"  # update path if needed
with open(DATA_PATH, "r") as f:
    hbcu_schools = json.load(f)

# --- Static options ---
campus_size_options = {"Small": "Small", "Medium": "Medium", "Large": "Large"}
academic_interest_options = [
    "STEM", "Nursing", "Public Health", "Education",
    "Criminal Justice", "Business", "Psychology",
    "Sociology", "Political Science", "Arts & Humanities"
]
priority_factor_options = [
    "Proximity", "Affordability", "Online availability",
    "Legacy or cultural history", "Campus life and activities", "Academic programs"
]

# --- Functions ---
def haversine_distance(zip1, zip2):
    nomi = pgeocode.Nominatim("us")
    loc1 = nomi.query_postal_code(zip1)
    loc2 = nomi.query_postal_code(zip2)
    if None in [loc1.latitude, loc1.longitude, loc2.latitude, loc2.longitude]:
        return None
    R = 3958.8
    dlat = math.radians(loc2.latitude - loc1.latitude)
    dlon = math.radians(loc2.longitude - loc1.longitude)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(loc1.latitude)) * math.cos(math.radians(loc2.latitude)) * math.sin(dlon/2)**2
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 2)

def attach_distances(user_zip, schools):
    for school in schools:
        school["distance"] = haversine_distance(user_zip, school["zip"])
    return schools

def build_prompt(user_input, school_list):
    prompt = (
        "You are a college recommendation assistant. Recommend the top 2–3 HBCUs based on the user's structured input. "
        "Use proximity only if the user said it was important, and do not display actual distances. "
        f"Strongly prioritize: {', '.join(user_input['priority_factors'])}. "
        "Use fields like affordability, online learning, majors, and legacy. Explain each recommendation in plain language."
    )
    return prompt, {
        "preferences": user_input,
        "schools": school_list
    }

# --- Streamlit UI ---
st.set_page_config(page_title="OnRamp Prototype", layout="wide")
st.title("🎓 OnRamp: HBCU Explorer (AI Prototype)")

with st.form("matcher_form"):
    zip_code = st.text_input("Enter your ZIP code:", value="46077")

    col1, col2 = st.columns(2)
    with col1:
        proximity = st.slider("Staying close to home", 1, 5, 3)
        affordability = st.slider("Affordability", 1, 5, 3)
        online = st.slider("Online or hybrid learning", 1, 5, 3)
    with col2:
        legacy = st.slider("Historical or cultural legacy", 1, 5, 3)
        social = st.slider("Campus life and student activity", 1, 5, 3)

    campus_size = st.radio("Preferred campus size:", list(campus_size_options.values()))

    interests = st.multiselect("Pick up to 2 academic interests:", academic_interest_options, max_selections=2)

    priorities = st.multiselect(
        "Select 2 or 3 most important factors:",
        priority_factor_options,
        default=["Affordability", "Academic programs"],
        max_selections=3
    )

    submitted = st.form_submit_button("Find My Matches")

if submitted:
    user_input = {
        "zip_code": zip_code,
        "proximity_importance": proximity,
        "affordability_importance": affordability,
        "online_importance": online,
        "legacy_importance": legacy,
        "social_importance": social,
        "campus_size_preference": campus_size,
        "interests": interests,
        "priority_factors": priorities
    }

    with st.spinner("Asking GPT for matches..."):
        enriched_schools = attach_distances(zip_code, hbcu_schools)
        prompt, payload = build_prompt(user_input, enriched_schools)

        try:
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload)}
                ],
                temperature=0.0,
                max_tokens=500
            )
            result = response.choices[0].message.content
            st.success("🎓 GPT Recommendations")
            st.markdown(result)
        except Exception as e:
            st.error(f"GPT API error: {e}")
