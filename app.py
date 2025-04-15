import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

@st.cache_data
def generate_simulated_dataset(n_segments=20, n_users=3, fixations_per_segment=3):
    import numpy as np
    np.random.seed(42)
    base_texts = [
        "Artificial intelligence is reshaping industries.",
        "Machine learning allows systems to learn from data.",
        "Climate change poses global environmental threats.",
        "Blockchain introduces decentralized record-keeping.",
        "Remote work is redefining modern office culture.",
        "Quantum computing could solve complex problems faster.",
        "Mental health awareness is increasing worldwide.",
        "Augmented reality is enhancing digital experiences.",
        "Biotech is advancing personalized medicine.",
        "Renewable energy adoption is accelerating.",
        "Cybersecurity is critical in the digital age.",
        "Space exploration is entering a new golden age.",
        "Education technology is transforming learning.",
        "Digital privacy concerns are rising.",
        "Food tech is solving sustainability challenges.",
        "Social media impacts communication and society.",
        "Autonomous vehicles use sensors and AI to drive.",
        "Genetic editing opens doors for treatment innovation.",
        "Robotics is automating labor-intensive tasks.",
        "5G networks enable faster data transfer and IoT."
    ]
    texts = base_texts[:n_segments]
    data = []
    for user in range(1, n_users + 1):
        for segment_id, text in enumerate(texts, 1):
            for _ in range(fixations_per_segment):
                data.append({
                    "user_id": f"user_{user:03}",
                    "segment_id": segment_id,
                    "text": text,
                    "x": np.random.randint(0, 1920),
                    "y": np.random.randint(0, 1080),
                    "fixation_duration": max(50, np.random.normal(300, 80)),
                    "regression_count": np.random.poisson(0.5),
                    "pupil_diameter": max(2.0, np.random.normal(3.2, 0.3))
                })
    return pd.DataFrame(data)

def aggregate_by_segment(df):
    agg_df = df.groupby(['user_id', 'segment_id', 'text']).agg({
        'fixation_duration': 'sum',
        'regression_count': 'sum',
        'pupil_diameter': 'mean'
    }).reset_index()
    agg_df["attention_score"] = (
        agg_df["fixation_duration"] +
        agg_df["regression_count"] * 80 +
        agg_df["pupil_diameter"] * 50
    )
    return agg_df

def summarize_text(text, model="gpt-4", max_tokens=250):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Please summarize the following text:\n\n{text}"}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred while summarizing: {str(e)}"

st.set_page_config(page_title="Gaze-Aware Summarizer", layout="wide")
st.title("👁️ Gaze-Aware Summarizer Dashboard")

df_raw = generate_simulated_dataset()
df_agg = aggregate_by_segment(df_raw)

user_ids = df_agg["user_id"].unique().tolist()
selected_user = st.sidebar.selectbox("Select User", user_ids)
top_n = st.sidebar.slider("Number of Segments to Summarize", 3, 10, 5)

user_df = df_agg[df_agg["user_id"] == selected_user]
top_df = user_df.sort_values("attention_score", ascending=False).head(top_n)
combined_text = " ".join(top_df["text"].tolist())

st.subheader(f"Top {top_n} Segments for {selected_user}")
st.dataframe(top_df[["segment_id", "attention_score", "text"]], use_container_width=True)

st.subheader("📊 Attention Scores")
fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.barplot(data=top_df, x="segment_id", y="attention_score", ax=ax1, palette="Blues_d")
ax1.set_title(f"Top Segments by Attention for {selected_user}")
st.pyplot(fig1)

st.subheader("🔥 Gaze Heatmap")
user_gaze = df_raw[df_raw["user_id"] == selected_user]
heatmap_df = user_gaze.copy()
heatmap_df["x_bin"] = pd.cut(heatmap_df["x"], bins=30, labels=False)
heatmap_df["y_bin"] = pd.cut(heatmap_df["y"], bins=20, labels=False)
pivot = heatmap_df.groupby(["y_bin", "x_bin"])["fixation_duration"].sum().unstack().fillna(0)
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.heatmap(pivot, cmap="hot", ax=ax2)
ax2.set_title(f"Gaze Heatmap for {selected_user}")
st.pyplot(fig2)

st.subheader("🧠 GPT-4 Summary")
with st.spinner("Summarizing using GPT-4..."):
    summary = summarize_text(combined_text)
st.success("Summary complete!")
st.write(summary)