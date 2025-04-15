import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_simulated_dataset(n_segments=20, n_users=3, fixations_per_segment=3):
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
                    "fixation_duration": np.random.normal(300, 80),
                    "regression_count": np.random.poisson(0.5),
                    "pupil_diameter": np.random.normal(3.2, 0.3)
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

def get_top_segments_to_summarize(agg_df, user_id, top_n=5):
    user_df = agg_df[agg_df["user_id"] == user_id]
    top_df = user_df.sort_values(by="attention_score", ascending=False).head(top_n)
    return top_df, " ".join(top_df["text"].tolist())

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

def plot_heatmap(df, user_id):
    df = df[df["user_id"] == user_id]
    plt.figure(figsize=(12, 6))
    heatmap_df = df.copy()
    heatmap_df["x_bin"] = pd.cut(heatmap_df["x"], bins=30, labels=False)
    heatmap_df["y_bin"] = pd.cut(heatmap_df["y"], bins=20, labels=False)
    pivot = heatmap_df.groupby(["y_bin", "x_bin"])["fixation_duration"].sum().unstack().fillna(0)
    sns.heatmap(pivot, cmap="hot", cbar_kws={"label": "Fixation Duration"})
    plt.title(f"Gaze Heatmap for {user_id}")
    plt.xlabel("X Position (Binned)")
    plt.ylabel("Y Position (Binned)")
    plt.tight_layout()
    plt.show()

def plot_attention_scores(top_df, user_id):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_df, x="segment_id", y="attention_score", palette="Blues_d")
    plt.title(f"Top Segments Summarized for {user_id}")
    plt.ylabel("Attention Score")
    plt.xlabel("Segment ID")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def main():
    if not OPENAI_API_KEY:
        print("Missing OpenAI API key. Set it in a .env file.")
        return

    df_raw = generate_simulated_dataset()
    df_agg = aggregate_by_segment(df_raw)

    user_id = "user_001"
    top_df, combined_text = get_top_segments_to_summarize(df_agg, user_id)

    print("Top Segments for Summarization:\n", top_df[["segment_id", "attention_score", "text"]])
    print("\nSummarizing the following text:\n", combined_text)

    summary = summarize_text(combined_text)
    print("\nSummary:\n", summary)

    plot_attention_scores(top_df, user_id)
    plot_heatmap(df_raw, user_id)

if __name__ == "__main__":
    main()