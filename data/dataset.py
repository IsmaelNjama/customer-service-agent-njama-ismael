from datasets import load_dataset
import pandas as pd


def load_bitext():
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    )
    df = pd.DataFrame(ds["train"])
    return df


# if __name__ == "__main__":
#     df = load_bitext()
#     print("Columns:", df.columns.tolist())
#     print("Intents:", df['intent'].unique())
#     print("Categories:", df['category'].unique())
#     print("Shape:", df.shape)
