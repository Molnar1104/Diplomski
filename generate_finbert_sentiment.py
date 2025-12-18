import pandas as pd
import pyarrow.parquet as pq
import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "thesis_data_sampled_100perDay.parquet" # Your existing parquet file
OUTPUT_FILE = "daily_sentiment_finbert.csv"          # New output file
BATCH_SIZE = 32                                      # Lower if you run out of RAM

def get_device():
    """Checks if a GPU is available to speed this up."""
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("⚠️ No GPU detected. Running on CPU (This will be slower).")
        return torch.device("cpu")

def generate_finbert_scores():
    # 1. Setup
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    device = get_device()
    
    print("⏳ Loading FinBERT model (ProsusAI/finbert)...")
    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
    model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
    model.to(device)
    model.eval() # Set model to evaluation mode (faster)

    # 2. Load Data
    print(f"📖 Reading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    
    # We use 'Lsa_summary' as the text. Fill NaNs just in case.
    texts = df['Lsa_summary'].fillna("").astype(str).tolist()
    dates = pd.to_datetime(df['Date'], errors='coerce')
    
    print(f"🚀 Processing {len(texts)} articles...")

    # 3. Processing Loop (Batched)
    results = []
    
    # tqdm creates the progress bar
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Analyzing Sentiment"):
        batch_texts = texts[i : i + BATCH_SIZE]
        
        # Tokenize inputs
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(device)

        # Get Predictions
        with torch.no_grad():
            outputs = model(**inputs)
            # FinBERT outputs 3 scores: [Positive, Negative, Neutral]
            probs = softmax(outputs.logits, dim=1)
            
        # Move back to CPU to save
        probs = probs.cpu().numpy()

        # FinBERT labels: 0=Positive, 1=Negative, 2=Neutral (Check model card to be sure)
        # ProsusAI/finbert mapping: {0: 'positive', 1: 'negative', 2: 'neutral'}
        
        for j, score_vector in enumerate(probs):
            # Calculate a single "Compound" score for the thesis
            # Score = Probability(Positive) - Probability(Negative)
            # Range: -1.0 (Pure Negative) to +1.0 (Pure Positive)
            pos = score_vector[0]
            neg = score_vector[1]
            neu = score_vector[2]
            
            compound_score = pos - neg
            
            # Map back to the correct date
            current_date = dates[i + j]
            
            results.append({
                'Date': current_date,
                'FinBERT_Score': compound_score,
                'Prob_Pos': pos,
                'Prob_Neg': neg,
                'Prob_Neu': neu
            })

    # 4. Aggregation (Daily Averages)
    print("\n📊 Aggregating daily scores...")
    result_df = pd.DataFrame(results)
    
    # Group by Date
    daily_sentiment = result_df.groupby(result_df['Date'].dt.date).agg({
        'FinBERT_Score': 'mean',  # Average sentiment of the day
        'Prob_Pos': 'mean',
        'Prob_Neg': 'mean',
        'Date': 'count'           # Volume of news
    }).rename(columns={'Date': 'News_Volume'})
    
    daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
    
    # 5. Save
    daily_sentiment.to_csv(OUTPUT_FILE)
    print("-" * 30)
    print(f"✅ DONE! Saved FinBERT sentiment to: {OUTPUT_FILE}")
    print(daily_sentiment.head())

if __name__ == "__main__":
    # Ensure necessary libraries are installed
    # pip install torch transformers tqdm
    generate_finbert_scores()