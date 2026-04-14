"""
Data Cleaner for CodeMentor-LLM
Cleans and filters the formatted dataset before training.
"""

import pandas as pd
from datasets import Dataset


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    
    """
    Remove duplicate rows from dataset.
    Args:
        df: pandas DataFrame with text column
    Returns:
        tuple of cleaned DataFrame and count of removed rows
    """
    
    original_count = len(df)
    df_deduped = df.drop_duplicates(subset=["text"])
    removed = original_count - len(df_deduped)
    return df_deduped, removed


def remove_nulls(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    
    """
    Remove null and empty rows from dataset.
    Args:
        df: pandas DataFrame with text column
    Returns:
        tuple of cleaned DataFrame and count of removed rows
    """
    
    original_count = len(df)
    df_clean = df.dropna(subset=["text"])
    df_clean = df_clean[df_clean["text"] != ""]
    removed = original_count - len(df_clean)
    return df_clean, removed


def filter_by_token_length(df: pd.DataFrame,tokenizer,min_tokens: int = 10,
                           max_tokens: int = 2048) -> tuple[pd.DataFrame, int]:
    
    """
    Filter samples by token length.
    Args:
        df: pandas DataFrame with text column
        tokenizer: Llama-3 tokenizer
        min_tokens: minimum token length
        max_tokens: maximum token length
    Returns:
        tuple of filtered DataFrame and count of removed rows
    """
    
    original_count = len(df)
    df["token_length"] = df["text"].apply(
        lambda x: len(tokenizer(x)["input_ids"])
    )
    df_filtered = df[
        (df["token_length"] >= min_tokens) &
        (df["token_length"] <= max_tokens)
    ].drop(columns=["token_length"])
    removed = original_count - len(df_filtered)
    return df_filtered, removed


def is_low_quality(text: str) -> bool:

    """
    Check if a sample is low quality.
    Low quality = assistant response shorter than 3 words.
    Args:
        text: formatted prompt text
    Returns:
        True if low quality, False otherwise
    """

    if "<|start_header_id|>assistant<|end_header_id|>" in text:
        response = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        response = response.replace("<|eot_id|>", "").strip()
        if len(response.split()) < 3:
            return True
    return False


def filter_low_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:

    """
    Remove low quality samples from dataset.
    Args:
        df: pandas DataFrame with text column
    Returns:
        tuple of filtered DataFrame and count of removed rows
    """

    original_count = len(df)
    df["is_low_quality"] = df["text"].apply(is_low_quality)
    df_filtered = df[df["is_low_quality"] == False].drop(
        columns=["is_low_quality"]
    )
    removed = original_count - len(df_filtered)
    return df_filtered, removed


def clean_dataset(df: pd.DataFrame, tokenizer) -> tuple[pd.DataFrame, dict]:

    """
    Run full cleaning pipeline on dataset.
    Args:
        df: pandas DataFrame with text column
        tokenizer: Llama-3 tokenizer
    Returns:
        tuple of cleaned DataFrame and cleaning log
    """

    cleaning_log = {}
    cleaning_log["original"] = len(df)

    # Step 1 — Remove nulls
    df, removed_nulls = remove_nulls(df)
    cleaning_log["after_null_filter"] = len(df)
    cleaning_log["removed_nulls"] = removed_nulls

    # Step 2 — Remove duplicates
    df, removed_dedup = remove_duplicates(df)
    cleaning_log["after_dedup"] = len(df)
    cleaning_log["removed_duplicates"] = removed_dedup

    # Step 3 — Filter by token length
    df, removed_tokens = filter_by_token_length(df, tokenizer)
    cleaning_log["after_token_filter"] = len(df)
    cleaning_log["removed_by_token_length"] = removed_tokens

    # Step 4 — Remove low quality
    df, removed_quality = filter_low_quality(df)
    cleaning_log["after_quality_filter"] = len(df)
    cleaning_log["removed_low_quality"] = removed_quality

    # Summary
    cleaning_log["total_removed"] = cleaning_log["original"] - len(df)
    cleaning_log["retention_rate"] = len(df) / cleaning_log["original"] * 100

    return df, cleaning_log


if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct"
    )

    # Load formatted dataset
    dataset = load_dataset("Abdulmoiz123/codementor-llm-formatted")
    df = pd.DataFrame(dataset["train"])

    # Clean dataset
    df_clean, log = clean_dataset(df, tokenizer)

    print("Cleaning Summary:")
    for key, value in log.items():
        print(f"  {key}: {value}")