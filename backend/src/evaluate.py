"""
Evaluation Script for CodeMentor-LLM
Computes ROUGE and BERTScore for base and SFT models.
"""

import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from config import MODEL_ID, bnb_config, SYSTEM_PROMPT


def load_tokenizer(model_id: str):
    """Load tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_sft_model(model_id: str, adapter_id: str):
    """Load base model with SFT adapter."""
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_id)
    return model


def generate_response(model, tokenizer, instruction: str, max_new_tokens: int = 256) -> str:
    """Generate response for a given instruction."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
        )

    response = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True
    )
    return response


def extract_instruction(text: str) -> str:
    """Extract instruction from formatted text."""
    if "<|start_header_id|>user<|end_header_id|>" in text:
        instruction = text.split("<|start_header_id|>user<|end_header_id|>")[-1]
        instruction = instruction.split("<|eot_id|>")[0].strip()
        return instruction
    return ""


def extract_reference(text: str) -> str:
    """Extract reference response from formatted text."""
    if "<|start_header_id|>assistant<|end_header_id|>" in text:
        response = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        response = response.replace("<|eot_id|>", "").strip()
        return response
    return ""


def compute_rouge(predictions: list, references: list) -> dict:
    """Compute ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    rouge1, rouge2, rougeL = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougeL.append(scores['rougeL'].fmeasure)
    return {
        "rouge1": sum(rouge1) / len(rouge1),
        "rouge2": sum(rouge2) / len(rouge2),
        "rougeL": sum(rougeL) / len(rougeL),
    }


def compute_bertscore(predictions: list, references: list) -> dict:
    """Compute BERTScore."""
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }


def evaluate(adapter_id: str, n_samples: int = 50):
    """Run full evaluation pipeline."""
    # Load tokenizer and model
    tokenizer = load_tokenizer(MODEL_ID)
    model = load_sft_model(MODEL_ID, adapter_id)

    # Load test dataset
    dataset = load_dataset("Abdulmoiz123/codementor-llm-splits")
    test_dataset = dataset["test"].select(range(n_samples))

    # Extract instructions and references
    instructions = [extract_instruction(s["text"]) for s in test_dataset]
    references = [extract_reference(s["text"]) for s in test_dataset]

    # Generate SFT predictions
    print("Generating SFT predictions...")
    sft_predictions = [
        generate_response(model, tokenizer, inst)
        for inst in instructions
    ]

    # Generate base predictions
    print("Generating base predictions...")
    model.disable_adapter_layers()
    base_predictions = [
        generate_response(model, tokenizer, inst)
        for inst in instructions
    ]
    model.enable_adapter_layers()

    # Compute metrics
    sft_rouge = compute_rouge(sft_predictions, references)
    base_rouge = compute_rouge(base_predictions, references)
    sft_bert = compute_bertscore(sft_predictions, references)
    base_bert = compute_bertscore(base_predictions, references)

    # Save results
    results = {
        "model": adapter_id,
        "base_model": MODEL_ID,
        "test_samples": n_samples,
        "rouge_scores": {"base": base_rouge, "sft": sft_rouge},
        "bert_scores": {"base": base_bert, "sft": sft_bert},
    }

    with open("data/results/sft_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Evaluation complete")
    print(f"ROUGE-L: Base={base_rouge['rougeL']:.4f} SFT={sft_rouge['rougeL']:.4f}")
    print(f"BERTScore F1: Base={base_bert['f1']:.4f} SFT={sft_bert['f1']:.4f}")

    return results


if __name__ == "__main__":
    evaluate("Abdulmoiz123/codementor-llm-sft")