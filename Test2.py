import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast
from tokenizers import Tokenizer
from datasets import load_dataset
import json
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, confusion_matrix,
    matthews_corrcoef, roc_curve, auc
)
from collections import Counter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Data Loading and Model Setup ---

# Load the dataset directly. We will use all of it for testing.
# The 'train' split is the default name given by the load_dataset function when loading a single file.
try:
    dna_ft_dataset = load_dataset('json', data_files='/home/u5849124/dna-llama/promoter_data/test.json')
    test_data = dna_ft_dataset["train"]
except Exception as e:
    print(f"Could not load dataset. Using a dummy dataset for demonstration. Error: {e}")
    # Create a dummy dataset if the original is not available
    dummy_data = {
        "train": [
            {"instruction": "Is this sequence a promoter?", "input": "AGCT...", "output": "promoter"},
            {"instruction": "Is this sequence a promoter?", "input": "GATT...", "output": "non-promoter"}
        ] * 100 # Larger sample for testing
    }
    with open("dummy_test.json", "w") as f:
        json.dump(dummy_data, f)
    dna_ft_dataset = load_dataset('json', data_files='dummy_test.json')
    test_data = dna_ft_dataset["train"]


# Load the tokenizer and model for the RK4 version
try:
    tokenizer = LlamaTokenizerFast.from_pretrained("LlaMaDNAPromoter_AdamW")
    model = LlamaForCausalLM.from_pretrained(
        "LlaMaDNAPromoter_AdamW",
        device_map="auto"
    )
except Exception as e:
    print(f"Could not load model 'LlaMaDNAPromoter_RK4'. Using a base model instead. Error: {e}")
    # Fallback to a base model if the custom one isn't available
    base_model = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    tokenizer = LlamaTokenizerFast.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto")

tokenizer.pad_token = tokenizer.eos_token


# --- Core Functions ---

def format_input(entry):
    """Formats the input entry into a standardized instruction prompt."""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text + "\n\n### Response:\n"


def inference_with_probs(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=8):
    """
    Performs inference on a given text, generates a response, and calculates
    the probabilities for the class labels.
    """
    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    ).to(model.device)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_output_tokens,
            temperature=0.01,
            pad_token_id=tokenizer.pad_token_id
        )
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        predicted_label = generated_text[len(text):].strip().lower()

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        probs = softmax(logits, dim=-1)
    
    class_labels = ["promoter", "non-promoter"]
    # We take the first token ID as LlamaTokenizerFast usually doesn't add a space token at the start.
    # Verify this behavior if you encounter issues.
    class_tokens = [tokenizer.encode(label, add_special_tokens=False)[0] for label in class_labels]
    class_probs = probs[0, class_tokens].cpu().numpy()
    
    if predicted_label not in class_labels:
        print(f"Warning: Unexpected response '{predicted_label}', defaulting to 'non-promoter'. Input: {text[:80]}...")
        predicted_label = "non-promoter"
    
    return predicted_label, class_probs


# --- Plotting Functions ---

def plot_confusion_matrix(y_true, y_pred, class_labels, output_file="confusion_matrix_adamw_0614.png"):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Confusion matrix saved to {output_file}")


def plot_roc_curve(y_true, y_probs_positive, output_file="roc_curve_adamw_0614.png"):
    """
    Generates and saves an ROC curve plot.
    Assumes y_probs_positive contains probabilities for the positive class ('promoter').
    """
    # Convert string labels to binary format (1 for positive, 0 for negative)
    y_true_binary = [1 if label == 'promoter' else 0 for label in y_true]

    # Calculate ROC curve points
    fpr, tpr, _ = roc_curve(y_true_binary, y_probs_positive)
    roc_auc = auc(fpr, tpr)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"ROC curve saved to {output_file}")


# --- Main Testing and Evaluation Function ---

def test_multiple_samples(test_data, model, tokenizer):
    """
    Tests the model on all samples in test_data, calculates metrics,
    and generates evaluation plots.
    """
    data_list = []
    y_true = []
    y_pred = []
    y_probs_positive_class = []  # Store probabilities for the 'promoter' class
    class_labels = ["promoter", "non-promoter"]

    for entry in tqdm(test_data, desc="Testing Samples"):
        input_text = format_input(entry)
        real_answer = entry["output"].strip().lower()
        predicted_label, class_probs = inference_with_probs(input_text, model, tokenizer)
        
        y_true.append(real_answer)
        y_pred.append(predicted_label)
        # The first probability corresponds to the 'promoter' class
        y_probs_positive_class.append(class_probs[0])

        data = {
            "instruction": entry["instruction"],
            "input": entry["input"],
            "output": real_answer,
            "model_response": predicted_label,
            "class_probabilities": class_probs.tolist()
        }
        data_list.append(data)

    print("\n--- Evaluation Results ---")
    print("True label distribution:", Counter(y_true))
    print("Predicted label distribution:", Counter(y_pred))

    if not y_true:
        print("No samples were evaluated.")
        return

    # --- Calculate and Print Metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Calculate AUC-ROC score using binary labels and positive class probabilities
    y_true_binary = [1 if label == 'promoter' else 0 for label in y_true]
    auc_roc = roc_auc_score(y_true_binary, y_probs_positive_class)

    print(f"\nTotal test samples: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"AUC-ROC Score: {auc_roc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0, digits=4))

    # --- Generate Plots ---
    plot_confusion_matrix(y_true, y_pred, class_labels, output_file="confusion_matrix_adamw_0614.png")
    plot_roc_curve(y_true, y_probs_positive_class, output_file="roc_curve_adamw_0614.png")

    # --- Save Results to JSON ---
    output_file = "model_test_results_adamw_0614.json"
    output_data = {
        "metrics": {
            "accuracy": accuracy,
            "mcc": mcc,
            "auc_roc": auc_roc,
            "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        },
        "results": data_list
    }
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)
    print(f"Detailed results saved to {output_file}")


# --- Execution ---
if __name__ == "__main__":
    # Call the testing function with the full dataset.
    test_multiple_samples(test_data, model, tokenizer)

