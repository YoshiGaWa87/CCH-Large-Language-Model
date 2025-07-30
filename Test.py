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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, confusion_matrix, matthews_corrcoef
from collections import Counter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset directly. We will use all of it for testing.
dna_ft_dataset = load_dataset('json', data_files='/home/u5849124/dna-llama/promoter_data/test.json')
# The 'train' split is the default name given by the load_dataset function when loading a single file.
test_data = dna_ft_dataset["train"] 

# If you still need a smaller subset for quick tests, you can uncomment the line below.
# test_data = dna_ft_dataset["train"].train_test_split(train_size=0.1, seed=42)["train"]

tokenizer = LlamaTokenizerFast.from_pretrained("LlaMaDNAPromoter_RK4")
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    "LlaMaDNAPromoter_RK4",
    device_map="auto"
)

def format_input(entry):
    """Formats the input entry into a standardized instruction prompt."""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text + "\n\n### Response:\n"

def build_prompt(entry):
    """Builds the full prompt including the desired response for context."""
    input_data = format_input(entry)
    desired_response = entry['output']
    return input_data + desired_response

def inference_with_probs(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=8):
    """
    Performs inference on a given text, generates a response, and calculates
    the probabilities for the class labels.
    """
    # Tokenize the input text and move it to the model's device
    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    ).to(model.device)
    
    # Generate tokens with no gradient calculation for efficiency
    with torch.no_grad():
        generated_tokens = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_output_tokens,
            temperature=0.01,
            pad_token_id=tokenizer.pad_token_id
        )
        # Decode the generated tokens to text
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        # Extract only the newly generated part of the text as the prediction
        predicted_label = generated_text[len(text):].strip().lower()

    # Get model outputs to calculate probabilities
    with torch.no_grad():
        outputs = model(input_ids)
        # Get logits for the last token
        logits = outputs.logits[:, -1, :]
        # Apply softmax to get probabilities
        probs = softmax(logits, dim=-1)
    
    class_labels = ["promoter", "non-promoter"]
    # Get the token IDs for our specific class labels
    # We take the second token [1] as the first is often a start-of-sequence token
    class_tokens = [tokenizer.encode(label)[1] for label in class_labels]
    # Get the probabilities corresponding to our class tokens
    class_probs = probs[0, class_tokens].cpu().numpy()
    
    # Handle cases where the model generates an unexpected response
    if predicted_label not in class_labels:
        print(f"Warning: Unexpected response '{predicted_label}', defaulting to 'non-promoter'. Input: {text[:50]}...")
        predicted_label = "non-promoter"
    
    return predicted_label, class_probs

def plot_confusion_matrix(y_true, y_pred, class_labels, output_file="confusion_matrix_rk4_v3.png"):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Confusion matrix saved to {output_file}")

def test_multiple_samples(test_data, model, tokenizer):
    """Tests the model on multiple samples and calculates evaluation metrics."""
    data_list = []
    
    y_true = []
    y_pred = []
    y_probs = []
    class_labels = ["promoter", "non-promoter"]

    # Iterate through all samples in the provided test_data
    for entry in tqdm(test_data, desc="Testing Samples"):
        input_text = format_input(entry)
        real_answer = entry["output"].strip().lower()
        predicted_label, class_probs = inference_with_probs(input_text, model, tokenizer)
        
        y_true.append(real_answer)
        y_pred.append(predicted_label)
        # We use the probability of the "promoter" class for AUC-ROC calculation
        y_probs.append(class_probs[0]) 

        data = {
            "instruction": entry["instruction"],
            "input": entry["input"],
            "output": real_answer,
            "model_response": predicted_label,
            "class_probabilities": class_probs.tolist()
        }
        data_list.append(data)

    print("True label distribution:", Counter(y_true))
    print("Predicted label distribution:", Counter(y_pred))

    if len(y_true) == 0:
        print("No samples to evaluate.")
        return 0, 0, 0, 0, 0, 0, data_list

    # --- Calculate and Print Metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Total test samples: {len(y_true)}")
    print(f"Accuracy: {accuracy:.2%}")

    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"Precision (Macro avg): {precision_macro:.2%}")
    print(f"Recall (Macro avg): {recall_macro:.2%}")
    print(f"F1-Score (Macro avg): {f1_macro:.2%}")
    print(f"Precision (Micro avg): {precision_micro:.2%}")
    print(f"Precision (Weighted avg): {precision_weighted:.2%}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    try:
        # Map labels to 0 and 1 for AUC calculation
        y_true_binary = [1 if label == 'promoter' else 0 for label in y_true]
        auc_roc = roc_auc_score(y_true_binary, y_probs)
        print(f"AUC-ROC: {auc_roc:.2%}")
    except ValueError as e:
        print(f"Could not calculate AUC-ROC: {e}")
        auc_roc = 0

    try:
        mcc = matthews_corrcoef(y_true, y_pred)
        print(f"MCC (Matthews Correlation Coefficient): {mcc:.2%}")
    except Exception as e:
        print(f"Could not calculate MCC: {e}")
        mcc = 0

    plot_confusion_matrix(y_true, y_pred, class_labels)

    # --- Save Results to JSON ---
    output_file = "llama3-8b-sft-test-results_with_metrics_rk4_v3.json"
    output_data = {
        "results": data_list,
        "metrics": {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_micro": precision_micro,
            "precision_weighted": precision_weighted,
            "auc_roc": auc_roc,
            "mcc": mcc
        }
    }
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)
    print(f"Results saved to {output_file}")

    return accuracy, precision_macro, recall_macro, f1_macro, auc_roc, mcc, data_list

# Call the testing function with the full dataset.
# The `num_samples` parameter is no longer needed.
accuracy, precision, recall, f1, auc_roc, mcc, results = test_multiple_samples(test_data, model, tokenizer)

