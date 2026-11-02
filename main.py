"""
LoRA fine-tuning and inference for an open-weight LLM (Meta Llama 3.2 1B
Instruct) on an instruction-following dataset, using CUDA.

Usage:
    python main.py train
    python main.py infer "Explain what a database index is in one sentence."
    python main.py chat
"""

import sys
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_PATH = "data/sample_dataset.jsonl"
OUTPUT_DIR = "output/llama-3.2-1b-lora"
MAX_LENGTH = 512
device = torch.device("cuda")


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found. This script requires an NVIDIA GPU with "
            "CUDA to run. Install CUDA-enabled PyTorch and run on a "
            "machine with an NVIDIA GPU."
        )


def format_example(tokenizer, example: dict) -> str:
    """Turn one instruction/output record into Llama 3.2's chat format."""
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_prompt(tokenizer, instruction: str) -> str:
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )


def train():
    require_cuda()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    def tokenize(example):
        text = format_example(tokenizer, example)
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            add_special_tokens=False,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to {OUTPUT_DIR}")


def infer(instruction: str):
    require_cuda()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt = build_prompt(tokenizer, instruction)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    print("=== Base model ===")
    print(generate(base_model, tokenizer, prompt))

    print("\n=== Fine-tuned model ===")
    finetuned_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    print(generate(finetuned_model, tokenizer, prompt))


def chat():
    """Interactive chat loop using the fine-tuned model only."""
    require_cuda()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

    print("Chatting with the fine-tuned model. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            instruction = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not instruction:
            continue
        if instruction.lower() in ("quit", "exit"):
            break

        prompt = build_prompt(tokenizer, instruction)
        answer = generate(model, tokenizer, prompt)
        print(f"Assistant: {answer}\n")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "infer", "chat"):
        print("Usage:")
        print("  python main.py train")
        print('  python main.py infer "Your instruction here"')
        print("  python main.py chat")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        train()
    elif command == "infer":
        if len(sys.argv) < 3:
            print('Usage: python main.py infer "Your instruction here"')
            sys.exit(1)
        infer(sys.argv[2])
    elif command == "chat":
        chat()


if __name__ == "__main__":
    main()
