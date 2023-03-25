from dataclasses import dataclass
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model


@dataclass
class args:
    micro_batch_size = 128  # this could actually be 5 but i like powers of 2
    batch_size = 128
    gradient_accumulation_steps = batch_size // micro_batch_size
    epochs = 3  # we don't need 3 tbh
    learning_rate = 3e-4  # the Karpathy constant
    cutoff_len = 256  # 256 accounts for about 96% of the data
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05


def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Berikut adalah instruksi yang menggambarkan suatu tugas, bersama dengan input yang memberikan konteks. Tuliskan sebuah jawaban yang melengkapi permintaan dengan tepat.
### Instruksi:
{data_point["instruction"]}
### Konteks:
{data_point["input"]}
### Jawaban:
{data_point["output"]}"""
    else:
        return f"""Berikut adalah instruksi yang menggambarkan suatu tugas. Tuliskan sebuah jawaban yang melengkapi permintaan dengan tepat.
### Instruksi:
{data_point["instruction"]}
### Jawaban:
{data_point["output"]}"""


def main():
    tokenizer = LlamaTokenizer.from_pretrained(
        "decapoda-research/llama-7b-hf", add_eos_token=True
    )

    def tokenize(prompt):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len + 1,
            padding="max_length",
        )
        return {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
        }

    model = LlamaForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf",
        load_in_8bit=True,
        device_map="auto",
    )

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    data = load_dataset("json", data_files="domba-dataset-52k.json")
    data = data.shuffle().map(lambda x: tokenize(generate_prompt(x)))

    training_args = TrainingArguments(
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=20,
        output_dir="./domba-lora",
        save_total_limit=3
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        train_dataset=data["train"],
        args=training_args,
        data_collator=data_collator,
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained("domba-lora")


if __name__ == "__main__":
    main()

