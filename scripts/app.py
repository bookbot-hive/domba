import gradio as gr
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


def generate_prompt(instruction, input=None):
    if input:
        return f"""Berikut adalah instruksi yang menggambarkan suatu tugas, bersama dengan input yang memberikan konteks. Tuliskan sebuah jawaban yang melengkapi permintaan dengan tepat.

### Instruksi:
{instruction}

### Konteks:
{input}

### Jawaban:"""
    else:
        return f"""Berikut adalah instruksi yang menggambarkan suatu tugas. Tuliskan sebuah jawaban yang melengkapi permintaan dengan tepat.

### Instruksi:
{instruction}

### Jawaban:"""


tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "bookbot/domba-lora-v0-1")

generation_config = GenerationConfig(
    temperature=0.2,
    top_p=0.75,
    num_beams=4,
)


def evaluate(instruction, input=None):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256,
    )
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        response = output.split("### Jawaban:")[1].strip()
        print("Jawaban:", response)
        return response


demo = gr.Interface(
    fn=evaluate,
    inputs=gr.Textbox(lines=5, label="Instruksi"),
    outputs=gr.Textbox(lines=5, label="Jawaban"),
    title="🐑 Domba LoRA - Indonesian LLaMA",
    description="Meta AI's LLaMA Conversational Model Fine-tuned on Alpaca Data translated to Indonesian.",
    examples=["Bagaimana saya bisa memulai karir sebagai ilmuwan data? Tulis dalam bentuk daftar.", "Kenapa kita harus sikat gigi dua kali sehari?"],
    allow_flagging="never",
)
demo.launch(share=False, server_name="0.0.0.0")
