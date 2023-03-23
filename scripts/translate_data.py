import json
import ctranslate2
import transformers
from tqdm.auto import tqdm

with open('alpaca_data.json', 'r') as f:
    data = json.load(f)

src_lang = "eng_Latn"
tgt_lang = "ind_Latn"

translator = ctranslate2.Translator("nllb-200-distilled-600M", device="cuda", compute_type="float16")
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=src_lang)

def translate(text):
    source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    target_prefix = [tgt_lang]
    results = translator.translate_batch([source], target_prefix=[target_prefix])
    target = results[0].hypotheses[0][1:]

    return tokenizer.decode(tokenizer.convert_tokens_to_ids(target))

def translate_item(item):
    translated_item = {}
    for key, value in item.items():
        if value:
            translated_value = translate(value)
            translated_item[key] = translated_value
        else:
            translated_item[key] = ''
    return translated_item

translated_data = [translate_item(datum) for datum in tqdm(data)]

with open(f'domba-dataset-52k.json', 'w') as f:
    json.dump(translated_data, f, ensure_ascii=False, indent=4)