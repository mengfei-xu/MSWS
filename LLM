import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import re

MODEL_PATH = ""
DATA_PATH = ""
OUTPUT_PATH = ""

PROMPT_TEMPLATE = """According to the following sentiment elements definition:
− The 'aspect term' refers to a specific feature... IMPORTANT: The aspect term MUST be the exact substring extracted from the original text, or 'NULL'. Do NOT paraphrase or change words.
- The 'aspect category' refers to the category that aspect belongs to, and the available categories include: 'battery design_features', 'battery general', 'battery operation_performance', 'battery quality', 'company general', 'cpu operation_performance', 'cpu general', 'display design_features', 'display general', 'display operation_performance', 'display quality', 'fans&cooling operation_performance', 'graphics general', 'graphics operation_performance', 'hard_disc general', 'hard_disc operation_performance', 'hardware general', 'hardware operation_performance', 'hardware quality', 'keyboard design_features', 'keyboard general', 'keyboard operation_performance', 'keyboard usability', 'laptop connectivity', 'laptop design_features', 'laptop general', 'laptop operation_performance', 'laptop portability', 'laptop price', 'laptop quality', 'laptop usability', 'memory general', 'multimedia_devices general', 'multimedia_devices operation_performance', 'optical_drives general', 'os general', 'os operation_performance', 'os usability', 'ports general', 'ports operation_performance', 'ports quality', 'power_supply general', 'power_supply operation_performance', 'shipping general', 'software general', 'software operation_performance', 'support general', 'support quality', 'warranty general'
− The 'opinion term' refers to the sentiment... IMPORTANT: The opinion term MUST be the exact substring extracted from the original text, or 'NULL'. Do NOT paraphrase.
− The 'sentiment polarity' refers to the degree of positivity, negativity or neutrality expressed in the opinion towards a particular aspect or feature of a product or service, and the available polarities include: 'positive', 'negative' and 'neutral'.

Recognize all sentiment elements with their corresponding aspect terms, aspect categories, opinion terms and sentiment polarity in the following text with the format of [['aspect_term', 'aspect_category', 'sentiment_polarity', 'opinion_term'], ...]:

Text: {text}

Output (only the list, no explanation):"""


def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def read_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and '####' in line:
                text = line.split('####')[0].strip()
                data.append(text)
    return data


def extract_list_from_response(response):
    response = response.replace('```json', '').replace('```', '').strip()
    response = ' '.join(response.split())
    start_idx = response.find('[')
    end_idx = response.rfind(']')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        list_str = response[start_idx:end_idx + 1]
        list_str = list_str.replace('\n', '').replace('\r', '')
        try:
            parsed = json.loads(list_str)
            for item in parsed:
                if isinstance(item, list) and len(item) == 4:
                    for i in range(len(item)):
                        if isinstance(item[i], str) and item[i].lower() == 'null':
                            item[i] = 'NULL'
            list_str = json.dumps(parsed, ensure_ascii=False, separators=(',', ', '))

            return list_str
        except Exception as e:
            list_str = list_str.replace('\n', '').replace('\r', '')

            list_str = re.sub(r"'null'", "'NULL'", list_str, flags=re.IGNORECASE)
            list_str = re.sub(r'"null"', '"NULL"', list_str, flags=re.IGNORECASE)
            return list_str
    return "[]"


def generate_prediction(model, tokenizer, text, max_new_tokens=512):
    prompt = PROMPT_TEMPLATE.format(text=text)
    messages = [
        {"role": "user", "content": prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
        response = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    elif "assistant" in generated_text.lower():
        response = generated_text.split("assistant")[-1].strip()
    else:
        response = generated_text[len(formatted_prompt):].strip()

    response = response.replace("<|eot_id|>", "").strip()

    cleaned_response = extract_list_from_response(response)

    return cleaned_response


def main():
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    data_texts = read_data(DATA_PATH)
    results = []
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        for text in tqdm(data_texts, desc="Processing"):
            try:
                prediction = generate_prediction(model, tokenizer, text)
                output_line = f"{text}####{prediction}\n"
                f_out.write(output_line)
                f_out.flush()  

                results.append({
                    'text': text,
                    'prediction': prediction
                })
            except Exception as e:
                f_out.write(f"{text}####[]\n")
                f_out.flush()

if __name__ == "__main__":
    main()
