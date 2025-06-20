import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

class Generator:
    def __init__(self):
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

        self.generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def generate(self, query_text, image_description, contexts):
        prompt = f"""Patient report:\n{query_text}

    Image Findings:\n{image_description}

    Similar Cases:\n"""
        for i, ctx in enumerate(contexts):
            prompt += f"Case {i+1}: {ctx['text']}\nDiagnosis: {ctx['label']}\n\n"

        prompt += "Based on the report, image, and similar cases, provide a likely diagnosis and recommend a treatment plan:\n"
        return self.generator(prompt, max_length=512)[0]["generated_text"]