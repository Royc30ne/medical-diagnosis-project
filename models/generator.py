from transformers import pipeline

class Generator:
    def __init__(self):
        self.generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")

    def generate(self, query, contexts):
        prompt = "Patient report:\n" + query + "\n\nSimilar Cases:\n"
        for i, ctx in enumerate(contexts):
            prompt += f"Case {i+1}: {ctx['text']}\nDiagnosis: {ctx['label']}\n\n"
        prompt += "Based on the report and similar cases, provide a likely diagnosis:\n"
        return self.generator(prompt, max_length=512)[0]["generated_text"]