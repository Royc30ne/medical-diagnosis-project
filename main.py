from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.retriever import Retriever
from models.generator import Generator

import os
import huggingface_hub

from dotenv import load_dotenv
load_dotenv()

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
es_host = os.getenv("ES_HOST")
es_api_key = os.getenv("ES_API_KEY")
print("Hugging Face Token:", huggingface_token)
print("Elasticsearch Host:", es_host)
print("Elasticsearch API Key:", es_api_key)
huggingface_hub.login(token=huggingface_token)

def diagnose_case(text_path, image_path):
    with open(text_path) as f:
        patient_text = f.read()

    # Initialize models
    print("Initializing models...")
    text_encoder = TextEncoder()
    image_encoder = ImageEncoder()
    retriever = Retriever([es_host], es_api_key)
    generator = Generator()

    print("Models initialized.")
    # Encode text and image
    print("Encoding text and image...")
    text_emb = text_encoder.encode([patient_text])
    image_emb = image_encoder.encode(image_path)

    combined_query = "Symptom & history: " + patient_text
    similar_cases = retriever.search(combined_query)
    output = generator.generate(combined_query, similar_cases)

    print("Diagnosis Suggestion:\n", output)

diagnose_case("data/patients/patient123.txt", "data/images/patient123.jpg")
