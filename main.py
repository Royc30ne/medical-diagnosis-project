import os
import tempfile
import shutil
import huggingface_hub

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.retriever import Retriever
from models.generator import Generator
from models.image_captioner import ImageCaptioner

load_dotenv()

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
es_host = os.getenv("ES_HOST")
es_api_key = os.getenv("ES_API_KEY")

print("Hugging Face Token:", huggingface_token)
print("Elasticsearch Host:", es_host)
print("Elasticsearch API Key:", es_api_key)
huggingface_hub.login(token=huggingface_token)

app = FastAPI()

# Load models once
text_encoder = TextEncoder()
image_encoder = ImageEncoder()
retriever = Retriever([es_host], es_api_key)
generator = Generator()
image_captioner = ImageCaptioner()

@app.post("/diagnose/")
async def diagnose(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        # Save uploaded image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(image.file, tmp)
            image_path = tmp.name

        # Generate image description
        image_desc = image_captioner.describe(image_path)

        # Retrieve similar cases using prompt + image_desc
        query = f"{prompt}\n\nImage: {image_desc}"
        similar_cases = retriever.search(query)

        # Generate diagnostic hypothesis
        response = generator.generate(prompt, image_desc, similar_cases)

        return JSONResponse(content={"diagnosis": response.strip()}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)