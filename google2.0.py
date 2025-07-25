import pathlib
from typing import Optional
import os
import requests
import io
import json
import re
import PIL.Image
from dotenv import load_dotenv
from IPython.display import display, Markdown
import google.generativeai as genai
from Config import AppConfig
import RAG2
import logging

logger = logging.getLogger("GoogleGemini")

load_dotenv()

rag = RAG2.RAG()

# Configure Gemini
genai.configure(api_key=AppConfig.GEMINI_API_KEY)

system_prompt = """
You are a highly intelligent medical AI assistant. Your task is to analyze the provided medical information (image and/or text), also the data fetched from the database that is being fed to you and provide a structured response in JSON format.

Your response MUST include the following keys:
- "diagnosis": A potential diagnosis based on the provided information. If no image is provided, state that the diagnosis is based on text only. If the information is insufficient, state that.
- "Medicine" : Tell if the exisiting medicine that is being taken is enough based on the inputs from the table or any better medicine must be taken. Be crisp, and do not beat around the bush
- "doctor_recommendation": The type of specialist or doctor to consult (e.g., "Cardiologist", "Dermatologist", "General Practitioner").
- "recovery_estimation": An estimated time for recovery. It must be to the dot.

Only respond with a single JSON object. No markdown, no text outside JSON.
"""

model = genai.GenerativeModel(
    'gemini-2.5-flash',
    system_instruction=system_prompt,
    generation_config=genai.GenerationConfig(response_mime_type="application/json")
)

def display_analysis(analysis: dict):
    print("\n" + "="*40)
    print("--- Medical AI Analysis Report ---")
    print("="*40)
    print(f"\n[+] Diagnosis: {analysis.get('diagnosis', 'N/A')}")
    print(f"[+] Medicine: {analysis.get('Medicine', 'N/A')}")
    print(f"[+] Recommended Specialist: {analysis.get('doctor_recommendation', 'N/A')}")
    print(f"[+] Estimated Recovery: {analysis.get('recovery_estimation', 'N/A')}")
    print("="*40)

def extract_json(text: str) -> Optional[dict]:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemini response: {e}")
    return None

if __name__ == '__main__':
    try:
        image_source = None
        img = None
        patient_id = input("Enter your patient ID: ").strip()
        full_name=input("Enter you full name please: ")
        if not patient_id.isalnum():
            raise ValueError("Patient ID must be alphanumeric.")

        use_image = input("Do you have an image to add? (yes/no): ").strip().lower()
        if use_image == 'yes':
            source_type = input("Is the image a local file or a URL? (file/url): ").strip().lower()
            if source_type == 'file':
                path = input("Enter path to your local image: ").strip()
                if os.path.exists(path):
                    image_source = path
                else:
                    logger.warning(f"Image file not found at {path}.")
            elif source_type == 'url':
                image_source = input("Enter image URL: ").strip()
            else:
                logger.warning("Invalid source type. Skipping image.")

        prompt_text = input("Please describe your medical concern:\n> ").strip()
        contents = [prompt_text]

        if image_source:
            logger.info("Attempting to load image...")
            try:
                if image_source.startswith(("http://", "https://")):
                    response = requests.get(image_source)
                    response.raise_for_status()
                    content_type = response.headers.get('Content-Type', '')
                    if 'image' not in content_type:
                        raise ValueError(f"URL does not return image. Content-Type: {content_type}")
                    img = PIL.Image.open(io.BytesIO(response.content))
                else:
                    img = PIL.Image.open(image_source)
                contents.append(img)
                logger.info("Image loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load image: {e}")

        print("Fetching patient records from database...")
        context_parts = []
        db_context = rag.fetch_patient_data(patient_id=patient_id or None, full_name=full_name)
        if db_context:
            context_parts.append(db_context)
        else:
            logger.warning("Loading...")

        use_doc = input("Do you want to upload a prescription document? (yes/no): ").strip().lower()
        if use_doc == 'yes':
            doc_link = input("Enter path or URL to the document: ").strip()
            try:
                doc_texts = rag.document_loader(doc_link)
                if doc_texts:
                    doc_text = "\n".join([doc.page_content for doc in doc_texts])
                    context_parts.append("--- Context from Uploaded Document ---\n" + doc_text)
                    logger.info("Document loaded and processed.")
                else:
                    logger.warning("No text extracted from document.")
            except Exception as e:
                logger.error(f"Document load failed: {e}")

        if context_parts:
            contents.append("\n\n".join(context_parts))

        logger.info("Generating Gemini response...")
        response = model.generate_content(contents)
        analysis = extract_json(response.text)

        if analysis:
            display_analysis(analysis)
        else:
            logger.error("Gemini returned invalid or no JSON.")
            print(response.text)

    except Exception as e:
        logger.exception(f"Fatal error occurred: {e}")
