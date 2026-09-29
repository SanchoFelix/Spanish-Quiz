import os
import json
from pypdf import PdfReader
from google import genai
from google.genai import types

LEVEL_DIR = "Coursework/Level 1"
OUTPUT_FILE = "Data/level-1-quiz.json"

# Initialize Gemini Client (automatically reads GEMINI_API_KEY environment variable)
client = genai.Client()

def extract_text_from_pdfs():
    combined_text = ""
    if not os.path.exists(LEVEL_DIR):
        print(f"Directory '{LEVEL_DIR}' not found.")
        return combined_text

    files = [f for f in os.listdir(LEVEL_DIR) if f.endswith('.pdf')]
    files.sort()

    for file_name in files:
        pdf_path = os.path.join(LEVEL_DIR, file_name)
        print(f"Reading {file_name}...")
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                combined_text += text + "\n"

    return combined_text

def generate_quiz_json(transcript_text):
    prompt = f"""
    You are a Spanish language instructor. Analyze the following Level 1 Spanish lesson transcripts and generate a 30-question multiple-choice quiz testing vocabulary, grammar, and translations covered in these transcripts.

    Return ONLY a raw JSON array of objects without any markdown wrappers or text preceding/following it.
    
    Each object must follow this exact format:
    [
        {{
            "id": 1,
            "type": "mcq",
            "question": "How do you say 'hello' in Spanish?",
            "options": ["A) Hola", "B) Adiós", "C) Gracias", "D) Por favor"],
            "answer": "A) Hola"
        }}
    ]

    Transcripts content:
    {transcript_text[:20000]}
    """

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3,
        ),
    )

    content = response.text.strip()
    return json.loads(content)

def main():
    print("Extracting text from Level 1 PDFs...")
    raw_text = extract_text_from_pdfs()

    if not raw_text:
        print("No text extracted from PDFs.")
        return

    print("Generating Level 1 quiz via Gemini...")
    quiz_data = generate_quiz_json(raw_text)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(quiz_data, f, indent=4, ensure_ascii=False)

    print(f"Successfully generated {len(quiz_data)} questions in '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()
