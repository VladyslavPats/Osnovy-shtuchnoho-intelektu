import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Завантажуємо змінні середовища з .env
load_dotenv()

# Ініціалізація клієнта Gemini за допомогою нового SDK
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

# Налаштування CORS (важливо для зв'язку з локальним HTML файлом)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Схема вхідних даних для генерації
class TaskRequest(BaseModel):
    topic: str
    words: list[str]

@app.post("/api/generate-task")
async def generate_task(req: TaskRequest):
    words_list = ", ".join(req.words)
    
    # Промпт, що змушує ШІ використовувати саме наші слова зі словника
    prompt = f"""
    You are an English teacher. Write a short paragraph (2 sentences) about '{req.topic}'.
    CRITICAL RULE: You MUST strictly include these exact 3 words in the text: {words_list}.
    Do not change their form. Replace exactly these 3 words with '___' in the text.
    The "answers" array MUST be exactly this list in correct order: {json.dumps(req.words)}.
    Return ONLY a JSON object: {{"text": "...", "answers": [...]}}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        
        # Очищення відповіді від Markdown-розмітки
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").replace("json\n", "", 1).strip()
            
        return json.loads(raw_text)
        
    except Exception as e:
        print(f"ПОМИЛКА ГЕНЕРАЦІЇ: {e}")
        return {"text": "Server error. Please try again.", "answers": []}

@app.post("/api/check-answer")
async def check_answer(req: dict):
    # Промпт для розумної перевірки синонімів через ШІ
    prompt = f"Check if the word '{req.get('user_word', '')}' is a valid synonym or exactly matches '{req.get('correct_word', '')}' in English context. Return ONLY a JSON object: {{\"is_correct\": true or false}}"
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        
        raw_text = response.text.strip().strip("`").replace("json\n", "", 1).strip()
        return json.loads(raw_text)
        
    except Exception as e:
        print(f"ПОМИЛКА ПЕРЕВІРКИ: {e}")
        return {"is_correct": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)