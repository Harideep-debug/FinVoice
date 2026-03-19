import os
import json
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# ==============================
# 🔐 SECURE API KEY HANDLING
# ==============================
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")

if not GENAI_API_KEY:
    raise ValueError("❌ GENAI_API_KEY is missing. Check your .env file")

if not MURF_API_KEY:
    raise ValueError("❌ MURF_API_KEY is missing. Check your .env file")

genai.configure(api_key=GENAI_API_KEY)

MURF_URL = "https://global.api.murf.ai/v1/speech/stream"

# ==============================
# 🎙️ FAILSAFE VOICE SELECTOR
# ==============================
def get_safe_murf_voice(language, frontend_id):
    frontend_id_lower = frontend_id.lower()

    is_female = any(name in frontend_id_lower for name in [
        "aditi", "anya", "priya", "kavya", "lakshmi",
        "shruti", "divya", "dharini", "venus"
    ])

    guaranteed_voices = {
        "English": {"male": "en-IN-aarav", "female": "en-IN-anya"},
        "Hindi":   {"male": "hi-IN-rohan", "female": "hi-IN-kavya"},
        "Telugu":  {"male": "te-IN-ram",   "female": "te-IN-shruti"},
        "Tamil":   {"male": "ta-IN-surya", "female": "ta-IN-dharini"}
    }

    safe_lang = language if language in guaranteed_voices else "English"
    gender = "female" if is_female else "male"

    return guaranteed_voices[safe_lang][gender]

# ==============================
# 🧠 SYSTEM PROMPT
# ==============================
def get_system_prompt(language):
    return f"""
    You are FinVoice, an expert financial rights advisor for Indian citizens.
    You specialize in: Income Tax, GST, RBI consumer rights, EPF/ESIC rules, 
    insurance rights, banking rights, loan rights, credit score laws, 
    consumer protection act, and legal financial documents.

    RULES:
    - ALWAYS respond in the {language} language.
    - Use simple everyday words, avoid jargon. If necessary, explain it briefly.
    - Structure: 1) Simple answer 2) Why it matters 3) What user can do.
    - Keep response under 100 words for voice optimization.
    - End with one actionable tip.
    - Never give binding legal advice — say "consult a CA/lawyer" when unsure.
    
    OUTPUT FORMAT:
    Return ONLY valid JSON (no markdown):
    {{
        "category": "Tax" | "GST" | "Banking Rights" | "Investment Rights" | "Insurance Rights" | "Consumer Rights" | "General",
        "response": "Your answer"
    }}
    """

# ==============================
# ❤️ HEALTH CHECK
# ==============================
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "FinVoice backend is running"}), 200

# ==============================
# 🚀 MAIN API
# ==============================
@app.route('/ask-finvoice', methods=['POST'])
def ask_finvoice():
    try:
        data = request.json

        query = data.get('query', '').strip()
        if not query:
            query = "Please analyze this document and explain its financial or legal implications in simple terms."

        language = data.get('language', 'English')

        raw_voice_id = data.get('voiceId', 'en-IN-aarav')
        safe_murf_id = get_safe_murf_voice(language, raw_voice_id)

        speed = data.get('speed', 1.0)
        history = data.get('history', [])
        image_base64 = data.get('image', None)

        # Format history
        formatted_history = []
        for msg in history:
            formatted_history.append({
                "role": "user" if msg['role'] == 'user' else "model",
                "parts": [msg['content']]
            })

        current_parts = [query]

        if image_base64:
            mime_type, b64_data = image_base64.split(';base64,')
            mime_type = mime_type.split(':')[1]
            image_bytes = base64.b64decode(b64_data)
            current_parts.append({
                "mime_type": mime_type,
                "data": image_bytes
            })

        formatted_history.append({"role": "user", "parts": current_parts})

        # ==============================
        # 🤖 GEMINI CALL
        # ==============================
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=get_system_prompt(language)
        )

        chat = model.start_chat(history=formatted_history[:-1])

        response = chat.send_message(
            formatted_history[-1]["parts"],
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                response_mime_type="application/json"
            )
        )

        raw_json_text = response.text.replace("```json", "").replace("```", "").strip()

        try:
            ai_data = json.loads(raw_json_text)
            ai_text = ai_data.get("response", "I could not process that.")
            category = ai_data.get("category", "General")
        except Exception:
            ai_text = raw_json_text
            category = "General"

        # ==============================
        # 🔊 MURF TTS
        # ==============================
        audio_base64_out = None

        locale_map = {
            "English": "en-IN",
            "Hindi": "hi-IN",
            "Telugu": "te-IN",
            "Tamil": "ta-IN"
        }

        headers = {
            "Content-Type": "application/json",
            "api-key": MURF_API_KEY
        }

        payload = {
            "text": ai_text,
            "voiceId": safe_murf_id,
            "locale": locale_map.get(language, "en-IN"),
            "format": "MP3",
            "rate": speed
        }

        murf_res = requests.post(MURF_URL, json=payload, headers=headers)

        if murf_res.status_code == 200:
            audio_base64_out = base64.b64encode(murf_res.content).decode('utf-8')
        else:
            print(f"Murf API Error: {murf_res.text}")

        return jsonify({
            "answer_text": ai_text,
            "category": category,
            "audio_base64": audio_base64_out
        })

    except Exception as e:
        print(f"Server Error: {str(e)}")

        if 'history' in locals() and len(history) > 0:
            history.pop()

        return jsonify({"error": "An internal server error occurred."}), 500

# ==============================
# ▶️ RUN SERVER
# ==============================
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)