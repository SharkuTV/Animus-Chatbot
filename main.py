import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from google import genai

app = FastAPI()

# Allow your GitHub Pages frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pull keys securely from Render's Environment Variables
groq_client = Groq(api_key=os.environ.get("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"))
gemini_client = genai.Client(api_key=os.environ.get("AQ.Ab8RN6LcOtIiF24ja9n3m7CO0giiV9fBSDH1U3zT4A9Dddb7Bg"))

class ChatRequest(BaseModel):
    prompt: str
    provider: str = "groq"  # Defaults to Groq; can also pass "gemini"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        if request.provider == "gemini":
            response = gemini_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=request.prompt,
            )
            return {"response": response.text}
        else:
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": request.prompt}],
                temperature=0.7,
            )
            return {"response": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "Animus backend is running!"}
