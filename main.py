import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

# Enable CORS so your GitHub Pages frontend can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your GitHub Pages URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client (pulls GROQ_API_KEY from environment variables)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Updated active Groq model
            messages=[
                {"role": "user", "content": request.prompt}
            ],
            temperature=0.7,
        )
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "Animus backend is running!"}
