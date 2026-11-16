import os
import uuid
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from google import genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pull keys securely from Render's Environment Variables
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# In-memory storage for chat sessions and latest lecture transcript
sessions_db = {}
latest_transcript = ""


class CreateSessionRequest(BaseModel):
    title: str


class ChatMessageRequest(BaseModel):
    session_id: str
    user_message: str


@app.get("/")
def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "Animus backend is running!"}


@app.get("/sessions")
def get_sessions():
    return sessions_db


@app.post("/sessions")
def create_session(req: CreateSessionRequest):
    session_id = str(uuid.uuid4())
    sessions_db[session_id] = {
        "title": req.title,
        "messages": [],
    }
    return {"session_id": session_id, "title": req.title}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id in sessions_db:
        del sessions_db[session_id]
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/chat")
async def chat_endpoint(req: ChatMessageRequest):
    if req.session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions_db[req.session_id]
    session["messages"].append({"text": req.user_message, "isUser": True, "isMarkdown": False})

    # Build conversation history with lecture context if available
    system_prompt = "You are Animus, a helpful AI study and lecture assistant."
    if latest_transcript:
        system_prompt += f"\n\nUse the following recorded lecture transcript as context when answering questions:\n{latest_transcript}"

    chat_history = [{"role": "system", "content": system_prompt}]
    for m in session["messages"]:
        role = "user" if m.get("isUser") else "assistant"
        chat_history.append({"role": role, "content": m["text"]})

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=chat_history,
            temperature=0.7,
        )
        reply = completion.choices[0].message.content
        session["messages"].append({"text": reply, "isUser": False, "isMarkdown": True})
        return {"response": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    global latest_transcript
    temp_path = None
    try:
        # Save uploaded audio temporarily for Groq Whisper
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_path = temp_audio.name

        # 1. Transcribe audio with Groq Whisper
        with open(temp_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=("lecture.webm", audio_file.read()),
                model="whisper-large-v3-turbo",
            )
        transcript_text = transcription.text.strip()
        latest_transcript = transcript_text

        if not transcript_text:
            return {
                "transcript": "(No speech detected)",
                "summary": "No speech was detected in the recording.",
            }

        # 2. Summarize transcript with Gemini
        summary_prompt = (
            "Summarize the following lecture transcript clearly using Markdown headings, "
            "bullet points, and a section for Key Takeaways:\n\n"
            f"{transcript_text}"
        )
        summary_response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=summary_prompt,
        )

        return {
            "transcript": transcript_text,
            "summary": summary_response.text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
