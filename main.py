import os
import uuid
import tempfile
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
users_db = {}

def get_user_store(user_id: str):
    if user_id not in users_db:
        users_db[user_id] = {"sessions": {}}
    return users_db[user_id]

def ensure_session(store: dict, session_id: str, title: str = "Study Session 1"):
    if session_id not in store["sessions"]:
        store["sessions"][session_id] = {
            "title": title,
            "messages": [],
            "transcript": "",
        }
    return store["sessions"][session_id]

class CreateSessionRequest(BaseModel):
    user_id: str
    title: str

class ChatMessageRequest(BaseModel):
    user_id: str
    session_id: str
    user_message: str
    transcript_context: Optional[str] = None
    history: Optional[list] = None

class AutoTitleRequest(BaseModel):
    user_id: str
    session_id: str
    text_content: str

@app.get("/")
def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "Animus backend is running!"}

@app.get("/sessions")
def get_sessions(user_id: str):
    store = get_user_store(user_id)
    return store["sessions"]

@app.post("/sessions")
def create_session(req: CreateSessionRequest):
    store = get_user_store(req.user_id)
    session_id = str(uuid.uuid4())
    store["sessions"][session_id] = {
        "title": req.title,
        "messages": [],
        "transcript": "",
    }
    return {"session_id": session_id, "title": req.title}

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, user_id: str):
    store = get_user_store(user_id)
    if session_id in store["sessions"]:
        del store["sessions"][session_id]
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.post("/auto-title")
async def auto_title(req: AutoTitleRequest):
    store = get_user_store(req.user_id)
    session = ensure_session(store, req.session_id)
    
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "Generate a short, punchy 3 to 5 word title summarizing the main topic of the following text. Return ONLY the title text, with no quotes or extra formatting."
                },
                {"role": "user", "content": req.text_content[:2000]}
            ],
            temperature=0.5,
        )
        new_title = completion.choices[0].message.content.strip().replace('"', '')
        if new_title:
            session["title"] = new_title
        return {"title": session["title"]}
    except Exception as e:
        return {"title": session["title"]}

@app.post("/chat")
async def chat_endpoint(req: ChatMessageRequest):
    store = get_user_store(req.user_id)
    session = ensure_session(store, req.session_id)

    if req.transcript_context is not None:
        session["transcript"] = req.transcript_context

    if req.history:
        session["messages"] = req.history
    else:
        session["messages"].append({"text": req.user_message, "isUser": True, "isMarkdown": False})

    system_prompt = (
        "You are Animus, an encouraging, clear, and interactive AI study and lecture assistant. "
        "When the user asks to be quizzed, you MUST generate exactly 1 question based on the lecture transcript. "
        "You must format your response strictly as a JSON object (without markdown code blocks) in this exact format: "
        "{\"type\": \"mcq\", \"question\": \"Question text here?\", \"options\": [\"A) ...\", \"B) ...\", \"C) ...\"], \"correct\": \"A\"} "
        "OR for short answer: "
        "{\"type\": \"short\", \"question\": \"Question text here?\"} "
        "Do not include any extra conversational text when generating a quiz question. "
        "For regular conversation, answer normally in markdown."
    )
    if session.get("transcript"):
        system_prompt += (
            "\n\nUse the following recorded lecture transcript from this chat session as context "
            f"when answering questions or quizzing the student:\n{session['transcript']}"
        )

    chat_history = [{"role": "system", "content": system_prompt}]
    for m in session["messages"]:
        role = "user" if m.get("isUser") else "assistant"
        chat_history.append({"role": role, "content": m["text"]})

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=chat_history,
            temperature=0.7,
        )
        reply = completion.choices[0].message.content
        session["messages"].append({"text": reply, "isUser": False, "isMarkdown": True})
        return {"response": reply}
    except Exception as e:
        return {"response": f"⚠️ Server Error: {str(e)}"}

@app.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    session_id: str = Form(...),
    bookmarks: str = Form(""),
):
    store = get_user_store(user_id)
    session = ensure_session(store, session_id)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_path = temp_audio.name

        with open(temp_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=("lecture.webm", audio_file.read()),
                model="whisper-large-v3-turbo",
            )
        transcript_text = transcription.text.strip()
        
        if transcript_text:
            if session["transcript"]:
                session["transcript"] += "\n\n" + transcript_text
            else:
                session["transcript"] = transcript_text

        if not transcript_text:
            return {
                "transcript": "(No speech detected)",
                "summary": "No speech was detected in this recording segment.",
            }

        bookmark_instruction = ""
        if bookmarks.strip():
            bookmark_instruction = (
                f"\n\nIMPORTANT: During the lecture, the student flagged these timestamps as confusing: {bookmarks}. "
                "Include a dedicated '🚩 Flagged Moments Clarified' section that breaks down the concepts discussed "
                "around those points in simple, easy-to-understand language."
            )

        summary_prompt = (
            "Summarize the following lecture transcript clearly using Markdown headings, "
            "bullet points, and a section for Key Takeaways."
            f"{bookmark_instruction}\n\nTranscript:\n{transcript_text}"
        )
        summary_completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.5,
        )

        return {
            "transcript": transcript_text,
            "summary": summary_completion.choices[0].message.content,
        }
    except Exception as e:
        print(f"UPLOAD AUDIO ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
