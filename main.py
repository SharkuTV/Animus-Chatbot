import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Groq API Key
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# In-memory storage for chat sessions
sessions = {}

class SessionCreate(BaseModel):
    title: str

class ChatMessage(BaseModel):
    session_id: str
    user_message: str

@app.get("/")
def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Animus API is running! Use /docs to view endpoints."}

@app.get("/sessions")
def get_sessions():
    return sessions

@app.post("/sessions")
def create_session(data: SessionCreate):
    import uuid
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "title": data.title,
        "messages": []
    }
    return {"session_id": session_id, "title": data.title}

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    temp_filename = "temp_lecture.webm"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    file_size = os.path.getsize(temp_filename)
    print(f"Received audio file, Size: {file_size} bytes")

    if file_size < 100:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise HTTPException(status_code=400, detail="Audio recording too short or empty.")

    try:
        # 1. Transcribe audio using Groq Whisper API
        with open(temp_filename, "rb") as audio_file:
            transcription_obj = client.audio.transcriptions.create(
                file=(temp_filename, audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="text"
            )
        
        transcript_text = transcription_obj if isinstance(transcription_obj, str) else getattr(transcription_obj, "text", str(transcription_obj))
        print("Transcription successful!")

        # 2. Summarize the transcript using GPT-OSS 20B on Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert AI assistant that summarizes lecture transcripts into clear, structured Markdown notes with key takeaways."
                },
                {
                    "role": "user",
                    "content": f"Please summarize the following transcript into clear key takeaways and bullet points:\n\n{transcript_text}"
                }
            ],
            model="openai/gpt-oss-20b",
        )
        summary_text = chat_completion.choices[0].message.content

        return {
            "transcript": transcript_text,
            "summary": summary_text
        }

    except Exception as e:
        print(f"Error processing audio with Groq: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/chat")
def chat_with_session(data: ChatMessage):
    if data.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[data.session_id]
    session["messages"].append({"text": data.user_message, "isUser": True, "isMarkdown": False})

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are Animus, a helpful AI assistant."},
                {"role": "user", "content": data.user_message}
            ],
            model="openai/gpt-oss-20b",
        )
        ai_reply = chat_completion.choices[0].message.content
    except Exception as e:
        ai_reply = f"Error generating response: {str(e)}"

    session["messages"].append({"text": ai_reply, "isUser": False, "isMarkdown": True})
    return {"response": ai_reply}
