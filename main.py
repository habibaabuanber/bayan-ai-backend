# # from fastapi import FastAPI

# # app = FastAPI()

# # @app.get("/")
# # def root():
# #     return {"message": "Backend is working ✅"}
# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List, Dict, Any
# import numpy as np
# import json
# from pathlib import Path
# from sklearn.metrics.pairwise import cosine_similarity
# from langdetect import detect
# from openai import OpenAI
# import requests

# # ------------------- CONFIG -------------------

#  # حطي مفتاحك هنا
# BOOKS_FILE = "books_dataset_enriched.jsonl"
# EMB_FILE = "books_embeddings.npy"
# META_FILE = "books_metadata.json"
# EMBED_MODEL = "text-embedding-3-small"
# TOP_K = 4

# # ------------------- LOAD DATA -------------------
# def load_books(path: str):
#     return [json.loads(line) for line in open(path, "r", encoding="utf-8")]

# def load_embeddings():
#     embeddings = np.load(EMB_FILE)
#     with open(META_FILE, "r", encoding="utf-8") as f:
#         metas = json.load(f)
#     return embeddings, metas

# books_raw = load_books(BOOKS_FILE)[:200]
# embeddings, metas = load_embeddings()

# # ------------------- SEARCH FUNCTION -------------------
# def embed_text(text: str):
#     resp = client.embeddings.create(model=EMBED_MODEL, input=text)
#     return resp.data[0].embedding

# def find_top_k(query: str, k: int = TOP_K):
#     query_emb = np.array(embed_text(query), dtype=np.float32).reshape(1, -1)
#     sims = cosine_similarity(query_emb, embeddings)[0]
#     idx = np.argsort(sims)[::-1][:k]
#     results = []
#     for i in idx:
#         m = metas[int(i)].copy()
#         m["_score"] = float(sims[int(i)])
#         results.append(m)
#     return results

# # ------------------- FASTAPI APP -------------------
# app = FastAPI()
# conversation_history: List[Dict] = []
# class ChatRequest(BaseModel):
#     message: str
#     # history: List[Dict[str, str]] = []  ✅ نشيل ده

# @app.get("/")
# def root():
#     return {"message": "Backend is working ✅"}

# @app.post("/chat")
# def chat(req: ChatRequest):
#     user_text = req.message

#     # 1) خزّن رسالة اليوزر في الـ history
#     conversation_history.append({"role": "user", "content": user_text})

#     # Auto detect language
#     try:
#         lang = "ar" if detect(user_text) == "ar" else "en"
#     except:
#         lang = "en"

#     # نحدد هل نرشح ولا لسه هنسأل
#     trigger_terms = ["recommend", "suggest", "surprise", "اقترح", "رشح", "نصيحة"]
#     need_recommend = any(t in user_text.lower() for t in trigger_terms) or len(conversation_history) >= 3

#     if need_recommend:
#         # Build full preference string من الـ history كله
#         full_query = " ; ".join([h["content"] for h in conversation_history if h["role"] == "user"])

#         best = find_top_k(full_query, k=TOP_K)

#         # حضّري وصف الكتب
#         books_block = ""
#         for b in best:
#             books_block += f"Title: {b['title']}\nAuthor: {b.get('authors','')}\nSummary: {b.get('summary','')}\n\n"

#         prompt = f"""
# You are a friendly librarian. The user likes: {full_query}.
# Explain in {lang} very briefly in just two or three sentences why these books would match them:
# {books_block}
# """

#         resp = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}]
#         )

#         reply = resp.choices[0].message.content

#         # 2) خزّن رد البوت
#         conversation_history.append({"role": "assistant", "content": reply})

#         return {"reply": reply, "books": best}

#     else:
#         # لسه بدري — نسأل سؤال متتابع
#         history_text = "\n".join([f"{h['role']}: {h['content']}" for h in conversation_history])
#         print("HISTORY: ",history_text)
#         prompt = f"""
# You are a curious librarian. Ask ONE short question that follows logically based on the user's last answer, to recommend a book later.
# Conversation so far:
# {history_text}
# Respond in {lang}.
# """

#         resp = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}]
#         )

#         reply = resp.choices[0].message.content

#         # خزّن رد البوت
#         conversation_history.append({"role": "assistant", "content": reply})

#         return {"reply": reply, "books": []}
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from openai import OpenAI
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import uuid
import time



# ------------------- CONFIG -------------------

import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY", "")  # optional

client = OpenAI(api_key=OPENAI_API_KEY)

BOOKS_FILE = "books_dataset_enriched.jsonl"
EMB_FILE = "books_embeddings.npy"
META_FILE = "books_metadata.json"
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 4

# ------------------- LOAD DATA -------------------
def load_books(path: str):
    return [json.loads(line) for line in open(path, "r", encoding="utf-8")]

def load_embeddings():
    embeddings = np.load(EMB_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        metas = json.load(f)
    return embeddings, metas

books_raw = load_books(BOOKS_FILE)[:200]
embeddings, metas = load_embeddings()

# ------------------- HELPERS -------------------
def embed_text(text: str):
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def find_top_k(query: str, k: int = TOP_K):
    query_emb = np.array(embed_text(query), dtype=np.float32).reshape(1, -1)
    sims = cosine_similarity(query_emb, embeddings)[0]
    idx = np.argsort(sims)[::-1][:k]
    results = []
    for i in idx:
        m = metas[int(i)].copy()
        m["_score"] = float(sims[int(i)])
        results.append(m)
    return results
# def find_top_k(query: str, user_lang: str = None, k: int = TOP_K):
#     query_emb = np.array(embed_text(query), dtype=np.float32).reshape(1, -1)
#     sims = cosine_similarity(query_emb, embeddings)[0]
#     idx = np.argsort(sims)[::-1]
#     results = []
#     for i in idx:
#         m = metas[int(i)].copy()
#         if user_lang and m.get("language") != user_lang:
#             continue
#         m["_score"] = float(sims[int(i)])
#         results.append(m)
#         if len(results) >= k:
#             break
#     return results

def get_cover_google(isbn: str):
    if not isbn:
        return None
    try:
        url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
        r = requests.get(url, timeout=6)
        data = r.json()
        items = data.get("items")
        if items:
            vol = items[0].get("volumeInfo", {})
            imgs = vol.get("imageLinks", {})
            for k in ("extraLarge", "large", "medium", "small", "thumbnail"):
                if imgs.get(k):
                    return imgs.get(k)
        return None
    except:
        return None

def get_cover_openlibrary(isbn: str):
    if not isbn:
        return None
    return f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"

def ensure_cover(book: Dict[str, Any]):
    if book.get("cover_url"):
        return book["cover_url"]
    isbn = book.get("isbn","")
    cover = get_cover_google(isbn) or get_cover_openlibrary(isbn)
    book["cover_url"] = cover
    return cover

# ------------------- FASTAPI APP -------------------
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ممكن بعدين تعمليها لدومين محدد
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.get("/")
# def root():
#     return {"message": "Backend is working ✅"}
@app.get("/")
def health_check():
    return {"status": "ok", "service": "Bayan AI Librarian"}

@app.get("/health")
def health():
    return {"status": "healthy"}

conversation_history: List[Dict[str, Any]] = []
user_prefs: Dict[str, Any] = {}
# SESSION store in-memory
SESSIONS: Dict[str, Dict[str, Any]] = {}
# session structure:
# SESSIONS[session_id] = {
#   "conversation_history": [{"role":"user"/"assistant", "content": "..."}],
#   "user_prefs": {"pref_1": "...", ...},
#   "recommended": False,
#   "last_active": timestamp
# }

SESSION_TTL_SECONDS = 60 * 60  

class ChatRequest(BaseModel):
    message: Optional[str] = None
    session_id: Optional[str] = None
    reset: Optional[bool] = False

def create_session() -> str:
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {
        "conversation_history": [],
        "user_prefs": {},
        "recommended": False,
        "last_active": time.time()
    }
    return sid

def get_session(session_id: Optional[str]) -> str:
    # if no session_id provided or invalid, create new
    if not session_id or session_id not in SESSIONS:
        return create_session()
    return session_id

def touch_session(session_id: str):
    if session_id in SESSIONS:
        SESSIONS[session_id]["last_active"] = time.time()

def clear_session(session_id: str):
    # reinitialize session (keep same id) or remove entirely
    if session_id in SESSIONS:
        SESSIONS[session_id] = {
            "conversation_history": [],
            "user_prefs": {},
            "recommended": False,
            "last_active": time.time()
        }

def sweep_expired_sessions():
    now = time.time()
    to_delete = [sid for sid, s in SESSIONS.items() if now - s.get("last_active", 0) > SESSION_TTL_SECONDS]
    for sid in to_delete:
        SESSIONS.pop(sid, None)


@app.post("/chat")
def chat(req: ChatRequest):
    # optional: sweep expired sessions occasionally
    sweep_expired_sessions()

    print("\n🟢 --- Incoming Request ---")
    print(f"Reset: {req.reset}")
    print(f"Session ID: {req.session_id}")
    print(f"Message: {req.message}")
    print("----------------------------\n")

    # If user asked to reset/new chat, create a fresh session
    if req.reset:
        new_sid = create_session()
        starter = "New chat started. Hi! What kind of books are you in the mood for?"
        print(f"🟡 [New Session Created] session_id={new_sid}")
        print("⬆️ Sending Starter Response\n")
        return {
            "session_id": new_sid,
            "reply": starter,
            "books": [],
            "follow_up": True,
        }

    # Ensure we have a session id
    sid = get_session(req.session_id)
    session = SESSIONS[sid]
    touch_session(sid)

    print(f"🟡 [Session Active] session_id={sid}")
    print(f"🧾 Current user_prefs: {list(session['user_prefs'].values())}")
    print(f"🕓 Last active: {time.strftime('%X', time.localtime(session['last_active']))}")

    # If no message provided, just return session id
    if not req.message:
        print("⚪ No message provided — returning Ready response.\n")
        return {"session_id": sid, "reply": "Ready.", "books": [], "follow_up": True}

    user_text = req.message.strip()
    session["conversation_history"].append({"role": "user", "content": user_text})
    pref_key = f"pref_{len(session['user_prefs']) + 1}"
    session["user_prefs"][pref_key] = user_text

    try:
        lang = "ar" if detect(user_text) == "ar" else "en"
    except:
        lang = "en"
    print(f"🌍 Detected language: {lang}")

    trigger_terms = ["recommend", "suggest", "surprise", "اقترح", "رشح", "نصيحة"]
    need_recommend = any(t in user_text.lower() for t in trigger_terms) or len(session["user_prefs"]) >= 4

    # --- Recommendation stage ---
    if need_recommend:
        print("🟣 [Stage] Generating recommendations...")
        full_query = " ; ".join(session["user_prefs"].values())
        print(f"📋 Full user query: {full_query}")
# , user_lang=lang
        best_books = find_top_k(full_query, k=TOP_K)
        print(f"📚 Found {len(best_books)} similar books")
        
        for b in best_books:
            ensure_cover(b)
        print("books:", best_books)
        books_block = ""
        for b in best_books:
            if b.get('language', '').lower() == lang.lower():  # بس الكتب اللي اللغة بتاعتها = lang
                books_block += f"Title: {b['title']}\nAuthor: {b.get('authors','')}\nSummary: {b.get('short_summary','')}\n\n"

        prompt = f"""
You are a helpful librarian. The user described preferences: {full_query}.
Below are candidate books from {books_block}. For each book, write one short line in {lang} explaining why it matches the user's preferences. Keep the response focused only on the books and their reasons.
start the recommendation with a short introductory sentence without hello or welcomeing .
Reply in {lang} don't suggest not existing book here in the {books_block}.
"""
        print("🤖 Sending prompt to LLM for recommendation explanation...")
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        reply = resp.choices[0].message.content
        print("✅ [LLM Reply Received]")

        response = {
            "session_id": sid,
            "reply": reply,
            "books":[b for b in best_books if b.get('language','').lower() == lang.lower()],
            "follow_up": False,
        }

        print("\n🔵 --- Outgoing Response (Recommendation) ---")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        print("------------------------------------------------\n")

        # clear_session(sid)
        return response

    # --- Follow-up question stage ---
    else:
        print("🟢 [Stage] Generating follow-up question...")
        history_text = "\n".join([f"{h['role']}: {h['content']}" for h in session["conversation_history"]])

        prompt = f"""
You are a friendly, curious librarian. Ask one short, natural follow-up question that helps select a book.
Do not ask more than one question. Keep it specific and not repetitive.
If the user seems to have already given genre/mood/length or examples, ask about details like favorite authors, pace, or setting.
Respond in {lang}.
Conversation:
{history_text}
"""
        print("🤖 Sending prompt to LLM for follow-up question...")
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        reply = resp.choices[0].message.content
        print("✅ [LLM Reply Received]")

        session["conversation_history"].append({"role": "assistant", "content": reply})
        touch_session(sid)

        response = {"session_id": sid, "reply": reply, "books": [], "follow_up": True}

        print("\n🔵 --- Outgoing Response (Follow-up) ---")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        print("------------------------------------------------\n")

        return response
