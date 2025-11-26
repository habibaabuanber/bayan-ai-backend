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

# books_raw = load_books(BOOKS_FILE)[:200]
# embeddings, metas = load_embeddings()
# ------------ LOAD MULTIPLE DATASETS ------------
def load_multiple_jsonl(paths: List[str]):
    all_books = []
    for p in paths:
        print(f"Loading books from: {p}")
        all_books.extend(load_books(p))
    return all_books

def load_multiple_embeddings(emb_paths: List[str], meta_paths: List[str]):
    all_embs = []
    all_metas = []

    for emb_file, meta_file in zip(emb_paths, meta_paths):
        print(f"Loading embeddings from: {emb_file}")
        embs = np.load(emb_file)
        all_embs.append(embs)

        print(f"Loading metadata from: {meta_file}")
        with open(meta_file, "r", encoding="utf-8") as f:
            metas = json.load(f)
        all_metas.extend(metas)

    # concatenate embeddings vertically
    final_embs = np.vstack(all_embs)
    return final_embs, all_metas


# ---------- LOAD DATASETS ----------
books_raw = load_multiple_jsonl([
    "books_dataset_enriched.jsonl",
    "second_dataset_clean.jsonl"
])

embeddings, metas = load_multiple_embeddings(
    ["books_embeddings.npy", "second_dataset_embeddings.npy"],
    ["books_metadata.json", "second_dataset_metadata.json"]
)

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

import requests
import urllib.parse

def get_cover_by_title(title: str, authors: str = ""):
    try:
        clean_title = title.strip().replace('"', '').replace(':', '')
        query = f"intitle:{clean_title}"
        
        if authors:
            first_author = authors.split('/')[0].split(',')[0].strip()
            clean_author = first_author.replace('"', '').replace(':', '')
            query += f" inauthor:{clean_author}"
        
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.googleapis.com/books/v1/volumes?q={encoded_query}&maxResults=3"
        
        print(f"🔍 Searching for cover: {query}")
        
        r = requests.get(url, timeout=10)
        data = r.json()
        
        items = data.get("items", [])
        print(f"📚 Found {len(items)} items")
        
        if items:
            # نجرب كل النتائج عشان نلاقي واحدة فيها صورة
            for item in items:
                vol = item.get("volumeInfo", {})
                imgs = vol.get("imageLinks", {})
                
                # نطبع معلومات الكتاب للتdebug
                found_title = vol.get('title', '')
                found_authors = vol.get('authors', [])
                print(f"   📖 Found: '{found_title}' by {found_authors}")
                
                for k in ("extraLarge", "large", "medium", "small", "thumbnail"):
                    if imgs.get(k):
                        print(f"   ✅ Found cover: {imgs.get(k)}")
                        return imgs.get(k)
        
        print("   ❌ No covers found")
        return None
        
    except Exception as e:
        print(f"   ❌ Error in get_cover_by_title: {e}")
        return None
    
def ensure_cover(book: Dict[str, Any]):
    if book.get("cover_url"):
        return book["cover_url"]
    
    isbn = book.get("isbn","")
    cover = None
    
    if isbn and isbn != "N/A" and isbn != "null":
        cover = get_cover_google(isbn) or get_cover_openlibrary(isbn)
        if cover:
            print(f"✅ Found cover by ISBN: {isbn}")
    
    if not cover:
        cover = get_cover_by_title(book.get('title', ''), book.get('authors', ''))
        if cover:
            print(f"✅ Found cover by title: {book.get('title', '')}")
    
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

def normalize_language(lang):
    lang = str(lang).lower().strip()
    if lang in ['eng', 'english', 'en-us', 'en_us', 'en']:
        return 'en' 
    elif lang in ['ar', 'ara', 'arabic']:
        return 'ar'
    return lang


# @app.post("/chat")
# def chat(req: ChatRequest):
#     # optional: sweep expired sessions occasionally
#     sweep_expired_sessions()

#     print("\n🟢 --- Incoming Request ---")
#     print(f"Reset: {req.reset}")
#     print(f"Session ID: {req.session_id}")
#     print(f"Message: {req.message}")
#     print("----------------------------\n")

#     # If user asked to reset/new chat, create a fresh session
#     if req.reset:
#         new_sid = create_session()
#         starter = "New chat started. Hi! What kind of books are you in the mood for?"
#         print(f"🟡 [New Session Created] session_id={new_sid}")
#         print("⬆️ Sending Starter Response\n")
#         return {
#             "session_id": new_sid,
#             "reply": starter,
#             "books": [],
#             "follow_up": True,
#         }

#     # Ensure we have a session id
#     sid = get_session(req.session_id)
#     session = SESSIONS[sid]
#     touch_session(sid)

#     print(f"🟡 [Session Active] session_id={sid}")
#     print(f"🧾 Current user_prefs: {list(session['user_prefs'].values())}")
#     print(f"🕓 Last active: {time.strftime('%X', time.localtime(session['last_active']))}")

#     # If no message provided, just return session id
#     if not req.message:
#         print("⚪ No message provided — returning Ready response.\n")
#         return {"session_id": sid, "reply": "Ready.", "books": [], "follow_up": True}

#     user_text = req.message.strip()
#     session["conversation_history"].append({"role": "user", "content": user_text})
#     pref_key = f"pref_{len(session['user_prefs']) + 1}"
#     session["user_prefs"][pref_key] = user_text

#     try:
#         lang = "ar" if detect(user_text) == "ar" else "en"
#     except:
#         lang = "en"
#     print(f"🌍 Detected language: {lang}")
#     normalized_lang = normalize_language(lang)
#     print(f"🌍 Detected language: {lang} → Normalized: {normalized_lang}")

#     trigger_terms = ["recommend", "suggest", "surprise", "اقترح", "رشح", "نصيحة"]
#     need_recommend = any(t in user_text.lower() for t in trigger_terms) or len(session["user_prefs"]) >= 4
    
#     last_assistant_msg = None
#     if session["conversation_history"]:
#         for msg in reversed(session["conversation_history"]):
#             if msg["role"] == "assistant":
#                 last_assistant_msg = msg["content"]
#                 break

#     is_language_response = (
#         last_assistant_msg and 
#         any(phrase in last_assistant_msg for phrase in ["أي لغة تفضل أن تقرأ بها الكتب؟ العربية أم الإنجليزية؟", "Which language do you prefer to read books in? Arabic or English?"]) and
#         any(word in user_text.lower() for word in ["english", "eng", "en", "الإنجليزية", "انجليزي", "انجلش", "عربي", "عربية", "arabic", "ar"])
#     )

#     if is_language_response and "preferred_reading_lang" not in session:
#         print("🟡 [Stage] Processing language preference...")
        
#         # تحديد لغة القراءة المفضلة من رد المستخدم
#         if any(word in user_text.lower() for word in ["english", "eng", "en", "الإنجليزية", "انجليزي", "انجلش"]):
#             session["preferred_reading_lang"] = "en"
#             if normalized_lang == "ar":
#                 confirmation = "حسناً، سأوصي لك بكتب باللغة الإنجليزية 📚"
#             else:
#                 confirmation = "Great! I'll recommend books in English 📚"
#         else:
#             session["preferred_reading_lang"] = "ar"
#             if normalized_lang == "ar":
#                 confirmation = "حسناً، سأوصي لك بكتب باللغة العربية 📚"
#                 follow_up = "هل تريد أن أبدأ التوصية؟"

#             else:
#                 confirmation = "Great! I'll recommend books in Arabic 📚"
#                 follow_up = "Should I start the recommendation?"

#         print(f"📖 User preferred reading language: {session['preferred_reading_lang']}")
#         full_reply = f"{confirmation} {follow_up}"
#         session["conversation_history"].append({"role": "assistant", "content": full_reply})
#         touch_session(sid)

#         response = {"session_id": sid, "reply": full_reply, "books": [], "follow_up": True}

#         print("\n🔵 --- Outgoing Response (Language Confirmation) ---")
#         print(json.dumps(response, indent=2, ensure_ascii=False))
#         print("------------------------------------------------\n")

#         return response

#     # 2. ثم تحقق إذا وصلنا لمرحلة التوصية ولكن لم نسأل عن اللغة بعد
#     if need_recommend and "preferred_reading_lang" not in session:
#         print("🟡 [Stage] Asking for preferred reading language...")
        
#         if normalized_lang == "ar":
#             question = "أي لغة تفضل أن تقرأ بها الكتب؟ العربية أم الإنجليزية؟"
#         else:
#             question = "Which language do you prefer to read books in? Arabic or English?"
        
#         session["conversation_history"].append({"role": "assistant", "content": question})
#         touch_session(sid)

#         response = {"session_id": sid, "reply": question, "books": [], "follow_up": True}

#         print("\n🔵 --- Outgoing Response (Language Question) ---")
#         print(json.dumps(response, indent=2, ensure_ascii=False))
#         print("------------------------------------------------\n")

#         return response

#     # 3. ثم التوصية بعد تحديد اللغة
#     if need_recommend and "preferred_reading_lang" in session:
#         print("🟣 [Stage] Generating recommendations...")
        
#         # استخدام لغة القراءة المفضلة بدلاً من لغة المستخدم الأساسية
#         reading_lang = session["preferred_reading_lang"]
#         normalized_reading_lang = normalize_language(reading_lang)
#         print(f"📖 Using preferred reading language: {reading_lang} → Normalized: {normalized_reading_lang}")

#         full_query = " ; ".join(session["user_prefs"].values())
#         print(f"📋 Full user query: {full_query}")
        
#         best_books = find_top_k(full_query, k=TOP_K)
#         print(f"📚 Found {len(best_books)} similar books")
        
#         for b in best_books:
#             ensure_cover(b)
        
#         print("books:", best_books)
        
#         # Debug info
#         print(f"🔍 Language Debug:")
#         print(f"   User reading lang: {reading_lang} → Normalized: {normalized_reading_lang}")
#         print(f"   Book languages: {[b.get('language') for b in best_books]}")
#         print(f"   Normalized book languages: {[normalize_language(b.get('language', '')) for b in best_books]}")
        
#         matched_books = []
#         books_block = ""
#         for b in best_books:
#             book_lang_normalized = normalize_language(b.get('language', ''))
#             if book_lang_normalized == normalized_reading_lang:
#                 print(f"✅ Book language matched: {b.get('language', '')} → User reading lang: {normalized_reading_lang}") 
#                 books_block += f"Title: {b['title']}\nAuthor: {b.get('authors','')}\nSummary: {b.get('short_summary','')}\n\n"
#                 matched_books.append(b)
#             else:
#                 print(f"❌ Book language NOT matched: {b.get('language', '')} → User wanted: {normalized_reading_lang}")
#         if not matched_books: 
#             books_block += f" There is no preferred language books but there are books in {normalize_language(best_books[0].get('language', ''))}: Title: {b['title']}\nAuthor: {b.get('authors','')}\nSummary: {b.get('short_summary','')}\n\n"

#         prompt = f"""
# You are a helpful librarian. The user described preferences: {full_query},Reply in {normalized_lang}.
# Below are candidate books from {books_block}. For each book, write one short line in {normalized_lang} explaining why it matches the user's preferences. Keep the response focused only on the books and their reasons.
# start the recommendation with a short introductory sentence without hello or welcomeing .
# don't suggest not existing book here  {books_block}.
# respond in {lang}.
# """
#         print("🤖 Sending prompt to LLM for recommendation explanation...")
#         resp = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         reply = resp.choices[0].message.content
#         print("✅ [LLM Reply Received]")

#         # ⚠️ صحح الخطأ هنا - استخدم matched_books بدل best_books
#         response = {
#             "session_id": sid,
#             "reply": reply,
#             "books": matched_books,  # ⬅️ هنا التصحيح
#             "follow_up": False,
#         }

#         print("\n🔵 --- Outgoing Response (Recommendation) ---")
#         print(json.dumps(response, indent=2, ensure_ascii=False))
#         print("------------------------------------------------\n")

#         return response

#     # 4. وأخيراً follow-up question
#     else:
#         print("🟢 [Stage] Generating follow-up question...")
#         history_text = "\n".join([f"{h['role']}: {h['content']}" for h in session["conversation_history"]])

#         prompt = f"""
# You are a friendly, curious librarian. Ask one short, natural follow-up question that helps select a book.
# Do not ask more than one question. Keep it specific and not repetitive.
# If the user seems to have already given genre/mood/length or examples, ask about details like favorite authors, pace, or setting.
# Respond in {lang}.
# Conversation:
# {history_text}
# """
#         print("🤖 Sending prompt to LLM for follow-up question...")
#         resp = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         reply = resp.choices[0].message.content
#         print("✅ [LLM Reply Received]")

#         session["conversation_history"].append({"role": "assistant", "content": reply})
#         touch_session(sid)

#         response = {"session_id": sid, "reply": reply, "books": [], "follow_up": True}

#         print("\n🔵 --- Outgoing Response (Follow-up) ---")
#         print(json.dumps(response, indent=2, ensure_ascii=False))
#         print("------------------------------------------------\n")

#         return response

@app.post("/chat")
def chat(req: ChatRequest):
    # تنظيف الجلسات المنتهية
    sweep_expired_sessions()

    print("\n🟢 --- Incoming Request ---")
    print(f"Reset: {req.reset}")
    print(f"Session ID: {req.session_id}")
    print(f"Message: {req.message}")
    print("----------------------------\n")

    # إذا طلب المستخدم اعادة المحادثة
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

    # التأكد من وجود جلسة
    sid = get_session(req.session_id)
    session = SESSIONS[sid]
    touch_session(sid)

    print(f"🟡 [Session Active] session_id={sid}")
    print(f"🧾 Current user_prefs: {list(session['user_prefs'].values())}")

    # إذا لم يتم تقديم رسالة
    if not req.message:
        print("⚪ No message provided — returning Ready response.\n")
        return {"session_id": sid, "reply": "Ready.", "books": [], "follow_up": True}

    user_text = req.message.strip()
    session["conversation_history"].append({"role": "user", "content": user_text})

    # حفظ التفضيلات
    pref_key = f"pref_{len(session['user_prefs']) + 1}"
    session["user_prefs"][pref_key] = user_text

    # كشف اللغة
    try:
        lang = "ar" if detect(user_text) == "ar" else "en"
    except:
        lang = "en"
    normalized_lang = normalize_language(lang)
    print(f"🌍 Detected language: {lang} → Normalized: {normalized_lang}")

     
    
    last_assistant_msg = None
    for msg in reversed(session["conversation_history"]):
        if msg["role"] == "assistant":
            last_assistant_msg = msg["content"]
            break

    # كشف إذا كان رد على توصيات سابقة
    is_response_to_recommendations = False
    is_negative_feedback = False
    
    if session.get("recommended") and last_assistant_msg:
        # تحليل إذا كان المستخدم يرد على التوصيات
        # is_response_to_recommendations = any(keyword in last_assistant_msg.lower() for keyword in [
        #     "recommend", "suggest", "book", "رواية", "اقترح", "كتاب"
        # ])
        
        # if is_response_to_recommendations:
        #     print(f"RESPONSE OF RECOMMENDATION:{is_response_to_recommendations}")
        #     # كشف المشاعر باستخدام GPT
        #     is_negative_feedback = detect_negative_feedback(user_text, session["conversation_history"], last_assistant_msg)
        #     print(f"🎭 Detected response to recommendations - Negative: {is_negative_feedback}")
        try:
            prompt = f"""
    The user just replied to your previous message.
    Assistant's last message: "{last_assistant_msg[:200]}..."
    User's message: "{user_text}"

    Question:
    Is the user giving feedback on the book recommendations in the assistant's message? 
    Respond ONLY with "yes" or "no".
    """
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10
            )
            gpt_response = resp.choices[0].message.content.strip().lower()
            is_response_to_recommendations = (gpt_response == "yes")
            print(f"RESPONSE OF RECOMMENDATION (GPT): {is_response_to_recommendations}")
        except Exception as e:
            print(f"❌ GPT failed to detect response: {e}")
            is_response_to_recommendations = False

        if is_response_to_recommendations:
            # كشف المشاعر باستخدام GPT
            is_negative_feedback = detect_negative_feedback(user_text, session["conversation_history"], last_assistant_msg)
            print(f"🎭 Detected response to recommendations - Negative: {is_negative_feedback}")

    # إذا كان رد سلبي على التوصيات
        if is_negative_feedback:

            print("🟠 [Stage] User dissatisfied with recommendations, generating follow-up questions...")

            # توليد أسئلة متابعة ديناميكية
            followup_prompt = f"""
        The user expressed dissatisfaction with the previous book recommendations.

        User message: "{user_text}"
        Conversation context: {session['conversation_history'][-4:]}

        Your task:
        - Ask one natural follow-up question.
        - Ask ONLY in the user's language: {normalized_lang}
        - Do NOT apologize excessively.
        - Do NOT analyze the books.
        - Do NOT modify preferences.
        - Question must be conversational and help us understand *what they didn’t like*.
        - Do NOT return anything except the questions.
        """

            try:
                followup_resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": followup_prompt}],
                    temperature=0.7
                )
                reply = followup_resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"❌ GPT follow-up generation failed: {e}")
                if normalized_lang == "ar":
                    reply = "طيب، ممكن توضّح أكتر إيه اللي مخليّك مش حابب الاقتراحات السابقة؟"
                else:
                    reply = "Could you tell me more about what you didn’t like in the previous recommendations?"

            # حفظ رسالة المساعد
            session["conversation_history"].append({"role": "assistant", "content": reply})
            touch_session(sid)

            response = {
                "session_id": sid,
                "reply": reply,
                "books": [],            # No books here
                "follow_up": True       # Because we want the user to answer
            }

            print("\n🟠 --- Outgoing Response (Negative Feedback Follow-up Questions) ---")
            print(json.dumps(response, indent=2, ensure_ascii=False))
            print("--------------------------------------------------------\n")

            return response

    # trigger_terms = ["recommend", "suggest", "surprise", "اقترح", "رشح", "نصيحة"]
    #  any(t in user_text.lower() for t in trigger_terms) or
    need_recommend = len(session["user_prefs"]) >= 4
    
    is_language_response = (
        last_assistant_msg and 
        any(phrase in last_assistant_msg for phrase in ["أي لغة تفضل أن تقرأ بها الكتب؟ العربية أم الإنجليزية؟", "Which language do you prefer to read books in? Arabic or English?"]) and
        any(word in user_text.lower() for word in ["english", "eng", "en", "الإنجليزية", "انجليزي", "انجلش", "عربي", "عربية", "arabic", "ar"])
    )

    if is_language_response and "preferred_reading_lang" not in session:
        print("🟡 [Stage] Processing language preference...")
        
        if any(word in user_text.lower() for word in ["english", "eng", "en", "الإنجليزية", "انجليزي", "انجلش"]):
            session["preferred_reading_lang"] = "en"
            if normalized_lang == "ar":
                confirmation = "حسناً، سأوصي لك بكتب باللغة الإنجليزية 📚"
            else:
                confirmation = "Great! I'll recommend books in English 📚"
        else:
            session["preferred_reading_lang"] = "ar"
            if normalized_lang == "ar":
                confirmation = "حسناً، سأوصي لك بكتب باللغة العربية 📚"
            else:
                confirmation = "Great! I'll recommend books in Arabic 📚"

        follow_up = generate_contextual_followup(session["conversation_history"], normalized_lang)
        full_reply = f"{confirmation}\n\n{follow_up}"
        
        session["conversation_history"].append({"role": "assistant", "content": full_reply})
        touch_session(sid)

        response = {"session_id": sid, "reply": full_reply, "books": [], "follow_up": True}

        print("\n🔵 --- Outgoing Response (Language Confirmation) ---")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        print("------------------------------------------------\n")

        return response

    if need_recommend and "preferred_reading_lang" not in session:
        print("🟡 [Stage] Asking for preferred reading language...")
        
        if normalized_lang == "ar":
            question = "أي لغة تفضل أن تقرأ بها الكتب؟ العربية أم الإنجليزية؟"
        else:
            question = "Which language do you prefer to read books in? Arabic or English?"
        
        session["conversation_history"].append({"role": "assistant", "content": question})
        touch_session(sid)

        response = {"session_id": sid, "reply": question, "books": [], "follow_up": True}

        print("\n🔵 --- Outgoing Response (Language Question) ---")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        print("------------------------------------------------\n")

        return response

    if need_recommend and "preferred_reading_lang" in session:
        print("🟣 [Stage] Generating initial recommendations...")
        
        reading_lang = session["preferred_reading_lang"]
        normalized_reading_lang = normalize_language(reading_lang)
        print(f"📖 Using preferred reading language: {reading_lang} → Normalized: {normalized_reading_lang}")

        full_query = " ; ".join(session["user_prefs"].values())
        print(f"📋 Full user query: {full_query}")
        
        best_books = find_top_k(full_query, k=TOP_K)
        print(f"📚 Found {len(best_books)} similar books")
        
        for b in best_books:
            ensure_cover(b)
        
        matched_books = []
        for b in best_books:
            book_lang_normalized = normalize_language(b.get('language', ''))
            if book_lang_normalized == normalized_reading_lang:
                matched_books.append(b)

        if not matched_books and best_books:
            matched_books = best_books[:2]
            print("⚠️ No books in preferred language, showing alternatives")

        books_titles = [b['title'] for b in matched_books]
        prompt = f"""
You are a helpful librarian. The user described preferences: {full_query}

Below are candidate books: {books_titles}

For each book, write one short line in {normalized_lang} explaining why it matches the user's preferences. 
Keep the response focused only on the books and their reasons.
Start the recommendation with a short introductory sentence without hello or welcoming.
Don't suggest non-existing books.
Respond in {normalized_lang}.
"""
        print("🤖 Sending prompt to LLM for recommendation explanation...")
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        reply = resp.choices[0].message.content
        print("✅ [LLM Reply Received]")

        # إضافة سؤال متابعة
        follow_up_question = generate_contextual_followup(
            session["conversation_history"], 
            normalized_lang, 
            is_after_recommendation=True
        )

        full_reply = f"{reply}\n\n{follow_up_question}"
        
        session["conversation_history"].append({"role": "assistant", "content": full_reply})
        session["recommended"] = True
        touch_session(sid)

        response = {
            "session_id": sid,
            "reply": full_reply,
            "books": matched_books,
            "follow_up": True,
        }

        print("\n🔵 --- Outgoing Response (Initial Recommendation) ---")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        print("------------------------------------------------\n")

        return response

    else:
        print("🟢 [Stage] Generating follow-up question...")
        
        follow_up_question = generate_contextual_followup(
            session["conversation_history"], 
            normalized_lang, 
            is_after_recommendation=False
        )
        
        session["conversation_history"].append({"role": "assistant", "content": follow_up_question})
        touch_session(sid)

        response = {
            "session_id": sid, 
            "reply": follow_up_question,
            "books": [], 
            "follow_up": True
        }

        print("\n🔵 --- Outgoing Response (Follow-up) ---")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        print("------------------------------------------------\n")

        return response
    
def detect_negative_feedback(user_text: str, conversation_history: List[Dict], last_recommendation: str) -> bool:
    """
    بتكتشف إذا المستخدم مش عاجباه التوصيات من context المحادثة
    """
    user_text_lower = user_text.lower()
    
    negative_indicators = [
        "مش ", "لا ", "ما ", "وش ", "ماذا", "غير", "تاني", "اخرى", "بديل",
        "not", "no", "other", "different", "another", "else", "instead"
    ]
    
    has_negative_indicator = any(indicator in user_text_lower for indicator in negative_indicators)
    
    is_short_response = len(user_text.split()) < 4
    
    try:
        prompt = f"""
Analyze if this user is expressing dissatisfaction with book recommendations.
User message: "{user_text}"
Last assistant recommendation: "{last_recommendation[:200]}..."

Respond ONLY with "yes" or "no".
"""
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        gpt_analysis = resp.choices[0].message.content.strip().lower()
        print(f"🎭 GPT negative feedback analysis: {gpt_analysis}")
        return gpt_analysis == "yes" or (has_negative_indicator and is_short_response)
    except:
        # fallback إذا API مش شغالة
        return has_negative_indicator and is_short_response
def generate_contextual_followup(conversation_history: List[Dict], user_lang: str, is_after_recommendation: bool = False) -> str:
    """
    بتولد أسئلة متابعة ذكية بناءً على context المحادثة
    """
    # نجمع معلومات من المحادثة
    history_text = "\n".join([f"{h['role']}: {h['content']}" for h in conversation_history[-6:]])  # آخر 3 تبادلات
    
    if is_after_recommendation:
        prompt = f"""
Based on this conversation, generate ONE natural follow-up question to understand why the user might not be satisfied with the recommendations and what they'd prefer instead.

Conversation:
{history_text}

Requirements:
- Ask ONE question only
- Be curious and helpful, not repetitive
- Focus on understanding their specific taste better
- Respond in {user_lang}
"""
    else:
        prompt = f"""
Based on this conversation, generate ONE natural follow-up question that helps understand the user's book preferences better.

Conversation:
{history_text}

Requirements:
- Ask ONE question only  
- Be natural and conversational
- Don't repeat previous questions
- Respond in {user_lang}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return resp.choices[0].message.content.strip()
    except:
        # Fallback questions
        fallback_questions_ar = [
            "ما هو آخر كتاب قرأته وأعجبك؟",
            "هل تفضل القصص الخيالية أم الواقعية؟",
            "أي نوع من الشخصيات تجذبك أكثر في الروايات؟",
            "ما المزاج الذي تبحث عنه في كتابك القادم؟"
        ]
        fallback_questions_en = [
            "What's the last book you read and enjoyed?",
            "Do you prefer fictional stories or realistic ones?",
            "What type of characters attract you most in novels?",
            "What mood are you looking for in your next book?"
        ]
        import random
        questions = fallback_questions_ar if user_lang == "ar" else fallback_questions_en
        return random.choice(questions)