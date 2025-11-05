# import streamlit as st
# import json
# import random
# from langdetect import detect
# from openai import OpenAI

# # Initialize the LLM

# # # Load dataset
# # with open("books_dataset_enriched.jsonl", "r", encoding="utf-8") as f:
# #     books = [json.loads(line) for line in f]

# # st.set_page_config(page_title="📚 Smart Librarian Chatbot", page_icon="📖")
# # st.title("📚 Smart Librarian – Find your next book!")

# # # Chat history
# # if "messages" not in st.session_state:
# #     st.session_state.messages = []

# # # Display chat history
# # for msg in st.session_state.messages:
# #     st.chat_message(msg["role"]).markdown(msg["content"])

# # # User input
# # if user_input := st.chat_input("Tell me what kind of story or topic you’re into..."):
# #     st.session_state.messages.append({"role": "user", "content": user_input})
# #     st.chat_message("user").markdown(user_input)

# #     # Detect language
# #     lang = "ar" if detect(user_input) == "ar" else "en"

# #     # Build a short prompt with book summaries
# #     dataset_summary = "\n".join(
# #         [f"{b['title']} ({b['language']}): {b['final_summary']}" for b in books if b["final_summary"]]
# #     )

# #     # Query model
# #     prompt = f"""
# #     You are a friendly librarian assistant.
# #     The user wrote in {lang}. Suggest a book that best matches their mood or interests.
# #     Respond in the same language.
# #     Choose one book from this dataset:
# #     {dataset_summary}
# #     Reply with the title, ISBN, location, and short summary.
# #     """

# #     response = client.chat.completions.create(
# #         model="gpt-4o-mini",
# #         messages=[
# #             {"role": "system", "content": "You are a helpful multilingual librarian."},
# #             {"role": "user", "content": prompt},
# #             {"role": "user", "content": user_input}
# #         ]
# #     )

# #     answer = response.choices[0].message.content

# #     st.session_state.messages.append({"role": "assistant", "content": answer})
# #     st.chat_message("assistant").markdown(answer)

# # Load dataset
# with open("books_dataset_enriched.jsonl", "r", encoding="utf-8") as f:
#     books = [json.loads(line) for line in f]

# st.set_page_config(page_title="📚 Smart Librarian Chatbot", page_icon="📖")
# st.title("📚 Smart Librarian – Find your perfect book!")

# # Initialize session states
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "stage" not in st.session_state:
#     st.session_state.stage = "intro"
# if "user_prefs" not in st.session_state:
#     st.session_state.user_prefs = {}

# # Display chat history
# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).markdown(msg["content"])

# # Define conversation flow
# def next_prompt():
#     stage = st.session_state.stage
#     prefs = st.session_state.user_prefs

#     if stage == "intro":
#         st.session_state.stage = "genre"
#         return "Hi there! 👋 What kind of stories do you usually enjoy? (e.g., fiction, mystery, history, fantasy...)"
    
#     elif stage == "genre":
#         st.session_state.stage = "mood"
#         return "Nice choice! 🌟 What kind of mood are you in? (something emotional, adventurous, relaxing, thought-provoking?)"
    
#     elif stage == "mood":
#         st.session_state.stage = "length"
#         return "Got it! 📖 Do you prefer something short and quick to read or a long deep novel?"
    
#     elif stage == "length":
#         st.session_state.stage = "ready"
#         return "Perfect! Give me a second to find the best match for you..."

#     return None
# # Handle user input
# if user_input := st.chat_input("Say something..."):
#     st.chat_message("user").markdown(user_input)
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     lang = "ar" if detect(user_input) == "ar" else "en"

#     # Collect preferences step by step
#     if st.session_state.stage == "intro":
#         st.session_state.user_prefs["genre"] = user_input
#     elif st.session_state.stage == "genre":
#         st.session_state.user_prefs["mood"] = user_input
#     elif st.session_state.stage == "mood":
#         st.session_state.user_prefs["length"] = user_input
#     elif st.session_state.stage == "length":
#         st.session_state.user_prefs["extra"] = user_input
#         st.session_state.stage = "ready"  # Move to ready immediately

#     # If ready, generate the recommendation right away
#     if st.session_state.stage == "ready":
#         prefs = st.session_state.user_prefs
#         dataset_summary = "\n".join(
#             [f"{b['title']} ({b['language']}): {b['final_summary']}" for b in books]
#         )

#         prompt = f"""
#         You are a friendly librarian assistant speaking in {lang}.
#         The user described their interests as:
#         - Genre: {prefs.get('genre', '')}
#         - Mood: {prefs.get('mood', '')}
#         - Length/Type: {prefs.get('length', '')}
#         Suggest ONE book from this dataset that fits their vibe:
#         {dataset_summary}
#         Reply in {lang} with:
#         1. 📖 Title
#         2. 🔢 ISBN
#         3. 📍 Library location
#         4. ✍️ A short personalized explanation of why it fits their mood.
#         """

#         with st.spinner("Finding your perfect match... 💫"):
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "You are a warm, conversational librarian."},
#                     {"role": "user", "content": prompt}
#                 ]
#             )

#         answer = response.choices[0].message.content
#         st.session_state.messages.append({"role": "assistant", "content": answer})
#         st.chat_message("assistant").markdown(answer)
#         st.session_state.stage = "done"

#     else:
#         next_q = next_prompt()
#         if next_q:
#             st.session_state.messages.append({"role": "assistant", "content": next_q})
#             st.chat_message("assistant").markdown(next_q)
# import streamlit as st
# import json
# import random
# from langdetect import detect
# from openai import OpenAI

# # Initialize OpenAI client
# client = OpenAI(api_key="YOUR_API_KEY")  # Replace with your key

# # Load your dataset
# with open("books_dataset_enriched.jsonl", "r", encoding="utf-8") as f:
#     books = [json.loads(line) for line in f]

# st.set_page_config(page_title="📚 Smart Librarian", page_icon="📖")
# st.title("📚 Smart Librarian – Your Personal Book Companion")

# # Initialize session states
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "prefs" not in st.session_state:
#     st.session_state.prefs = {}
# if "stage" not in st.session_state:
#     st.session_state.stage = "intro"

# # Display chat history
# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).markdown(msg["content"])

# # Define helper to generate next dynamic question
# def generate_next_question(history, prefs, lang="en"):
#     history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
#     prompt = f"""
#     You are a friendly, curious librarian chatbot helping a visitor find the perfect book.
#     Ask one thoughtful, natural next question to better understand their interests.
#     Avoid repeating previous questions. Keep it conversational.
#     Examples:
#     - What kind of vibe do you like in stories?
#     - Do you enjoy more emotional or adventurous stories?
#     - Any favorite authors or books you’ve liked before?
#     - Are you in the mood for something deep, fun, or inspiring?

#     Respond in {lang}.
#     Conversation so far:
#     {history_text}
#     """
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message.content.strip()

# # Define book recommendation function
# def recommend_book(prefs):
#     dataset_summary = "\n".join(
#         [f"{b['title']} ({b['language']}): {b.get('short_summary', '')}" for b in books]
#     )

#     prompt = f"""
#     The user described their interests as:
#     {prefs}

#     From the dataset below, choose the TOP 3–4 books that best match the user’s interests.
#     {dataset_summary}

#     Reply in a friendly way, including:
#     📖 Title
#     ✍️ Author
#     🔢 ISBN
#     📍 Library Location
#     💬 One-sentence reason why this book matches their vibe.
#     """

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "You are a warm, conversational librarian."},
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response.choices[0].message.content.strip()

# # Handle user input
# if user_input := st.chat_input("Say something..."):
#     st.chat_message("user").markdown(user_input)
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     lang = "ar" if detect(user_input) == "ar" else "en"

#     # Save user input
#     st.session_state.prefs[len(st.session_state.prefs)] = user_input

#     # If we already asked 3-4 questions, recommend
#     if len(st.session_state.prefs) >= 4:
#         st.session_state.stage = "recommend"
#         answer = recommend_book(st.session_state.prefs)
#         st.session_state.messages.append({"role": "assistant", "content": answer})
#         st.chat_message("assistant").markdown(answer)
#         st.session_state.stage = "done"

#     # Otherwise, generate next dynamic question
#     elif st.session_state.stage in ["intro", "ask"]:
#         st.session_state.stage = "ask"
#         next_q = generate_next_question(st.session_state.messages, st.session_state.prefs, lang)
#         st.session_state.messages.append({"role": "assistant", "content": next_q})
#         st.chat_message("assistant").markdown(next_q)
# book_chatbot_streamlit.py
import os
import json
import time
import requests
from pathlib import Path

import streamlit as st
import numpy as np
from langdetect import detect
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv() 
# ------------------------------- CONFIG -------------------------------

# if not OPENAI_API_KEY:
#     raise RuntimeError("Please set OPENAI_API_KEY environment variable before running.")
with open("test.txt", "w") as f:
    f.write("ok")

import os


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
BOOKS_FILE = "books_dataset_enriched.jsonl"   # your enriched dataset
EMB_FILE = "books_embeddings.npy"
META_FILE = "books_meta.json"                 # metadata list (title,authors,isbn,location,summary,cover_url,...)

EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 64
TOP_K = 4

# ------------------------------- HELPERS -------------------------------
def load_books(path: str) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found — place your dataset in this file.")
    books = [json.loads(line) for line in open(path, "r", encoding="utf-8")]
    return books

def compute_book_text_for_embedding(book: Dict[str, Any]) -> str:
    # Combine useful text fields for embedding
    parts = [
        book.get("title", ""),
        book.get("authors", ""),
        book.get("final_summary", book.get("short_summary", "")),
        book.get("language", ""),
        book.get("genre", ""),
    ]
    return " — ".join([p for p in parts if p])

import numpy as np
import json
import os
from tqdm import tqdm

def create_embeddings_if_missing(books, batch_size=100):
    EMB_FILE = "books_embeddings.npy"
    META_FILE = "books_metadata.json"

    # ✅ If files already exist
    if os.path.exists(EMB_FILE) and os.path.exists(META_FILE):
        embeddings = np.load(EMB_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            metas = json.load(f)
        print(f"Loaded {len(embeddings)} embeddings.")
        return embeddings, metas

    print("🚀 Generating embeddings safely in batches...")
    all_embeddings = []
    all_metas = []

    for i in tqdm(range(0, len(books), batch_size)):
        batch = books[i:i+batch_size]
        texts = [b["title"] + " " + b.get("final_summary", "") for b in batch]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        batch_embeddings = [e.embedding for e in response.data]
        all_embeddings.extend(batch_embeddings)
        all_metas.extend(batch)

        # ✅ Convert to NumPy float32 array
        arr = np.array(all_embeddings, dtype=np.float32)

        # ✅ Save safely (temp file -> rename)
        tmp_emb_file = EMB_FILE + ".tmp"
        tmp_meta_file = META_FILE + ".tmp"

                # ✅ Save safely (force flush before rename)
        with open(tmp_emb_file, "wb") as f:
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())

        with open(tmp_meta_file, "w", encoding="utf-8") as f:
            json.dump(all_metas, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # ✅ Rename safely after writing is guaranteed done
        os.replace(tmp_emb_file, EMB_FILE)
        os.replace(tmp_meta_file, META_FILE)

    print(f"✅ Done. Saved {len(all_embeddings)} embeddings.")
    return np.array(all_embeddings, dtype=np.float32), all_metas

def embed_text(text: str) -> List[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def find_top_k(embeddings: np.ndarray, metas: List[Dict[str, Any]], query: str, k: int = TOP_K):
    query_emb = np.array(embed_text(query), dtype=np.float32).reshape(1, -1)
    sims = cosine_similarity(query_emb, embeddings)[0]
    idx = np.argsort(sims)[::-1][:k]
    results = []
    for i in idx:
        m = metas[int(i)].copy()
        m["_score"] = float(sims[int(i)])
        results.append(m)
    return results

# ------------------------------- COVERS -------------------------------
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
            # prefer larger if available
            for k in ("extraLarge", "large", "medium", "small", "thumbnail"):
                if imgs.get(k):
                    return imgs.get(k)
            return imgs.get("thumbnail")
    except Exception:
        return None

def get_cover_openlibrary(isbn: str):
    if not isbn:
        return None
    # Open Library cover URL format (large)
    return f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"

def ensure_cover(m):
    if m.get("cover_url"):
        return m["cover_url"]
    isbn = m.get("isbn", "")
    g = get_cover_google(isbn)
    if g:
        m["cover_url"] = g
        return g
    ol = get_cover_openlibrary(isbn)
    # Quick check if exists (OpenLibrary returns a placeholder for missing ISBNs, but usually returns 1xx/200)
    try:
        r = requests.get(ol, timeout=4)
        if r.status_code == 200 and len(r.content) > 1000:
            m["cover_url"] = ol
            return ol
    except Exception:
        pass
    m["cover_url"] = None
    return None

# ------------------------------- LLM Q + RECOMMEND -------------------------------
def generate_dynamic_question(history: List[Dict[str, str]], lang: str = "en"):
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    prompt = f"""
You are a friendly, curious librarian. Ask one short, natural follow-up question that helps select a book.
Do not ask more than one question. Keep it specific and not repetitive.
If the user seems to have already given genre/mood/length or examples, ask about details like favorite authors, pace, or setting.
Respond in {lang}.
Conversation:
{history_text}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=80
    )
    return resp.choices[0].message.content.strip()

def ask_recommendation_reason_for_list(user_prefs: Dict[str, Any], books_subset: List[Dict[str, Any]], lang: str = "en"):
    # Build dataset block (compact)
    block = ""
    for b in books_subset:
        block += f"Title: {b['title']}\nAuthor: {b.get('authors','')}\nISBN: {b.get('isbn','')}\library_location: {b.get('library_location','')}\nSummary: {b.get('summary','')}\n\n"

    prompt = f"""
You are a warm librarian. The user described preferences: {json.dumps(user_prefs, ensure_ascii=False)}.
Below are candidate books. For each book, write one short line (in {lang}) explaining why it matches the user's preferences.
Also at top produce a short friendly header recommending these books.

{block}
Reply in {lang}.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":"You are a helpful librarian."},
                  {"role":"user","content":prompt}],
        max_tokens=512
    )
    return resp.choices[0].message.content.strip()

# ------------------------------- STREAMLIT UI -------------------------------
st.set_page_config(page_title="📚 Smart Librarian", page_icon="📖", layout="wide")
st.title("📚 Smart Librarian – Discover your next read")

# Load and ensure embeddings
with st.spinner("Loading dataset and embeddings..."):
    books_raw = load_books(BOOKS_FILE)[:200]
    embeddings, metas = create_embeddings_if_missing(books_raw)  # returns embeddings np.array and metas list

# Session state for chat
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts {role, content}
if "prefs" not in st.session_state:
    st.session_state.prefs = {}
if "q_count" not in st.session_state:
    st.session_state.q_count = 0
if "recommended" not in st.session_state:
    st.session_state.recommended = None

# Show history
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

col1, col2 = st.columns([3,1])
with col1:
    user_text = st.chat_input("Tell me what you're in the mood for (genre, vibe, authors you like, or say 'surprise me')...")

with col2:
    if st.button("Reset conversation"):
        st.session_state.history = []
        st.session_state.prefs = {}
        st.session_state.q_count = 0
        st.session_state.recommended = None
        st.experimental_rerun()

if user_text:
    # add user message to history
    st.session_state.history.append({"role":"user","content":user_text})
    # detect language
    try:
        lang = "ar" if detect(user_text) == "ar" else "en"
    except Exception:
        lang = "en"

    # store preference as incremental items
    st.session_state.prefs[f"pref_{len(st.session_state.prefs)+1}"] = user_text
    st.session_state.q_count += 1

    # Decide when to recommend: after 3 inputs or user asks 'recommend' or 'surprise me'
    trigger_terms = ["recommend", "suggest", "surprise me", "اقترح", "اقترح لي", "نصيحة"]
    if any(t in user_text.lower() for t in trigger_terms) or st.session_state.q_count >= 3:
        # Build user query string from prefs
        user_desc = " ; ".join(list(st.session_state.prefs.values()))
        # find top K matches
        with st.spinner("Looking for best matches..."):
            best = find_top_k(embeddings, metas, user_desc, k=TOP_K)

            # ensure covers available (try to fetch if missing)
            for b in best:
                if not b.get("cover_url"):
                    c = ensure_cover(b)
                    b["cover_url"] = c

            # ask LLM to give friendly reasons for each
            reasons_text = ask_recommendation_reason_for_list(st.session_state.prefs, best, lang=lang)

        # display best books nicely
        st.session_state.recommended = best
        st.session_state.history.append({"role":"assistant","content":reasons_text})
        st.chat_message("assistant").markdown(reasons_text)

        # Show each book card
        for b in best:
            with st.container():
                cols = st.columns([1,3])
                if b.get("cover_url"):
                    try:
                        cols[0].image(b["cover_url"], width=140)
                    except Exception:
                        cols[0].image(None)
                else:
                    cols[0].write("")  # empty space if no cover

                info_md = f"**{b['title']}**  \n"
                if b.get("authors"):
                    info_md += f"*{b.get('authors')}*  \n"
                info_md += f"**ISBN:** {b.get('isbn', 'N/A')}  \n"
                info_md += f"**library_location:** {b.get('library_location', 'Unknown')}  \n\n"
                info_md += f"{b.get('summary','')[:400]}..."
                cols[1].markdown(info_md)

    else:
        # generate one dynamic follow-up question
        with st.spinner("Let me ask a quick question to narrow it down..."):
            next_q = generate_dynamic_question(st.session_state.history, lang=lang)
        st.session_state.history.append({"role":"assistant","content":next_q})
        st.chat_message("assistant").markdown(next_q)
