from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import time
import json
from langdetect import detect

# إنشاء التطبيق
app = FastAPI(title="Book Recommendation API", version="1.0.0")

# إضافة CORS للسماح بالطلبات من المتصفح
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # في الإنتاج غير هذا إلى أصول محددة
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELS ====================
class ChatRequest(BaseModel):
    message: Optional[str] = None
    session_id: Optional[str] = None
    reset: bool = False

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    books: List[Dict[str, Any]]
    follow_up: bool


# ==================== ROUTES ====================
@app.get("/")
def read_root():
    """الصفحة الرئيسية"""
    return {
        "message": "Welcome to Book Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /chat",
            "docs": "GET /docs",
            "health": "GET /health"
        }
    }

@app.get("/health")
def health_check():
    """فحص صحة الخادم"""
    return {"status": "healthy", "sessions": len(SESSIONS)}
@app.post("/chat")
def chat(req: ChatRequest):
    """الـ endpoint الرئيسي للدردشة"""
    # تنظيف الجلسات المنتهية
    sweep_expired_sessions()
    
    # تسجيل الطلب الوارد
    log_incoming_request(req)
    
    # إذا طلب reset، أنشئ جلسة جديدة
    if req.reset:
        return handle_new_session()
    
    # الحصول على الجلسة الحالية
    session_data = get_current_session(req.session_id)
    sid = session_data["sid"]
    session = session_data["session"]
    
    # إذا لم تكن هناك رسالة
    if not req.message:
        return handle_empty_message(sid)
    
    # معالجة الرسالة باستخدام النظام الذكي
    return process_user_message(req.message, sid, session)
# ==================== GLOBALS ====================
SESSIONS = {}
SESSION_TIMEOUT = 1800  # 30 دقيقة
TOP_K = 5

# ==================== SESSION UTILS ====================
def create_session() -> str:
    """إنشاء جلسة جديدة"""
    from uuid import uuid4
    sid = str(uuid4())[:8]
    SESSIONS[sid] = {
        "created_at": time.time(),
        "last_activity": time.time(),
        "user_prefs": {},
        "conversation_history": [],
        "recommended": False,
    }
    return sid

def get_session(session_id: str) -> str:
    """الحصول على جلسة أو إنشاء واحدة جديدة"""
    if session_id not in SESSIONS:
        return create_session()
    return session_id

def touch_session(sid: str):
    """تحديث وقت الجلسة"""
    if sid in SESSIONS:
        SESSIONS[sid]["last_activity"] = time.time()

def sweep_expired_sessions():
    """تنظيف الجلسات المنتهية"""
    current_time = time.time()
    expired = [
        sid for sid, data in SESSIONS.items()
        if current_time - data["last_activity"] > SESSION_TIMEOUT
    ]
    for sid in expired:
        del SESSIONS[sid]
    if expired:
        print(f"🧹 Swept {len(expired)} expired sessions")

# ==================== LANGUAGE UTILS ====================
def normalize_language(lang: str) -> str:
    """تطبيع اللغة"""
    lang = lang.lower().strip()
    if lang in ["ar", "arabic", "عربي", "عربية"]:
        return "ar"
    elif lang in ["en", "english", "eng", "انجليزي", "انجليزى"]:
        return "en"
    return "en"  # Default

def detect_and_normalize_language(text: str) -> Dict:
    """كشف اللغة وتطبيعها"""
    try:
        lang = "ar" if detect(text) == "ar" else "en"
    except:
        lang = "en"
    
    normalized_lang = normalize_language(lang)
    print(f"🌍 Detected language: {lang} → Normalized: {normalized_lang}")
    
    return {"detected": lang, "normalized": normalized_lang}

# ==================== CONVERSATION UTILS ====================
def get_last_assistant_message(conversation_history: List[Dict]) -> Optional[str]:
    """الحصول على آخر رسالة للمساعد"""
    for msg in reversed(conversation_history):
        if msg["role"] == "assistant":
            return msg["content"]
    return None

def update_conversation_history(session: Dict, role: str, content: str):
    """تحديث سجل المحادثة"""
    session["conversation_history"].append({"role": role, "content": content})

def save_user_preference(session: Dict, user_text: str):
    """حفظ تفضيلات المستخدم"""
    pref_key = f"pref_{len(session['user_prefs']) + 1}"
    session["user_prefs"][pref_key] = user_text

# ==================== LOGGING UTILS ====================
def log_incoming_request(req: ChatRequest):
    """تسجيل معلومات الطلب الوارد"""
    print("\n🟢 --- Incoming Request ---")
    print(f"Reset: {req.reset}")
    print(f"Session ID: {req.session_id}")
    print(f"Message: {req.message}")
    print("----------------------------\n")

def log_response(response_type: str, response: Dict):
    """تسجيل الرد الصادر"""
    print(f"\n🔵 --- Outgoing Response ({response_type}) ---")
    print(json.dumps(response, indent=2, ensure_ascii=False))
    print("-" * 50 + "\n")

# ==================== SESSION HANDLERS ====================
def handle_new_session():
    """معالجة جلسة جديدة"""
    new_sid = create_session()
    starter = "New chat started. Hi! What kind of books are you in the mood for?"
    
    print(f"🟡 [New Session Created] session_id={new_sid}")
    print("⬆️ Sending Starter Response\n")
    
    response = {
        "session_id": new_sid,
        "reply": starter,
        "books": [],
        "follow_up": True,
    }
    
    log_response("New Session", response)
    return response

def get_current_session(session_id: str):
    """الحصول على الجلسة الحالية"""
    sid = get_session(session_id)
    session = SESSIONS[sid]
    touch_session(sid)
    
    print(f"🟡 [Session Active] session_id={sid}")
    print(f"🧾 Current user_prefs: {list(session['user_prefs'].values())}")
    
    return {"sid": sid, "session": session}

def handle_empty_message(sid: str):
    """معالجة حالة عدم وجود رسالة"""
    print("⚪ No message provided — returning Ready response.\n")
    
    response = {
        "session_id": sid,
        "reply": "Ready.",
        "books": [],
        "follow_up": True
    }
    
    log_response("Empty Message", response)
    return response

# ==================== FEEDBACK HANDLERS ====================
def gpt_detect_response_to_recommendations(user_text: str, last_assistant_msg: str) -> bool:
    """استخدام GPT للتحقق إذا كان ردًا على التوصيات"""
    prompt = f"""
    The user just replied to your previous message.
    Assistant's last message: "{last_assistant_msg[:200]}..."
    User's message: "{user_text}"
    Question: Is the user giving feedback on the book recommendations in the assistant's message?
    Respond ONLY with "yes" or "no".
    """
    
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10
    )
    
    gpt_response = resp.choices[0].message.content.strip().lower()
    result = (gpt_response == "yes")
    print(f"RESPONSE OF RECOMMENDATION (GPT): {result}")
    
    return result

def check_feedback_on_recommendations(user_text: str, session: Dict, last_assistant_msg: Optional[str]) -> Dict:
    """التحقق إذا كان المستخدم يرد على التوصيات"""
    is_response_to_recommendations = False
    is_negative_feedback = False
    
    if session.get("recommended") and last_assistant_msg:
        try:
            is_response_to_recommendations = gpt_detect_response_to_recommendations(
                user_text, last_assistant_msg
            )
            
            if is_response_to_recommendations:
                is_negative_feedback = detect_negative_feedback(
                    user_text, session["conversation_history"], last_assistant_msg
                )
                print(f"🎭 Detected response to recommendations - Negative: {is_negative_feedback}")
                
        except Exception as e:
            print(f"❌ GPT failed to detect response: {e}")
    
    return {
        "is_response_to_recommendations": is_response_to_recommendations,
        "is_negative_feedback": is_negative_feedback
    }
def detect_negative_feedback(user_text: str, conversation_history: List[Dict], last_recommendation: str) -> bool:
    """كشف الردود السلبية باستخدام LLM"""
    
    prompt = f"""
    Analyze if the user is expressing dissatisfaction or negative feedback about the book recommendations.
    
    Last recommendation: "{last_recommendation[:300]}"
    User's response: "{user_text}"
    
    Consider:
    1. Is the user saying they don't like the recommendations?
    2. Is the user saying the books are not relevant?
    3. Is the user expressing disappointment?
    4. Is the user asking for different books?
    
    Respond with ONLY "yes" or "no".
    
    Examples of negative feedback:
    - "I don't like these"
    - "These are not what I wanted"
    - "Not relevant to my interests"
    - "Can you suggest something else?"
    
    Examples of NOT negative feedback:
    - "Thanks, these look interesting"
    - "Tell me more about the first book"
    - "Do you have more options?"
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().lower()
        is_negative = result == "yes"
        print(f"🎭 LLM Negative Feedback Detection: {is_negative} ({result})")
        return is_negative
        
    except Exception as e:
        print(f"❌ LLM feedback detection failed: {e}")
        # Fallback إلى التحليل البسيط
        return detect_negative_feedback_fallback(user_text)

def detect_negative_feedback_fallback(user_text: str) -> bool:
    """Fallback لتحليل الردود السلبية"""
    user_text_lower = user_text.lower()
    
    negative_indicators = [
        # English
        "don't like", "do not like", "not good", "not what i wanted",
        "not relevant", "not interested", "not helpful", "bad",
        "terrible", "awful", "horrible", "disappointed", "disappointing",
        "useless", "waste", "wrong", "incorrect", "not right",
        "try again", "start over", "different", "another", "other",
        "something else", "no thanks", "no thank you",
        
        # Arabic
        "مش عاجبني", "لا يعجبني", "ليس جيد", "ليس ما أردت",
        "غير مناسب", "غير مهتم", "غير مفيد", "سيء",
        "فظيع", "مخيب", "خيبة أمل", "غير صحيح",
        "حاول مرة أخرى", "ابدأ من جديد", "مختلف", "آخر",
        "شيء آخر", "لا شكرا"
    ]
    
    # التحقق من وجود مؤشرات سلبية
    has_negative = any(indicator in user_text_lower for indicator in negative_indicators)
    
    # أيضاً تحقق من النفي
    negation_words = ["not", "no", "don't", "doesn't", "isn't", "wasn't", "weren't", "لا", "ليس", "ما", "مش"]
    has_negation = any(word in user_text_lower.split() for word in negation_words)
    
    # إذا كان النص قصيراً ويحتوي على نفي
    is_short_negative = len(user_text.split()) < 5 and has_negation
    
    result = has_negative or is_short_negative
    print(f"🎭 Fallback Negative Detection: {result} (has_negative: {has_negative}, is_short_negative: {is_short_negative})")
    
    return result


def handle_negative_feedback(user_text: str, sid: str, session: Dict, normalized_lang: str):
    """معالجة الرد السلبي على التوصيات - محدثة"""
    print("🟠 [Stage] User dissatisfied with recommendations, generating smart follow-up...")
    
    # تحليل سبب عدم الرضا باستخدام LLM
    analysis_prompt = f"""
    The user expressed dissatisfaction with book recommendations.
    
    User's negative feedback: "{user_text}"
    Conversation context: {session['conversation_history'][-4:]}
    
    Analyze why the user might be dissatisfied:
    1. Are the books not relevant to their interests?
    2. Are the books not in their preferred language?
    3. Are the books not matching their preferred genre/type?
    4. Something else?
    
    Based on your analysis, suggest ONE helpful follow-up question that will help us understand what they really want.
    Write the question in {normalized_lang}.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.7,
            max_tokens=150
        )
        
        reply = response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"❌ Smart negative feedback analysis failed: {e}")
        # Fallback
        if normalized_lang == "ar":
            reply = "عذراً، يبدو أن التوصيات لم تكن مناسبة. هل يمكنك إخباري أكثر عن نوع الكتب التي تبحث عنها تحديداً؟"
        else:
            reply = "I'm sorry the recommendations weren't right. Could you tell me more specifically what kind of books you're looking for?"
    
    # حفظ رسالة المساعد
    update_conversation_history(session, "assistant", reply)
    touch_session(sid)
    
    response = {
        "session_id": sid,
        "reply": reply,
        "books": [],
        "follow_up": True
    }
    
    log_response("Smart Negative Feedback Response", response)
    return response
def gpt_detect_response_to_recommendations(user_text: str, last_assistant_msg: str) -> bool:
    """استخدام GPT للتحقق إذا كان ردًا على التوصيات"""
    if not last_assistant_msg:
        return False
    
    prompt = f"""
    Determine if the user is responding to/referencing the book recommendations in the assistant's last message.
    
    Assistant's last message: "{last_assistant_msg[:200]}..."
    User's message: "{user_text}"
    
    Respond with ONLY "yes" or "no".
    
    Consider:
    - Is the user commenting on the recommended books?
    - Is the user asking about specific books mentioned?
    - Is the user expressing opinion about the recommendations?
    - Is the user requesting changes/modifications to recommendations?
    """
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        
        gpt_response = resp.choices[0].message.content.strip().lower()
        result = (gpt_response == "yes")
        print(f"🤖 GPT Response Detection: {result} ({gpt_response})")
        return result
        
    except Exception as e:
        print(f"❌ GPT detection failed: {e}")
        # Fallback بسيط
        return simple_detect_response_to_recommendations(user_text, last_assistant_msg)

def simple_detect_response_to_recommendations(user_text: str, last_assistant_msg: str) -> bool:
    """تحليل بسيط للرد على التوصيات"""
    if not last_assistant_msg:
        return False
    
    user_lower = user_text.lower()
    last_msg_lower = last_assistant_msg.lower()
    
    # إذا كان الرد يحتوي على كلمات مرتبطة بالتوصيات
    recommendation_keywords = [
        "book", "books", "recommend", "recommendation", "suggest", "suggestion",
        "title", "author", "this book", "these books", "that book",
        "كتاب", "كتب", "اقتراح", "توصية", "رواية", "عنوان", "مؤلف"
    ]
    
    # إذا كانت الرسالة الأخيرة تحتوي على كلمات توصية
    has_recommendation_in_last_msg = any(keyword in last_msg_lower for keyword in recommendation_keywords)
    
    # إذا كان رد المستخدم يحتوي على إشارة للتوصيات
    has_response_to_recommendation = any(keyword in user_lower for keyword in recommendation_keywords)
    
    # أو إذا كان يعلق على شيء محدد في التوصيات
    is_commenting = any(word in user_lower for word in ["like", "don't like", "good", "bad", "interesting", "عاجب", "مش عاجب"])
    
    return has_recommendation_in_last_msg and (has_response_to_recommendation or is_commenting)
def gpt_detect_response_to_recommendations(user_text: str, last_assistant_msg: str) -> bool:
    """استخدام GPT للتحقق إذا كان ردًا على التوصيات"""
    if not last_assistant_msg:
        return False
    
    prompt = f"""
    Determine if the user is responding to/referencing the book recommendations in the assistant's last message.
    
    Assistant's last message: "{last_assistant_msg[:200]}..."
    User's message: "{user_text}"
    
    Respond with ONLY "yes" or "no".
    
    Consider:
    - Is the user commenting on the recommended books?
    - Is the user asking about specific books mentioned?
    - Is the user expressing opinion about the recommendations?
    - Is the user requesting changes/modifications to recommendations?
    """
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        
        gpt_response = resp.choices[0].message.content.strip().lower()
        result = (gpt_response == "yes")
        print(f"🤖 GPT Response Detection: {result} ({gpt_response})")
        return result
        
    except Exception as e:
        print(f"❌ GPT detection failed: {e}")
        # Fallback بسيط
        return simple_detect_response_to_recommendations(user_text, last_assistant_msg)

def simple_detect_response_to_recommendations(user_text: str, last_assistant_msg: str) -> bool:
    """تحليل بسيط للرد على التوصيات"""
    if not last_assistant_msg:
        return False
    
    user_lower = user_text.lower()
    last_msg_lower = last_assistant_msg.lower()
    
    # إذا كان الرد يحتوي على كلمات مرتبطة بالتوصيات
    recommendation_keywords = [
        "book", "books", "recommend", "recommendation", "suggest", "suggestion",
        "title", "author", "this book", "these books", "that book",
        "كتاب", "كتب", "اقتراح", "توصية", "رواية", "عنوان", "مؤلف"
    ]
    
    # إذا كانت الرسالة الأخيرة تحتوي على كلمات توصية
    has_recommendation_in_last_msg = any(keyword in last_msg_lower for keyword in recommendation_keywords)
    
    # إذا كان رد المستخدم يحتوي على إشارة للتوصيات
    has_response_to_recommendation = any(keyword in user_lower for keyword in recommendation_keywords)
    
    # أو إذا كان يعلق على شيء محدد في التوصيات
    is_commenting = any(word in user_lower for word in ["like", "don't like", "good", "bad", "interesting", "عاجب", "مش عاجب"])
    
    return has_recommendation_in_last_msg and (has_response_to_recommendation or is_commenting)
def check_feedback_on_recommendations(user_text: str, session: Dict, last_assistant_msg: Optional[str]) -> Dict:
    """التحقق إذا كان المستخدم يرد على التوصيات - محدثة"""
    is_response_to_recommendations = False
    is_negative_feedback = False
    
    if session.get("recommended") and last_assistant_msg:
        try:
            # استخدام GPT للكشف
            is_response_to_recommendations = gpt_detect_response_to_recommendations(
                user_text, last_assistant_msg
            )
            
            if is_response_to_recommendations:
                # استخدام GPT للكشف عن السلبية
                is_negative_feedback = detect_negative_feedback(
                    user_text, session["conversation_history"], last_assistant_msg
                )
                print(f"🎭 Detected response to recommendations - Negative: {is_negative_feedback}")
                
        except Exception as e:
            print(f"❌ GPT detection failed: {e}")
            # استخدام fallback
            is_response_to_recommendations = simple_detect_response_to_recommendations(user_text, last_assistant_msg)
            if is_response_to_recommendations:
                is_negative_feedback = detect_negative_feedback_fallback(user_text)
    
    return {
        "is_response_to_recommendations": is_response_to_recommendations,
        "is_negative_feedback": is_negative_feedback
    }
    
# ==================== LANGUAGE HANDLERS ====================
def check_language_response(user_text: str, session: Dict, last_assistant_msg: Optional[str], normalized_lang: str):
    """التحقق من الإجابة على سؤال اللغة"""
    is_language_response = (
        last_assistant_msg and
        any(phrase in last_assistant_msg for phrase in [
            "أي لغة تفضل أن تقرأ بها الكتب؟ العربية أم الإنجليزية؟",
            "Which language do you prefer to read books in? Arabic or English?"
        ]) and
        any(word in user_text.lower() for word in [
            "english", "eng", "en", "الإنجليزية", "انجليزي", "انجلش",
            "عربي", "عربية", "arabic", "ar"
        ])
    )
    
    if is_language_response and "preferred_reading_lang" not in session:
        return process_language_preference(user_text, session, normalized_lang)
    
    return None

def process_language_preference(user_text: str, session: Dict, normalized_lang: str):
    """معالجة تفضيل لغة القراءة"""
    print("🟡 [Stage] Processing language preference...")
    
    # تحديد اللغة المختارة
    if any(word in user_text.lower() for word in [
        "english", "eng", "en", "الإنجليزية", "انجليزي", "انجلش"
    ]):
        session["preferred_reading_lang"] = "en"
        confirmation = "Great! I'll recommend books in English 📚" if normalized_lang != "ar" else "حسناً، سأوصي لك بكتب باللغة الإنجليزية 📚"
    else:
        session["preferred_reading_lang"] = "ar"
        confirmation = "Great! I'll recommend books in Arabic 📚" if normalized_lang != "ar" else "حسناً، سأوصي لك بكتب باللغة العربية 📚"
    
    # توليد سؤال متابعة
    follow_up = generate_contextual_followup(
        session["conversation_history"], normalized_lang
    )
    full_reply = f"{confirmation}\n\n{follow_up}"
    
    # تحديث المحادثة
    update_conversation_history(session, "assistant", full_reply)
    touch_session(session.get("sid", ""))
    
    # إعداد الرد
    response = {
        "session_id": session.get("sid", ""),
        "reply": full_reply,
        "books": [],
        "follow_up": True
    }
    
    log_response("Language Confirmation", response)
    return response

def ask_preferred_language(sid: str, session: Dict, normalized_lang: str):
    """سؤال عن لغة القراءة المفضلة"""
    print("🟡 [Stage] Asking for preferred reading language...")
    
    question = (
        "أي لغة تفضل أن تقرأ بها الكتب؟ العربية أم الإنجليزية؟"
        if normalized_lang == "ar"
        else "Which language do you prefer to read books in? Arabic or English?"
    )
    
    update_conversation_history(session, "assistant", question)
    touch_session(sid)
    
    response = {
        "session_id": sid,
        "reply": question,
        "books": [],
        "follow_up": True
    }
    
    log_response("Language Question", response)
    return response

# ==================== RECOMMENDATION HANDLERS ====================
def filter_books_by_language(books: List[Dict], target_lang: str) -> List[Dict]:
    """تصفية الكتب حسب اللغة"""
    matched_books = []
    
    for book in books:
        book_lang_normalized = normalize_language(book.get('language', ''))
        if book_lang_normalized == target_lang:
            matched_books.append(book)
    
    # إذا لم توجد كتب باللغة المطلوبة
    if not matched_books and books:
        matched_books = books[:2]
        print("⚠️ No books in preferred language, showing alternatives")
    
    return matched_books

def generate_recommendation_explanation(query: str, books: List[Dict], language: str) -> str:
    """توليد شرح للتوصيات"""
    books_titles = [book['title'] for book in books]
    
    prompt = f"""
    You are a helpful librarian. The user described preferences: {query}
    Below are candidate books: {books_titles}
    
    For each book, write one short line in {language} explaining why it matches the user's preferences.
    Keep the response focused only on the books and their reasons.
    Start the recommendation with a short introductory sentence without hello or welcoming.
    Don't suggest non-existing books.
    Respond in {language}.
    """
    
    print("🤖 Sending prompt to LLM for recommendation explanation...")
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    reply = resp.choices[0].message.content
    print("✅ [LLM Reply Received]")
    
    return reply

def generate_book_recommendations(sid: str, session: Dict, normalized_lang: str):
    """توليد توصيات الكتب"""
    print("🟣 [Stage] Generating initial recommendations...")
    
    # الحصول على لغة القراءة المفضلة
    reading_lang = session["preferred_reading_lang"]
    normalized_reading_lang = normalize_language(reading_lang)
    print(f"📖 Using preferred reading language: {reading_lang} → Normalized: {normalized_reading_lang}")
    
    # بناء الاستعلام
    full_query = " ; ".join(session["user_prefs"].values())
    print(f"📋 Full user query: {full_query}")
    
    # البحث عن الكتب
    best_books = find_top_k(full_query, k=TOP_K)
    print(f"📚 Found {len(best_books)} similar books")
    
    # ضمان وجود صور الغلاف
    for book in best_books:
        ensure_cover(book)
    
    # تصفية الكتب حسب اللغة
    matched_books = filter_books_by_language(
        best_books, normalized_reading_lang
    )
    
    # توليد الشرح
    explanation = generate_recommendation_explanation(
        full_query, matched_books, normalized_lang
    )
    
    # إضافة سؤال متابعة
    follow_up_question = generate_contextual_followup(
        session["conversation_history"], normalized_lang, is_after_recommendation=True
    )
    full_reply = f"{explanation}\n\n{follow_up_question}"
    
    # تحديث المحادثة
    update_conversation_history(session, "assistant", full_reply)
    session["recommended"] = True
    touch_session(sid)
    
    # إعداد الرد
    response = {
        "session_id": sid,
        "reply": full_reply,
        "books": matched_books,
        "follow_up": True,
    }
    
    log_response("Initial Recommendation", response)
    return response

def generate_follow_up_question(sid: str, session: Dict, normalized_lang: str, is_after_recommendation: bool = False):
    """توليد سؤال متابعة"""
    print("🟢 [Stage] Generating follow-up question...")
    
    follow_up_question = generate_contextual_followup(
        session["conversation_history"], normalized_lang, is_after_recommendation
    )
    
    update_conversation_history(session, "assistant", follow_up_question)
    touch_session(sid)
    
    response = {
        "session_id": sid,
        "reply": follow_up_question,
        "books": [],
        "follow_up": True
    }
    
    log_response("Follow-up", response)
    return response

# ==================== MAIN MESSAGE PROCESSING ====================
def should_save_as_preference(user_text: str, session: Dict, last_assistant_msg: Optional[str]) -> bool:
    """تحديد إذا كان يجب حفظ النص كتفضيل"""
    user_text_lower = user_text.lower().strip()
    
    # لا تحفظ إجابات على أسئلة اللغة
    language_question_indicators = [
        "أي لغة تفضل",
        "which language",
        "arabic or english",
        "العربية أم الإنجليزية"
    ]
    
    if last_assistant_msg:
        last_msg_lower = last_assistant_msg.lower()
        if any(indicator in last_msg_lower for indicator in language_question_indicators):
            # إذا كان الرد على سؤال اللغة، لا تحفظه كتفضيل
            if any(word in user_text_lower for word in ["arabic", "english", "عربي", "انجليزي", "ar", "en"]):
                return False
    
    # لا تحفظ ردود سلبية مثل "none", "لا", "nothing"
    negative_responses = ["none", "nothing", "لا", "ليس لدي", "لا أعرف", "لا يوجد"]
    if user_text_lower in negative_responses:
        return False
    
    # لا تحفظ إجابات على أسئلة المتابعة العامة
    follow_up_keywords = ["what's the last book", "آخر كتاب", "ما هو آخر كتاب"]
    if last_assistant_msg and any(keyword in last_assistant_msg.lower() for keyword in follow_up_keywords):
        if len(user_text.split()) < 3:  # إجابات قصيرة
            return False
    
    # القاعدة الأساسية: تحفظ فقط نصوص ذات معنى
    if len(user_text.split()) < 2:  # كلمة واحدة فقط
        return False
    
    return True

def save_user_preference(session: Dict, user_text: str, last_assistant_msg: Optional[str] = None):
    """حفظ تفضيلات المستخدم - محسنة"""
    if not should_save_as_preference(user_text, session, last_assistant_msg):
        print(f"⏩ Skipping save: '{user_text}' (not a meaningful preference)")
        return
    
    pref_key = f"pref_{len(session['user_prefs']) + 1}"
    session["user_prefs"][pref_key] = user_text
    print(f"💾 Saved preference {pref_key}: '{user_text[:50]}...'")
def analyze_preferences_with_llm(conversation_history: List[Dict]) -> Dict:
    """تحليل محادثة المستخدم باستخدام LLM لفهم التفضيلات"""
    # أخذ آخر 6 رسائل للمحادثة
    recent_history = conversation_history[-6:]
    
    prompt = f"""
    Analyze this book recommendation conversation and extract:
    1. What the user wants (genres, topics, specific interests)
    2. What the user does NOT want (dislikes, things to avoid)
    3. Reading language preference (Arabic/English)
    4. Key search terms for finding books
    
    Conversation:
    {format_conversation(recent_history)}
    
    Return a JSON with this structure:
    {{
        "wants": ["list", "of", "topics", "genres"],
        "does_not_want": ["things", "to", "avoid"],
        "language": "ar" or "en",
        "search_terms": ["keywords", "for", "search"],
        "summary": "brief summary of preferences"
    }}
    
    Be specific. If the user mentions "pharaohs", include "ancient egypt", "egyptian history", etc.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # أو gpt-4o إذا تريد
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        analysis = json.loads(response.choices[0].message.content)
        print(f"🤖 LLM Analysis: {analysis}")
        return analysis
        
    except Exception as e:
        print(f"❌ LLM analysis failed: {e}")
        # Fallback بسيط
        return {
            "wants": [],
            "does_not_want": [],
            "language": "ar",
            "search_terms": extract_keywords_fallback(conversation_history),
            "summary": "User preferences"
        }
def smart_recommendation_system(sid: str, session: Dict, normalized_lang: str):
    """نظام توصيات ذكي يعتمد على LLM"""
    print("🧠 [Stage] Using intelligent recommendation system...")
    
    # Step 1: تحليل التفضيلات باستخدام LLM
    preferences_analysis = analyze_preferences_with_llm(session["conversation_history"])
    
    # تحديث لغة الجلسة إذا تم اكتشافها
    detected_lang = preferences_analysis.get("language", normalized_lang)
    if detected_lang and "preferred_reading_lang" not in session:
        session["preferred_reading_lang"] = detected_lang
    
    # Step 2: بناء استعلام ذكي
    meaningful_prefs = [
        pref for pref in session["user_prefs"].values() 
        if len(pref.split()) > 1
    ]
    
    user_query = " ".join(meaningful_prefs) if meaningful_prefs else "books"
    
    # Step 3: البحث الذكي
    reading_lang = session.get("preferred_reading_lang", detected_lang)
    matched_books = intelligent_book_search(
        user_query, 
        preferences_analysis, 
        reading_lang
    )
    
    # Step 4: إذا لم توجد نتائج، حاول استراتيجيات مختلفة
    if not matched_books:
        print("🔄 No books found, trying alternative strategies...")
        
        # استراتيجية 1: البحث باستخدام كلمات أعمق
        broader_search_terms = generate_broader_search_terms(preferences_analysis)
        for term in broader_search_terms[:3]:
            matched_books = intelligent_book_search(term, preferences_analysis, reading_lang)
            if matched_books:
                break
        
        # استراتيجية 2: البحث باللغة الأخرى
        if not matched_books and reading_lang == "ar":
            print("🔄 Trying English books as fallback...")
            matched_books = intelligent_book_search(
                user_query, 
                preferences_analysis, 
                "en"
            )
    
    # Step 5: توليد الشرح الذكي
    if matched_books:
        explanation = generate_intelligent_explanation(
            matched_books, 
            preferences_analysis, 
            normalized_lang
        )
        
        # تحديث المحادثة
        update_conversation_history(session, "assistant", explanation)
        session["recommended"] = True
        touch_session(sid)
        
        response = {
            "session_id": sid,
            "reply": explanation,
            "books": matched_books,
            "follow_up": True,
        }
        
        log_response("Intelligent Recommendation", response)
        return response
    
    else:
        # إذا لم توجد كتب على الإطلاق
        return generate_no_books_smart_response(sid, session, preferences_analysis, normalized_lang)

def generate_intelligent_explanation(books: List[Dict], preferences: Dict, language: str) -> str:
    """توليد شرح مختصر وجذاب للتوصيات"""
    
    books_info = []
    for book in books[:4]:  # 4 كتب كحد أقصى
        title = book.get("title", "Unknown Book") or "Unknown Book"
        authors = book.get("authors", "Unknown Author") or "Unknown Author"
        
        books_info.append({
            "title": title,
            "authors": authors
        })
    
    prompt = f"""
    You are a helpful book recommender. Write a SHORT, engaging response.
    
    User is looking for: {preferences.get('summary', 'interesting books')}
    
    Books you're recommending (show only titles and authors):
    {json.dumps(books_info, ensure_ascii=False)}
    
    Write a concise message that:
    1. Starts with a friendly greeting (1 line max)
    2. Mentions you found some books matching their interests
    3. Lists the books clearly (title by author)
    4. Ends with ONE simple question to continue
    
    Keep it VERY SHORT - maximum 8-10 lines total.
    Be enthusiastic but concise.
    Write in {language}.
    
    Example format:
    "Great! Based on your interest in Victorian romance, here are some recommendations:
    
    1. Jane Eyre by Charlotte Brontë
    2. Pride and Prejudice by Jane Austen
    
    Which one sounds most interesting to you?"
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # استخدام mini للأقصر
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200  # 👈 هذا المهم! يحدد الطول
        )
        
        reply = response.choices[0].message.content.strip()
        print(f"📝 Reply length: {len(reply)} characters")
        return reply
        
    except Exception as e:
        print(f"❌ Intelligent explanation failed: {e}")
        # Fallback مختصر
        return generate_short_fallback_explanation(books, preferences, language)

def generate_short_fallback_explanation(books: List[Dict], preferences: Dict, language: str) -> str:
    """شرح مختصر fallback"""
    
    if language == "ar":
        lines = ["عثرت على بعض الكتب التي قد تعجبك:"]
        
        for i, book in enumerate(books[:3], 1):
            title = book.get("title", "كتاب")
            authors = book.get("authors", "مؤلف")
            lines.append(f"{i}. {title} - {authors}")
        
        lines.append("أي منها يلفت انتباهك؟")
        return "\n".join(lines)
    
    else:
        lines = ["Great! Here are some books you might like:"]
        
        for i, book in enumerate(books[:3], 1):
            title = book.get("title", "Book")
            authors = book.get("authors", "Author")
            lines.append(f"{i}. {title} by {authors}")
        
        lines.append("Which one catches your eye?")
        return "\n".join(lines)
    
def generate_broader_search_terms(preferences: Dict) -> List[str]:
    """توليد مصطلحات بحث أوسع"""
    
    prompt = f"""
    Based on these user preferences, suggest 5 broader or related search terms for finding books:
    
    User wants: {preferences.get('wants', [])}
    
    For example, if user wants "pharaohs history", broader terms could be:
    - ancient egypt civilization
    - egyptian archaeology
    - history of ancient egypt
    - egyptian pharaohs and pyramids
    - ancient world history
    
    Return ONLY a JSON array of strings, nothing else.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.4
        )
        
        result = json.loads(response.choices[0].message.content)
        if isinstance(result, dict) and "terms" in result:
            return result["terms"]
        elif isinstance(result, list):
            return result
        else:
            return []
            
    except Exception as e:
        print(f"❌ Broad terms generation failed: {e}")
        return []
    
def format_conversation(history: List[Dict]) -> str:
    """تنسيق المحادثة للنموذج"""
    formatted = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)

def process_user_message(message: str, sid: str, session: Dict):
    """معالجة رسالة المستخدم باستخدام النظام الذكي - كاملة"""
    # تنظيف النص
    user_text = message.strip()
    
    # تحديث سجل المحادثة
    update_conversation_history(session, "user", user_text)
    
    # الحصول على آخر رسالة للمساعد
    last_assistant_msg = get_last_assistant_message(session["conversation_history"])
    
    # حفظ تفضيلات ذكي
    save_user_preference_smart(session, user_text, last_assistant_msg)
    
    # كشف اللغة
    lang_info = detect_and_normalize_language(user_text)
    normalized_lang = lang_info["normalized"]
    
    # التحقق إذا كان ردًا على التوصيات (للتعامل مع الردود السلبية)
    if session.get("recommended") and last_assistant_msg:
        feedback_info = check_feedback_on_recommendations(
            user_text, session, last_assistant_msg
        )
        
        # معالجة الرد السلبي على التوصيات
        if feedback_info.get("is_negative_feedback"):
            return handle_negative_feedback(user_text, sid, session, normalized_lang)
    
    # التحقق من الإجابة على سؤال اللغة
    language_response_result = check_language_response(
        user_text, session, last_assistant_msg, normalized_lang
    )
    if language_response_result:
        return language_response_result
    
    # التحقق إذا كان الوقت مناسبًا للتوصيات
    need_recommend = check_if_ready_for_recommendations(session)
    
    if need_recommend:
        # استخدم النظام الذكي للتوصيات
        return smart_recommendation_system(sid, session, normalized_lang)
    else:
        # استخدم النظام الذكي للمتابعة
        return generate_smart_follow_up(sid, session, normalized_lang)

def save_user_preference_smart(session: Dict, user_text: str, last_assistant_msg: Optional[str]):
    """حفظ تفضيلات ذكية باستخدام LLM"""
    # لا تحفظ نصوص قصيرة جداً
    if len(user_text.strip().split()) < 2:
        print(f"⏩ Skipping short text: '{user_text}'")
        return
    
    # لا تحفظ إجابات على أسئلة اللغة
    if last_assistant_msg and any(phrase in last_assistant_msg.lower() for phrase in [
        "which language", "أي لغة", "arabic or english", "العربية أم الإنجليزية"
    ]):
        if any(word in user_text.lower() for word in ["arabic", "english", "عربي", "انجليزي"]):
            print(f"⏩ Skipping language response: '{user_text}'")
            # لكن احفظ اللغة
            if "arabic" in user_text.lower() or "عربي" in user_text.lower():
                session["preferred_reading_lang"] = "ar"
            elif "english" in user_text.lower() or "انجليزي" in user_text.lower():
                session["preferred_reading_lang"] = "en"
            return
    
    # استخدام LLM للتحقق إذا كان النص تفضيلاً حقيقياً
    try:
        prompt = f"""
        Determine if this user message contains book preferences or interests that should be saved for book recommendations.
        
        Message: "{user_text}"
        Context: Last assistant message was: "{last_assistant_msg[:100] if last_assistant_msg else 'None'}"
        
        Respond with ONLY "yes" or "no".
        
        Save as preference if:
        - User mentions genres, topics, or types of books they like
        - User describes what they're looking for in a book
        - User shares reading preferences
        
        Do NOT save if:
        - It's a simple greeting or acknowledgment
        - It's a response to a specific question without new preferences
        - It's negative feedback about previous recommendations
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )
        
        should_save = response.choices[0].message.content.strip().lower() == "yes"
        
        if should_save:
            pref_key = f"pref_{len(session['user_prefs']) + 1}"
            session["user_prefs"][pref_key] = user_text
            print(f"🤖 LLM decided to save preference {pref_key}: '{user_text[:50]}...'")
        else:
            print(f"🤖 LLM decided NOT to save: '{user_text}'")
            
    except Exception as e:
        print(f"❌ LLM preference check failed: {e}")
        # Fallback إلى القواعد البسيطة
        if should_save_preference_fallback(user_text, last_assistant_msg):
            pref_key = f"pref_{len(session['user_prefs']) + 1}"
            session["user_prefs"][pref_key] = user_text
            print(f"💾 Fallback saved preference {pref_key}: '{user_text[:50]}...'")

def should_save_preference_fallback(user_text: str, last_assistant_msg: Optional[str]) -> bool:
    """Fallback للتحقق إذا كان يجب حفظ النص كتفضيل"""
    user_text = user_text.strip()
    
    # لا تحفظ نصوص فارغة أو قصيرة جداً
    if not user_text or len(user_text) < 3:
        return False
    
    # لا تحفظ إجابات على أسئلة محددة
    if last_assistant_msg:
        last_msg_lower = last_assistant_msg.lower()
        
        # إذا كان سؤال عن اللغة
        if any(phrase in last_msg_lower for phrase in [
            "which language", "أي لغة", "arabic or english", "العربية أم الإنجليزية"
        ]):
            return False
        
        # إذا كان سؤال نعم/لا بسيط
        if last_assistant_msg.endswith("?") and len(user_text.split()) < 3:
            simple_questions = ["yes", "no", "y", "n", "ok", "okay", "نعم", "لا", "حسنا", "طيب"]
            if user_text.lower() in simple_questions:
                return False
    
    # التحقق من نوع المحتوى
    words = user_text.lower().split()
    
    # لا تحفظ محتوى غير متعلق بالكتب
    unrelated_keywords = [
        "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye",
        "مرحبا", "اهلا", "شكرا", "مع السلامة", "وداعا"
    ]
    
    if any(keyword in user_text.lower() for keyword in unrelated_keywords):
        return False
    
    # تحقق إذا كان النص يحتوي على كلمات متعلقة بالكتب
    book_related_keywords = [
        # English
        "book", "books", "novel", "story", "stories", "read", "reading",
        "genre", "author", "fiction", "non-fiction", "history", "historical",
        "romance", "mystery", "fantasy", "science", "biography", "poetry",
        "adventure", "crime", "thriller", "horror", "classic", "modern",
        
        # Arabic
        "كتاب", "كتب", "رواية", "قصة", "قصص", "قراءة", "مطالعة",
        "نوع", "مؤلف", "خيال", "واقعي", "تاريخ", "تاريخي",
        "رومانسي", "غموض", "خيال علمي", "سيرة", "شعر", "مغامرة",
        "جريمة", "إثارة", "رعب", "كلاسيكي", "حديث"
    ]
    
    # إذا كان النص يحتوي على كلمات متعلقة بالكتب
    has_book_terms = any(term in user_text.lower() for term in book_related_keywords)
    
    # أو إذا كان النص طويلاً بما فيه الكفاية
    is_long_enough = len(words) >= 3
    
    # أو إذا كان يحتوي على وصف أو طلب
    has_request = any(word in user_text.lower() for word in ["want", "need", "looking for", "أريد", "أحتاج", "أبحث عن"])
    
    return has_book_terms or is_long_enough or has_request
           
def check_if_ready_for_recommendations(session: Dict) -> bool:
    """التحقق إذا كان جاهزاً للتوصيات باستخدام تحليل ذكي"""
    # الشرط الأساسي: 3 تفضيلات ذات معنى
    meaningful_prefs = [
        pref for pref in session["user_prefs"].values() 
        if len(pref.split()) > 1
    ]
    
    if len(meaningful_prefs) >= 3:
        return True
    
    # أو إذا كان لدى المستخدم طلب محدد
    last_user_msg = None
    for msg in reversed(session["conversation_history"]):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break
    
    if last_user_msg:
        # تحقق إذا كان الطلب يحتوي على كلمات مفتاحية
        keywords = ["need", "want", "looking for", "أريد", "أحتاج", "أبحث عن"]
        book_terms = ["book", "books", "novel", "story", "كتاب", "كتب", "رواية"]
        
        has_request = any(kw in last_user_msg.lower() for kw in keywords)
        has_book_ref = any(term in last_user_msg.lower() for term in book_terms)
        
        if has_request and has_book_ref:
            return True
    
    return False

def generate_smart_follow_up(sid: str, session: Dict, language: str):
    """توليد سؤال متابعة ذكي باستخدام LLM"""
    
    prompt = f"""
    You're helping someone find books. Based on this conversation, ask ONE natural follow-up question 
    to understand their book preferences better.
    
    Conversation history (recent):
    {format_conversation(session["conversation_history"][-4:])}
    
    Ask a question that:
    1. Is relevant to what they've already said
    2. Helps narrow down book recommendations
    3. Is conversational and friendly
    4. In {language} language
    
    Return ONLY the question, nothing else.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        
        question = response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"❌ Smart follow-up failed: {e}")
        # Fallback
        if language == "ar":
            question = "ما نوع الكتب التي تستمتع بقراءتها عادة؟"
        else:
            question = "What type of books do you usually enjoy reading?"
    
    update_conversation_history(session, "assistant", question)
    touch_session(sid)
    
    response = {
        "session_id": sid,
        "reply": question,
        "books": [],
        "follow_up": True
    }
    
    log_response("Smart Follow-up", response)
    return response
def generate_contextual_followup(conversation_history: List[Dict], user_lang: str, is_after_recommendation: bool = False) -> str:
    """بتولد أسئلة متابعة ذكية بناءً على context المحادثة"""
    # نجمع معلومات من المحادثة
    history_text = "\n".join([f"{h['role']}: {h['content']}" for h in conversation_history[-6:]])  # آخر 3 تبادلات
    
    if is_after_recommendation:
        prompt = f"""
        Based on this conversation, generate ONE natural follow-up question to understand why the user might not be satisfied with the recommendations and what they'd prefer instead.
        Conversation: {history_text}
        Requirements:
        - Ask ONE question only
        - Be curious and helpful, not repetitive
        - Focus on understanding their specific taste better
        - Respond in {user_lang}
        """
    else:
        prompt = f"""
        Based on this conversation, generate ONE natural follow-up question that helps understand the user's book preferences better.
        Conversation: {history_text}
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
        import random
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
        
        questions = fallback_questions_ar if user_lang == "ar" else fallback_questions_en
        return random.choice(questions)
def generate_no_books_smart_response(sid: str, session: Dict, preferences: Dict, language: str):
    """إنشاء رد ذكي عندما لا توجد كتب"""
    
    prompt = f"""
    You're a helpful librarian. You couldn't find books matching the user's preferences.
    
    User wants: {preferences.get('wants', [])}
    User language: {language}
    
    Write a helpful message that:
    1. Apologizes briefly for not finding matching books
    2. Suggests alternative approaches (e.g., try different keywords, search in other language)
    3. Asks if they'd like to adjust their preferences
    4. Ends with an encouraging note
    
    Keep it friendly and helpful. Write in {language}.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        reply = response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"❌ Smart no-books response failed: {e}")
        if language == "ar":
            reply = "عذراً، لم أجد كتباً تطابق تفضيلاتك. هل تريد تعديل شروط البحث أو البحث باللغة الإنجليزية؟"
        else:
            reply = "Sorry, I couldn't find books matching your preferences. Would you like to adjust your search or try English books?"
    
    update_conversation_history(session, "assistant", reply)
    touch_session(sid)
    
    return {
        "session_id": sid,
        "reply": reply,
        "books": [],
        "follow_up": True
    }
def should_save_preference(user_text: str, session: Dict) -> bool:
    """التحقق إذا كان يجب حفظ النص كتفضيل (مبسط)"""
    user_text = user_text.strip()
    
    # لا تحفظ نصوص قصيرة جداً
    if len(user_text.split()) < 2:
        return False
    
    # لا تحفظ إجابات على أسئلة محددة
    last_assistant_msg = get_last_assistant_message(session["conversation_history"])
    if last_assistant_msg:
        last_msg_lower = last_assistant_msg.lower()
        # إذا كان سؤال عن اللغة
        if any(phrase in last_msg_lower for phrase in [
            "which language", "أي لغة", "arabic or english"
        ]):
            return False
        # إذا كان سؤال متابعة بسيط
        if "?" in last_assistant_msg and len(user_text.split()) < 4:
            return False
    
    return True

def save_user_preference_simple(session: Dict, user_text: str):
    """حفظ تفضيل مبسط"""
    if not user_text or len(user_text.strip()) < 3:
        return
    
    # تحقق إذا كانت هذه المعلومة جديدة
    existing_text = " ".join(session["user_prefs"].values()).lower()
    if user_text.lower() not in existing_text:
        pref_key = f"pref_{len(session['user_prefs']) + 1}"
        session["user_prefs"][pref_key] = user_text
        print(f"💾 Saved simple preference: '{user_text[:50]}...'")
def extract_keywords_fallback(conversation_history: List[Dict]) -> List[str]:
    """استخراج كلمات مفتاحية كـ fallback"""
    keywords = []
    for msg in conversation_history[-4:]:  # آخر 4 رسائل
        if msg["role"] == "user":
            text = msg["content"].lower()
            # كلمات مفتاحية شائعة
            common_keywords = [
                "book", "novel", "story", "history", "historical", 
                "fiction", "non-fiction", "arabic", "english",
                "pharaoh", "egypt", "ancient", "قراءة", "كتاب",
                "رواية", "قصة", "تاريخ", "فرعون", "مصر"
            ]
            
            for keyword in common_keywords:
                if keyword in text:
                    keywords.append(keyword)
    
    return list(set(keywords))[:5]  # إزالة التكرارات
def filter_books_with_llm(books: List[Dict], preferences: Dict, language: str, top_k: int = 5) -> List[Dict]:
    """فلترة الكتب باستخدام LLM لتحديد الأنسب"""
    
    # إذا كانت الكتب قليلة، لا نحتاج فلترة
    if len(books) <= top_k:
        return books[:top_k]
    
    # إذا كان client غير متاح، استخدم فلترة بسيطة
    if client is None:
        print("⚠️ OpenAI client not available, using simple filtering")
        # فلترة حسب اللغة أولاً
        lang_filtered = filter_books_by_language(books, language)
        return lang_filtered[:top_k] if lang_filtered else books[:top_k]
    
    # إعداد البيانات للـ LLM - مع معالجة القيم None
    books_info = []
    for i, book in enumerate(books[:30]):  # حد 30 كتاب للتحليل
        # معالجة القيم None
        title = book.get("title", "Unknown Title") or "Unknown Title"
        authors = book.get("authors", "Unknown Author") or "Unknown Author"
        summary = book.get("short_summary", "") or ""
        
        books_info.append({
            "id": i,
            "title": title,
            "authors": authors,
            "summary": summary[:300] if summary else "",  # ✅ معالجة None
            "language": book.get("language", ""),
            "category": book.get("library_location", "").split("–")[0] if book.get("library_location") else ""
        })
    
    filter_prompt = f"""
    Select the most relevant books based on user preferences.
    
    USER PREFERENCES:
    - Wants: {preferences.get('wants', [])}
    - Does NOT want: {preferences.get('does_not_want', [])}
    - Language: {language}
    
    AVAILABLE BOOKS (ID, Title, Authors, Summary):
    {json.dumps(books_info, ensure_ascii=False)}
    
    TASK:
    1. Analyze each book against user preferences
    2. Select {top_k} books that best match what the user wants
    3. AVOID books that match what the user does NOT want
    4. Prioritize books in {language} if available
    
    Return a JSON array of book IDs (numbers only).
    Example: [3, 7, 1, 9, 4]
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": filter_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=200
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # استخراج IDs من الرد
        selected_ids = []
        if isinstance(result, dict):
            # جرب مفاتيح مختلفة
            for key in ["selected_books", "books", "ids", "selected_ids", "recommendations"]:
                if key in result and isinstance(result[key], list):
                    selected_ids = result[key]
                    break
        elif isinstance(result, list):
            selected_ids = result
        
        print(f"🤖 LLM Selected Book IDs: {selected_ids}")
        
        # استرجاع الكتب المختارة مع التحقق
        selected_books = []
        for idx in selected_ids[:top_k]:
            if isinstance(idx, int) and 0 <= idx < len(books):
                selected_books.append(books[idx])
            elif isinstance(idx, str) and idx.isdigit():
                idx_int = int(idx)
                if 0 <= idx_int < len(books):
                    selected_books.append(books[idx_int])
        
        # إذا لم يتم اختيار أي كتب، استخدم أول top_k كتب
        if not selected_books:
            print("⚠️ LLM returned no valid selections, using top books")
            selected_books = books[:top_k]
        
        return selected_books
        
    except Exception as e:
        print(f"❌ LLM filtering failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: فلترة بسيطة حسب اللغة
        lang_filtered = filter_books_by_language(books, language)
        return lang_filtered[:top_k] if lang_filtered else books[:top_k]


def intelligent_book_search(user_query: str, preferences_analysis: Dict, language: str, top_k: int = 5):
    """بحث ذكي عن الكتب باستخدام LLM لتحسين الاستعلام"""
    
    # إذا كان client غير متاح
    if client is None:
        print("⚠️ OpenAI client not available, using regular search")
        results = find_top_k(user_query, k=top_k)
        # فلترة حسب اللغة
        return filter_books_by_language(results, language)[:top_k]
    
    # Step 1: تحسين الاستعلام باستخدام LLM
    try:
        query_optimization_prompt = f"""
        Create an optimized search query for finding books.
        
        User wants: {preferences_analysis.get('wants', [])}
        User query: "{user_query}"
        Language: {language}
        
        Create a concise search query in {language} that includes:
        - Main topic/keywords
        - Related terms
        - Genre/style
        
        Return ONLY the query string.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query_optimization_prompt}],
            temperature=0.2,
            max_tokens=50
        )
        
        optimized_query = response.choices[0].message.content.strip()
        print(f"🔍 LLM Optimized Query: '{optimized_query}'")
        
    except Exception as e:
        print(f"❌ Query optimization failed: {e}")
        optimized_query = user_query
    
    # Step 2: البحث باستخدام الاستعلام المحسن
    print(f"🔍 Searching with: '{optimized_query}'")
    try:
        initial_results = find_top_k(optimized_query, k=top_k * 3)  # أحضر أكثر
    except Exception as e:
        print(f"❌ Search failed: {e}")
        initial_results = []
    
    if not initial_results:
        print(f"❌ No results with optimized query, trying original...")
        try:
            initial_results = find_top_k(user_query, k=top_k * 2)
        except Exception as e:
            print(f"❌ Original search also failed: {e}")
            return []
    
    print(f"📚 Found {len(initial_results)} initial results")
    
    # Step 3: تنظيف البيانات - إزالة الكتب التي بها مشاكل
    clean_books = []
    for book in initial_results:
        # تأكد أن الكتاب له عنوان على الأقل
        if book.get("title"):
            # معالجة القيم None
            if book.get("short_summary") is None:
                book["short_summary"] = ""
            clean_books.append(book)
    
    if not clean_books:
        print("❌ No valid books found after cleaning")
        return []
    
    # Step 4: فلترة باستخدام LLM (إذا كانت هناك نتائج كافية)
    if len(clean_books) > top_k:
        filtered_books = filter_books_with_llm(clean_books, preferences_analysis, language, top_k)
        print(f"✅ LLM filtered to {len(filtered_books)} books")
        return filtered_books
    else:
        print(f"✅ Using all {len(clean_books)} books (not enough for filtering)")
        return clean_books[:top_k]
    
def generate_intelligent_explanation(books: List[Dict], preferences: Dict, language: str) -> str:
    """توليد شرح ذكي للتوصيات باستخدام LLM"""
    
    # إعداد معلومات الكتب مع معالجة القيم None
    books_info = []
    for book in books:
        title = book.get("title", "Unknown Book") or "Unknown Book"
        authors = book.get("authors", "Unknown Author") or "Unknown Author"
        summary = book.get("short_summary", "") or ""
        
        books_info.append({
            "title": title,
            "authors": authors,
            "summary": summary[:200] if summary else "",
            "language": book.get("language", "")
        })
    
    prompt = f"""
    You are a knowledgeable librarian helping a user find books.
    
    User Preferences Summary:
    {preferences.get('summary', 'Looking for interesting books')}
    
    User specifically wants: {preferences.get('wants', [])}
    User wants to avoid: {preferences.get('does_not_want', [])}
    
    Here are the books you're recommending:
    {json.dumps(books_info, ensure_ascii=False)}
    
    Write a personalized recommendation message that:
    1. Starts with a warm, engaging introduction
    2. Briefly explains why each book matches their preferences
    3. Highlights what makes each book special
    4. Ends with an open-ended question to continue the conversation
    
    Write in {language}, keep it conversational and friendly.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"❌ Intelligent explanation failed: {e}")
        # Fallback تقليدي
        try:
            return generate_recommendation_explanation(
                preferences.get('summary', 'User preferences'),
                books,
                language
            )
        except:
            # Fallback أبسط
            if language == "ar":
                return "هذه بعض الكتب التي قد تناسب اهتماماتك. أتمنى أن تجد ما تبحث عنه!"
            else:
                return "Here are some books that might match your interests. I hope you find what you're looking for!"

def clean_book_data(book: Dict) -> Dict:
    """تنظيف بيانات الكتاب - معالجة القيم None"""
    cleaned = book.copy()
    
    # معالجة جميع الحقول النصية
    text_fields = ["title", "authors", "short_summary", "publisher", "library_location"]
    for field in text_fields:
        if field in cleaned and cleaned[field] is None:
            cleaned[field] = ""
    
    if not cleaned.get("title"):
        cleaned["title"] = "Unknown Book"
    
    return cleaned

def smart_recommendation_system(sid: str, session: Dict, normalized_lang: str):
    """نظام توصيات ذكي يعتمد على LLM"""
    print("🧠 [Stage] Using intelligent recommendation system...")
    
    try:
        # Step 1: تحليل التفضيلات باستخدام LLM
        preferences_analysis = analyze_preferences_with_llm(session["conversation_history"])
        
        # تحديث لغة الجلسة إذا تم اكتشافها
        detected_lang = preferences_analysis.get("language", normalized_lang)
        if detected_lang and "preferred_reading_lang" not in session:
            session["preferred_reading_lang"] = detected_lang
        
        # Step 2: بناء استعلام ذكي
        meaningful_prefs = [
            pref for pref in session["user_prefs"].values() 
            if pref and len(pref.split()) > 1
        ]
        
        user_query = " ".join(meaningful_prefs) if meaningful_prefs else "books"
        
        # Step 3: البحث الذكي
        reading_lang = session.get("preferred_reading_lang", detected_lang)
        matched_books = intelligent_book_search(
            user_query, 
            preferences_analysis, 
            reading_lang
        )
        
        # Step 4: تنظيف بيانات الكتب
        for book in matched_books:
            book = clean_book_data(book)
            ensure_cover(book)
        
        # Step 5: إذا لم توجد نتائج، حاول استراتيجيات مختلفة
        if not matched_books:
            print("🔄 No books found, trying alternative strategies...")
            
            # استراتيجية 1: البحث باستخدام كلمات أعمق
            broader_search_terms = generate_broader_search_terms(preferences_analysis)
            for term in broader_search_terms[:3]:
                matched_books = intelligent_book_search(term, preferences_analysis, reading_lang)
                if matched_books:
                    break
            
            # استراتيجية 2: البحث باللغة الأخرى
            if not matched_books and reading_lang == "ar":
                print("🔄 Trying English books as fallback...")
                matched_books = intelligent_book_search(
                    user_query, 
                    preferences_analysis, 
                    "en"
                )
        
        # Step 6: توليد الشرح الذكي
        if matched_books:
            explanation = generate_intelligent_explanation(
                matched_books, 
                preferences_analysis, 
                normalized_lang
            )
            
            # تحديث المحادثة
            update_conversation_history(session, "assistant", explanation)
            session["recommended"] = True
            touch_session(sid)
            
            response = {
                "session_id": sid,
                "reply": explanation,
                "books": matched_books,
                "follow_up": True,
            }
            
            log_response("Intelligent Recommendation", response)
            return response
        
        else:
            # إذا لم توجد كتب على الإطلاق
            return generate_no_books_smart_response(sid, session, preferences_analysis, normalized_lang)
            
    except Exception as e:
        print(f"❌ Error in smart recommendation system: {e}")
        import traceback
        traceback.print_exc()
        
        # رد خطأ
        if normalized_lang == "ar":
            reply = "عذراً، حدث خطأ في نظام التوصيات. يرجى المحاولة مرة أخرى."
        else:
            reply = "Sorry, there was an error in the recommendation system. Please try again."
        
        update_conversation_history(session, "assistant", reply)
        touch_session(sid)
        
        return {
            "session_id": sid,
            "reply": reply,
            "books": [],
            "follow_up": True
        }
# ==================== EXTERNAL DEPENDENCIES (PLACEHOLDERS) ====================
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect

load_dotenv() 

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
# def embed_text(text: str):
#     resp = client.embeddings.create(model=EMBED_MODEL, input=text)
#     return resp.data[0].embedding

def embed_text(text: str):
    """توليد embedding للنص - مع معالجة الأخطاء"""
    if not text or not text.strip():
        print("⚠️ Empty text for embedding")
        # إرجاع embedding فارغ بالحجم الصحيح
        if embeddings is not None and len(embeddings) > 0:
            return [0.0] * embeddings.shape[1]
        else:
            return [0.0] * 1536  # الحجم الافتراضي
    
    if client is None:
        print("⚠️ OpenAI client not available for embedding")
        if embeddings is not None and len(embeddings) > 0:
            return [0.0] * embeddings.shape[1]
        else:
            return [0.0] * 1536
    
    try:
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=text
        )
        return resp.data[0].embedding
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        # Fallback embedding
        if embeddings is not None and len(embeddings) > 0:
            return [0.0] * embeddings.shape[1]
        else:
            return [0.0] * 1536
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

def find_top_k(query: str, k: int = TOP_K):
    """بحث عن أفضل K كتب مشابهة للاستعلام - مع معالجة الأخطاء"""
    
    if embeddings is None or len(embeddings) == 0:
        print("❌ No embeddings loaded")
        return []
    
    if not metas or len(metas) == 0:
        print("❌ No metadata loaded")
        return []
    
    try:
        # توليد embedding للاستعلام
        query_emb = np.array(embed_text(query), dtype=np.float32).reshape(1, -1)
        
        # حساب التشابه
        sims = cosine_similarity(query_emb, embeddings)[0]
        
        # الحصول على أعلى K مؤشرات
        top_indices = np.argsort(sims)[::-1][:k]
        
        results = []
        for idx in top_indices:
            idx_int = int(idx)
            if 0 <= idx_int < len(metas):
                m = metas[idx_int].copy()
                m["_score"] = float(sims[idx_int])
                results.append(m)
            else:
                print(f"⚠️ Warning: Index {idx_int} out of bounds for metadata (size {len(metas)})")
        
        print(f"🔍 Search for '{query[:50]}...' returned {len(results)} results")
        return results
        
    except Exception as e:
        print(f"❌ Error in find_top_k: {e}")
        import traceback
        traceback.print_exc()
        return []
    
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
# ==================== RUN APP ====================
# import uvicorn

# if __name__ == "__main__":
#     uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)