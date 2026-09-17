import streamlit as st
from openai import OpenAI
import html
import ast
import operator as op
import base64
import uuid
import io
import wave
import re


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Hub",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GROQ / OPENAI CLIENT
# ============================================================

try:
    client = OpenAI(
        api_key=st.secrets["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
except Exception:
    client = None

MODEL = "openai/gpt-oss-20b"

# The free Groq tier caps requests at ~8000 tokens (prompt + reply combined).
# Roughly 4 characters ≈ 1 token, so we keep pasted code/documents well
# under that limit and warn the user when we trim their input.
MAX_INPUT_CHARS = 10000


def truncate_text(text, max_chars=MAX_INPUT_CHARS):
    if len(text) > max_chars:
        return text[:max_chars], True
    return text, False


# For code that is too long even for MAX_INPUT_CHARS, we split it into
# smaller chunks (on line boundaries so we don't cut a line in half) and
# send each chunk as its own request instead of throwing away the rest.
CODE_CHUNK_CHARS = 12000


def split_into_chunks(text, chunk_size=CODE_CHUNK_CHARS):
    lines = text.split("\n")
    chunks = []
    current_lines = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for the newline

        if current_lines and current_len + line_len > chunk_size:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_len = 0

        current_lines.append(line)
        current_len += line_len

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


# ============================================================
# AI FUNCTION
# ============================================================

def ask_ai(user_text, system=None):
    if client is None:
        return "⚠️ GROQ_API_KEY is missing. Please check your Streamlit secrets."

    try:
        messages = []

        if system:
            messages.append(
                {
                    "role": "system",
                    "content": system,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )

        return response.choices[0].message.content

    except Exception as e:
        error_text = str(e)

        if "rate_limit_exceeded" in error_text or "413" in error_text or "too large" in error_text.lower():
            return (
                "⚠️ That text is too long for the AI to process in one go "
                "(the free tier has a small token limit per request).\n\n"
                "Please try again with a shorter piece of code/document, "
                "or split it into smaller parts and ask about each one separately."
            )

        return f"⚠️ Something went wrong talking to the AI:\n\n{error_text}"


def ask_ai_stream(user_text, system=None):
    """Generator version of ask_ai(): yields the reply text piece by piece
    as it arrives, instead of waiting for the full response."""

    if client is None:
        yield "⚠️ GROQ_API_KEY is missing. Please check your Streamlit secrets."
        return

    messages = []

    if system:
        messages.append(
            {
                "role": "system",
                "content": system,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except Exception as e:
        error_text = str(e)

        if "rate_limit_exceeded" in error_text or "413" in error_text or "too large" in error_text.lower():
            yield (
                "⚠️ That text is too long for the AI to process in one go "
                "(the free tier has a small token limit per request).\n\n"
                "Please try again with a shorter piece of code/document, "
                "or split it into smaller parts and ask about each one separately."
            )
        else:
            yield f"⚠️ Something went wrong talking to the AI:\n\n{error_text}"


# ============================================================
# VISION AI FUNCTION (for the "See" page)
# ============================================================

# Groq's multimodal (image-understanding) model. Groq rotates its vision
# lineup periodically — if this model is ever retired, check
# https://console.groq.com/docs/vision for the current model id.
# Groq's multimodal (image-understanding) models. Both are currently listed
# as "Preview" models by Groq, and preview-model access can vary by account,
# so we try qwen3.6 first and automatically fall back to qwen3.8 if the
# first one isn't available on this API key.
# See https://console.groq.com/docs/vision for the current model ids.
VISION_MODELS = ["qwen/qwen3.6-27b", "qwen/qwen3.8-27b"]


def ask_ai_vision(image_bytes, mime_type, user_text, system=None):
    if client is None:
        return "⚠️ GROQ_API_KEY is missing. Please check your Streamlit secrets."

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64_image}"

    messages = []

    if system:
        messages.append(
            {
                "role": "system",
                "content": system,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
            ],
        }
    )

    last_error = None

    for model_id in VISION_MODELS:

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
            )

            return response.choices[0].message.content

        except Exception as e:
            error_text = str(e)
            last_error = error_text

            if "rate_limit_exceeded" in error_text or "413" in error_text or "too large" in error_text.lower():
                return (
                    "⚠️ That image is too large for the AI to process "
                    "(the free tier has a small request-size limit).\n\n"
                    "Please try a smaller or more compressed image."
                )

            if "model_not_found" in error_text or "does not exist" in error_text:
                # Try the next model in VISION_MODELS instead of giving up.
                continue

            return f"⚠️ Something went wrong talking to the vision model:\n\n{error_text}"

    return (
        "⚠️ None of the vision models this app knows about "
        f"({', '.join(VISION_MODELS)}) are available on your Groq API key.\n\n"
        "Open https://console.groq.com/playground , pick a vision model "
        "from the model dropdown, and confirm you can chat with it there. "
        "Whichever model id works in the playground is the one to put in "
        "VISION_MODELS in this app.\n\n"
        f"Last error: {last_error}"
    )


# ============================================================
# TEXT-TO-SPEECH (voice output for AI answers)
# ============================================================

# Groq's Orpheus TTS only speaks English (or Arabic, with a different
# model) and limits each request to ~200 characters, so long answers are
# split into small chunks and the resulting audio clips are stitched
# together into one WAV file.
TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "troy"
TTS_CHUNK_CHARS = 190

# Only the first part of a very long answer is read aloud, so a big
# document summary doesn't trigger dozens of sequential API calls.
MAX_TTS_CHARS = 1500


def split_text_for_tts(text, chunk_size=TTS_CHUNK_CHARS):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(sentence), chunk_size):
                chunks.append(sentence[i:i + chunk_size])
            continue

        if current and len(current) + 1 + len(sentence) > chunk_size:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()

    if current:
        chunks.append(current)

    return chunks


def text_to_speech(text, voice=TTS_VOICE):
    if client is None:
        return None, "⚠️ GROQ_API_KEY is missing."

    plain_text = re.sub(r"[*_`#]", "", text or "").strip()

    if not plain_text:
        return None, None

    trimmed = False
    if len(plain_text) > MAX_TTS_CHARS:
        plain_text = plain_text[:MAX_TTS_CHARS]
        trimmed = True

    chunks = split_text_for_tts(plain_text)

    if not chunks:
        return None, None

    try:
        audio_segments = []

        for chunk in chunks:
            response = client.audio.speech.create(
                model=TTS_MODEL,
                voice=voice,
                input=chunk,
                response_format="wav",
            )
            audio_segments.append(response.content)

        with wave.open(io.BytesIO(audio_segments[0]), "rb") as first_clip:
            params = first_clip.getparams()

        combined = io.BytesIO()

        with wave.open(combined, "wb") as out_wav:
            out_wav.setparams(params)

            for segment in audio_segments:
                with wave.open(io.BytesIO(segment), "rb") as clip:
                    out_wav.writeframes(clip.readframes(clip.getnframes()))

        note = (
            "Only the first part of this answer was read aloud "
            "(long-answer limit)."
            if trimmed
            else None
        )

        return combined.getvalue(), note

    except Exception as e:
        return None, f"⚠️ Couldn't generate audio for this answer:\n\n{e}"


# ============================================================
# DISPLAY AI RESULT
# ============================================================

def show_ai_result(text, title=None):
    safe_text = html.escape(str(text or "")).replace("\n", "<br>")

    heading = (
        f"<b>{html.escape(title)}</b><br><br>"
        if title
        else ""
    )

    # Log this result to the sidebar history (capped so it never grows
    # without bound across a long session).
    if "history" in st.session_state and text and str(text).strip():
        st.session_state.history.append(
            {
                "title": title or "AI Result",
                "text": str(text),
            }
        )
        st.session_state.history = st.session_state.history[-50:]

    # Unique element id so the copy button copies only this result,
    # even when several results are shown on the same page.
    result_id = f"ai-result-{uuid.uuid4().hex[:8]}"

    st.markdown(
        f"""
        <div class="ai-result">
            {heading}
            <div id="{result_id}">{safe_text}</div>
            <button
                onclick="
                    navigator.clipboard.writeText(
                        document.getElementById('{result_id}').innerText
                    );
                    this.innerText = '✅ Copied';
                    setTimeout(() => {{ this.innerText = '📋 Copy'; }}, 1500);
                "
                class="copy-btn"
            >📋 Copy</button>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("tts_enabled", False) and text and str(text).strip():

        with st.spinner("Generating audio... 🔊"):
            audio_bytes, note = text_to_speech(text)

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")

        if note:
            st.caption(note)


def show_ai_result_stream(chunk_generator, title=None):
    """Like show_ai_result(), but takes a generator of text pieces and
    displays them progressively as they arrive, ChatGPT-style. Once the
    stream finishes, it re-renders the final answer through
    show_ai_result() so the Copy button, voice output, and history log
    all still work exactly as before."""

    heading = (
        f"<b>{html.escape(title)}</b><br><br>"
        if title
        else ""
    )

    placeholder = st.empty()

    placeholder.markdown(
        f'<div class="ai-result">{heading}<i>Thinking...</i></div>',
        unsafe_allow_html=True,
    )

    full_text = ""

    for piece in chunk_generator:
        full_text += piece
        safe_partial = html.escape(full_text).replace("\n", "<br>")
        placeholder.markdown(
            f"""
            <div class="ai-result">
                {heading}{safe_partial}<span class="typing-cursor">▌</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    placeholder.empty()

    show_ai_result(full_text, title)

    return full_text


# ============================================================
# DOCUMENT TEXT EXTRACTION
# ============================================================

def extract_text(file):
    name = file.name.lower()

    try:

        # TXT / CSV
        if name.endswith(".txt") or name.endswith(".csv"):
            return file.read().decode(
                "utf-8",
                errors="ignore"
            )

        # PDF
        if name.endswith(".pdf"):
            from pypdf import PdfReader

            reader = PdfReader(file)

            return "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        # DOCX
        if name.endswith(".docx"):
            import docx

            document = docx.Document(file)

            return "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
            )

        # XLSX
        if name.endswith(".xlsx"):
            import openpyxl

            workbook = openpyxl.load_workbook(
                file,
                data_only=True
            )

            lines = []

            for sheet in workbook.worksheets:

                lines.append(
                    f"--- Sheet: {sheet.title} ---"
                )

                for row in sheet.iter_rows(
                    values_only=True
                ):

                    values = [
                        str(cell)
                        for cell in row
                        if cell is not None
                    ]

                    if values:
                        lines.append(
                            " | ".join(values)
                        )

            return "\n".join(lines)

    except Exception as e:
        return f"[Could not read file: {e}]"

    return "[Unsupported file type]"


# ============================================================
# SAFE CALCULATOR
# ============================================================

ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def safe_calculate(expression):

    def evaluate(node):

        if isinstance(node, ast.Constant):

            if isinstance(node.value, (int, float)):
                return node.value

            raise ValueError("Invalid number")

        if isinstance(node, ast.BinOp):

            operator_type = type(node.op)

            if operator_type not in ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if operator_type is ast.Pow and abs(right) > 100:
                raise ValueError("Power too large")

            return ALLOWED_OPERATORS[operator_type](
                left,
                right
            )

        if isinstance(node, ast.UnaryOp):

            operator_type = type(node.op)

            if operator_type not in ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed")

            return ALLOWED_OPERATORS[operator_type](
                evaluate(node.operand)
            )

        raise ValueError("Invalid expression")

    tree = ast.parse(
        expression,
        mode="eval"
    )

    return evaluate(tree.body)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

.stApp {
    background: #0d0d12;
    color: #e6e3ec;
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #14121c 0%,
        #17111a 50%,
        #10151c 100%
    );
    border-right: 1px solid #26212f;
}

[data-testid="stSidebar"] * {
    color: #d9d5e3 !important;
}

.brand {
    font-size: 30px;
    font-weight: 900;
    color: #a892f0;
    text-align: center;
    margin-bottom: 5px;
}

.brand-sub {
    text-align: center;
    color: #8a83a0;
    font-size: 12px;
    margin-bottom: 25px;
}

.hero {
    background: linear-gradient(
        135deg,
        #211c33,
        #2b1a26,
        #142330
    );

    padding: 40px;
    border-radius: 28px;
    margin-bottom: 25px;
    border: 1px solid #2c2738;
}

.hero h1 {
    color: #f1eefb;
    font-size: 38px;
    font-weight: 900;
    margin-bottom: 8px;
}

.hero p {
    color: #b7b1c9;
    font-size: 16px;
}

.page-header {
    font-size: 34px;
    font-weight: 900;
    color: #f1eefb;
    margin-bottom: 5px;
}

.page-subtitle {
    color: #a39cb5;
    margin-bottom: 25px;
}

.card {
    padding: 24px;
    border-radius: 22px;
    margin-bottom: 18px;
    min-height: 145px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 8px 25px rgba(0,0,0,0.35);
}

.purple {
    background: #221b35;
}

.mint {
    background: #142d26;
}

.peach {
    background: #322418;
}

.pink {
    background: #331e29;
}

.blue {
    background: #16232f;
}

.yellow {
    background: #332e14;
}

.card-icon {
    font-size: 30px;
    margin-bottom: 8px;
}

.card-title {
    font-size: 19px;
    font-weight: 900;
    color: #f1eefb;
}

.card-text {
    color: #a39cb5;
    font-size: 13px;
    margin-top: 5px;
}

.ai-result {
    background: #17151f;
    border: 1px solid #2c2738;
    border-radius: 20px;
    padding: 24px;
    margin-top: 20px;
    color: #e6e3ec;
    line-height: 1.7;
    box-shadow: 0 8px 25px rgba(0,0,0,0.35);
}

.copy-btn {
    margin-top: 16px;
    padding: 8px 18px;
    border-radius: 12px;
    border: 1px solid #33304a;
    background: #1b1826;
    color: #e6e3ec;
    font-weight: 700;
    font-size: 13px;
    cursor: pointer;
}

.copy-btn:hover {
    border-color: #8a6be2;
    color: #c9b8ff;
}

.typing-cursor {
    display: inline-block;
    color: #a892f0;
    animation: blink-cursor 0.9s step-start infinite;
}

@keyframes blink-cursor {
    50% {
        opacity: 0;
    }
}

.info-box {
    background: #1c1830;
    border-left: 5px solid #8a6be2;
    padding: 18px;
    border-radius: 14px;
    color: #cfc9dd;
    margin: 18px 0;
}

.quote {
    background: #221822;
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    color: #d9b9d0;
    font-style: italic;
    margin: 20px 0;
}

.small-note {
    color: #8a83a0;
    font-size: 12px;
}

.stButton > button {
    border-radius: 14px;
    border: 1px solid #33304a;
    background: #1b1826;
    color: #e6e3ec;
    font-weight: 800;
}

.stButton > button:hover {
    border-color: #8a6be2;
    color: #c9b8ff;
}

textarea,
input {
    border-radius: 14px !important;
    background-color: #17151f !important;
    color: #e6e3ec !important;
}

@media (max-width: 768px) {

    .hero {
        padding: 25px;
    }

    .hero h1 {
        font-size: 28px;
    }

    .page-header {
        font-size: 28px;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

# List of uploaded documents for the Documents page, each item is a dict:
# {"name": "<filename>", "text": "<extracted text>"}
if "docs" not in st.session_state:
    st.session_state.docs = []

# Running history of AI results shown across all pages, each item is a
# dict: {"title": "...", "text": "..."}. Newest entries are appended last.
if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">✦ AI HUB</div>
        <div class="brand-sub">
            Your smart little AI companion ♡
        </div>
        """,
        unsafe_allow_html=True,
    )

    pages = [
        "Home",
        "See",
        "Study",
        "Solve",
        "Create",
        "Talk",
        "Translate",
        "Code",
        "Documents",
        "Workspace",
    ]

    for page in pages:

        if st.button(
            page,
            use_container_width=True,
            key=f"nav_{page}",
        ):

            st.session_state.page = page
            st.rerun()

    st.markdown("---")

    st.checkbox(
        "🔊 Read answers aloud",
        key="tts_enabled",
        help=(
            "When on, AI answers are also read aloud (English only, "
            "first ~1500 characters)."
        ),
    )

    st.markdown("---")

    if st.session_state.history:

        st.markdown("### 🕘 Recent History")

        recent_history = list(reversed(st.session_state.history[-5:]))

        for i, entry in enumerate(recent_history):

            preview = entry["text"].strip().replace("\n", " ")[:45]

            if len(entry["text"]) > 45:
                preview += "..."

            with st.expander(f"{entry['title']} — {preview}"):
                st.write(entry["text"])

        st.markdown("---")

    st.markdown(
        """
        <div class="small-note">
        ✨ Learn<br>
        💡 Solve<br>
        🎨 Create<br>
        💬 Talk<br>
        📄 Understand<br>
        💻 Code
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "Home":

    st.markdown(
        """
        <div class="hero">
            <h1>Hey there! 💗</h1>
            <p>
                What do you want to do today?
                Your little AI Hub is ready ✨
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = [
        (
            "See",
            "👁️",
            "Understand images and visual content.",
            "purple",
        ),
        (
            "Study",
            "📚",
            "Learn topics in a simple way.",
            "mint",
        ),
        (
            "Solve",
            "🧮",
            "Calculate and solve problems.",
            "peach",
        ),
        (
            "Create",
            "🎨",
            "Create ideas, captions and content.",
            "pink",
        ),
        (
            "Talk",
            "🎙️",
            "Talk with your AI assistant.",
            "blue",
        ),
        (
            "Translate",
            "🌍",
            "Translate text into different languages.",
            "yellow",
        ),
        (
            "Code",
            "💻",
            "Explain, debug and improve code.",
            "purple",
        ),
        (
            "Documents",
            "📄",
            "Ask questions about your files.",
            "mint",
        ),
        (
            "Workspace",
            "✨",
            "Organize your ideas and projects.",
            "pink",
        ),
    ]

    cols = st.columns(3)

    for i, (
        title,
        icon,
        description,
        color,
    ) in enumerate(cards):

        with cols[i % 3]:

            st.markdown(
                f"""
                <div class="card {color}">
                    <div class="card-icon">{icon}</div>
                    <div class="card-title">{title}</div>
                    <div class="card-text">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                f"Open {title} →",
                key=f"home_{title}",
                use_container_width=True,
            ):

                st.session_state.page = title
                st.rerun()

    st.markdown("### ⚡ Quick actions")

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        if st.button(
            "📷 Open Camera",
            use_container_width=True,
        ):
            st.session_state.page = "See"
            st.rerun()

    with q2:
        if st.button(
            "📁 Upload File",
            use_container_width=True,
        ):
            st.session_state.page = "Documents"
            st.rerun()

    with q3:
        if st.button(
            "💬 New Chat",
            use_container_width=True,
        ):
            st.session_state.page = "Talk"
            st.rerun()

    with q4:
        if st.button(
            "🎙️ Voice Input",
            use_container_width=True,
        ):
            st.session_state.page = "Talk"
            st.rerun()

    st.markdown("### 💭 Ask me anything")

    question = st.text_input(
        "Ask your question",
        placeholder="Type anything here...",
    )

    if st.button(
        "Ask AI ✨",
        use_container_width=True,
    ):

        if question.strip():

            show_ai_result_stream(
                ask_ai_stream(question),
                "AI says 💗",
            )

        else:
            st.warning(
                "Please type something first."
            )


# ============================================================
# SEE
# ============================================================

elif st.session_state.page == "See":

    st.markdown(
        '<div class="page-header">👁️ See</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Let AI help you understand images.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📷 Camera")

        camera_image = (
            st.camera_input("Take a picture")
            if hasattr(st, "camera_input")
            else None
        )

    with col2:

        st.markdown("### 🖼️ Upload")

        uploaded_image = st.file_uploader(
            "Choose an image",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
            ],
        )

    source = (
        camera_image
        if camera_image is not None
        else uploaded_image
    )

    if source is not None:

        st.image(
            source,
            caption="Selected image",
            use_container_width=True,
        )

        vision_question = st.text_input(
            "What do you want to know about this image?",
            placeholder="Example: What is in this image?",
        )

        if st.button(
            "Analyze Image ✨",
            use_container_width=True,
        ):

            if vision_question.strip():

                with st.spinner(
                    "Looking at your image... 👁️"
                ):

                    image_bytes = source.getvalue()
                    mime_type = getattr(source, "type", None) or "image/png"

                    result = ask_ai_vision(
                        image_bytes,
                        mime_type,
                        vision_question,
                        system=(
                            "You are a helpful visual assistant. Look at "
                            "the image carefully and answer the user's "
                            "question about it accurately and concisely."
                        ),
                    )

                show_ai_result(
                    result,
                    "Image Assistant 👁️",
                )

            else:
                st.warning(
                    "Please enter a question about this image."
                )

    else:

        st.markdown(
            """
            <div class="info-box">
                📸 Take a photo or upload an image to get started.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# STUDY
# ============================================================

elif st.session_state.page == "Study":

    st.markdown(
        '<div class="page-header">📚 Study</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Learn anything in a simple way.</div>',
        unsafe_allow_html=True,
    )

    topic = st.text_input(
        "What do you want to study?",
        placeholder="Example: DBMS normalization",
    )

    study_style = st.selectbox(
        "Choose your learning style",
        [
            "Simple explanation",
            "Short notes",
            "Quiz",
            "Flashcards",
            "Exam answer",
        ],
    )

    if st.button(
        "Learn with AI ✨",
        use_container_width=True,
    ):

        if topic.strip():

            prompt = f"""
Teach me the following topic:

Topic:
{topic}

Learning style:
{study_style}

Keep the explanation clear, simple,
student-friendly and easy to remember.
"""

            show_ai_result_stream(
                ask_ai_stream(
                    prompt,
                    system=(
                        "You are a patient and encouraging tutor. "
                        "Explain concepts clearly for a college student."
                    ),
                ),
                "Your Study Notes 📚",
            )

        else:
            st.warning(
                "Please enter a topic first."
            )


# ============================================================
# SOLVE
# ============================================================

elif st.session_state.page == "Solve":

    st.markdown(
        '<div class="page-header">🧮 Solve</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Quickly solve mathematical expressions.</div>',
        unsafe_allow_html=True,
    )

    expression = st.text_input(
        "Enter expression",
        placeholder="Example: (25 + 5) * 2 / 3",
    )

    if st.button(
        "Calculate ✨",
        use_container_width=True,
    ):

        if expression.strip():

            try:

                result = safe_calculate(
                    expression
                )

                show_ai_result(
                    f"{expression} = {result}",
                    "Answer 🧮",
                )

            except Exception as e:

                st.error(
                    f"Could not calculate: {e}"
                )

        else:

            st.warning(
                "Please enter an expression."
            )

    st.markdown(
        """
        <div class="info-box">
            Allowed operators:
            + &nbsp; - &nbsp; * &nbsp; / &nbsp; % &nbsp; **
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CREATE
# ============================================================

elif st.session_state.page == "Create":

    st.markdown(
        '<div class="page-header">🎨 Create</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Turn your ideas into something useful.</div>',
        unsafe_allow_html=True,
    )

    create_kind = st.selectbox(
        "What do you want to create?",
        [
            "Idea",
            "Caption",
            "Email",
            "Presentation outline",
            "Project description",
        ],
    )

    create_prompt = st.text_area(
        "Tell me what you need",
        placeholder="Describe your idea...",
        height=180,
    )

    if st.button(
        "Create with AI ✨",
        use_container_width=True,
    ):

        if create_prompt.strip():

            full_prompt = f"""
Create the following:

Type:
{create_kind}

User request:
{create_prompt}

Make it useful, clear and polished.
"""

            show_ai_result_stream(
                ask_ai_stream(
                    full_prompt,
                    system=(
                        "You are a creative and helpful writing assistant."
                    ),
                ),
                "Created for you ✨",
            )

        else:

            st.warning(
                "Please describe what you want to create."
            )


# ============================================================
# TALK
# ============================================================

elif st.session_state.page == "Talk":

    st.markdown(
        '<div class="page-header">🎙️ Talk</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Talk to your AI assistant.</div>',
        unsafe_allow_html=True,
    )

    if hasattr(st, "audio_input"):

        talk_language = st.selectbox(
            "Spoken language (helps accuracy, optional)",
            [
                "Auto-detect",
                "English",
                "Telugu",
                "Hindi",
                "Tamil",
                "Kannada",
            ],
        )

        LANGUAGE_CODES = {
            "English": "en",
            "Telugu": "te",
            "Hindi": "hi",
            "Tamil": "ta",
            "Kannada": "kn",
        }

        audio = st.audio_input(
            "Record your voice"
        )

        if audio is not None:

            st.audio(audio)

            if st.button(
                "Convert Voice to Text ✨",
                use_container_width=True,
            ):

                if client is None:

                    st.error(
                        "GROQ_API_KEY is missing."
                    )

                else:

                    try:

                        with st.spinner(
                            "Listening... 🎧"
                        ):

                            audio_bytes = audio.getvalue()

                            transcription_kwargs = {
                                "model": "whisper-large-v3",
                                "file": (
                                    "recording.wav",
                                    audio_bytes,
                                    "audio/wav",
                                ),
                            }

                            language_code = LANGUAGE_CODES.get(
                                talk_language
                            )

                            if language_code:
                                transcription_kwargs["language"] = language_code

                            transcript = client.audio.transcriptions.create(
                                **transcription_kwargs
                            )

                        text = transcript.text

                        show_ai_result(
                            text,
                            "You said 🎙️",
                        )

                        show_ai_result_stream(
                            ask_ai_stream(
                                text,
                                system=(
                                    "You are a friendly AI voice assistant. "
                                    "Reply naturally and helpfully, and always "
                                    "reply in the same language the user just "
                                    "spoke in (for example, if they spoke in "
                                    "Telugu, reply in Telugu; if in Hindi, "
                                    "reply in Hindi; if in English, reply in "
                                    "English)."
                                ),
                            ),
                            "AI Reply 💗",
                        )

                    except Exception as e:

                        st.error(
                            f"Voice processing failed: {e}"
                        )

    else:

        st.info(
            "Voice input is not available "
            "in this Streamlit version."
        )

    st.markdown(
        "### 💬 Or type your message"
    )

    talk_text = st.text_area(
        "Message",
        placeholder="Say something to your AI...",
        height=120,
    )

    if st.button(
        "Send Message ✨",
        use_container_width=True,
    ):

        if talk_text.strip():

            show_ai_result_stream(
                ask_ai_stream(
                    talk_text,
                    system=(
                        "You are a friendly and helpful AI assistant. "
                        "Always reply in the same language the user just "
                        "typed in (for example, if they typed in Telugu, "
                        "reply in Telugu; if in Hindi, reply in Hindi; if "
                        "in English, reply in English)."
                    ),
                ),
                "AI Reply 💗",
            )

        else:

            st.warning(
                "Please type a message."
            )


# ============================================================
# TRANSLATE
# ============================================================

elif st.session_state.page == "Translate":

    st.markdown(
        '<div class="page-header">🌍 Translate</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Translate naturally and clearly.</div>',
        unsafe_allow_html=True,
    )

    translation_text = st.text_area(
        "Text to translate",
        placeholder="Enter your text...",
        height=180,
    )

    target_language = st.selectbox(
        "Translate to",
        [
            "English",
            "Telugu",
            "Hindi",
            "French",
            "German",
            "Spanish",
        ],
    )

    if st.button(
        "Translate ✨",
        use_container_width=True,
    ):

        if translation_text.strip():

            prompt = f"""
Translate the following text into {target_language}.

Text:
{translation_text}

Keep the meaning natural and accurate.
Do not add unnecessary explanation.
"""

            show_ai_result_stream(
                ask_ai_stream(
                    prompt,
                    system=(
                        "You are a precise and fluent translator."
                    ),
                ),
                f"Translation · {target_language}",
            )

        else:

            st.warning(
                "Please enter some text first."
            )


# ============================================================
# CODE
# ============================================================

elif st.session_state.page == "Code":

    st.markdown(
        '<div class="page-header">💻 Code</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Understand, debug and improve your code.</div>',
        unsafe_allow_html=True,
    )

    code = st.text_area(
        "Paste your code",
        placeholder=(
            "Paste your Python, Java, C++, Flutter "
            "or other code here..."
        ),
        height=350,
    )

    action = st.selectbox(
        "What should AI do?",
        [
            "Explain code",
            "Find bugs",
            "Improve code",
            "Generate test cases",
        ],
    )

    if st.button(
        "Analyze Code ✨",
        use_container_width=True,
    ):

        if code.strip():

            if len(code) <= CODE_CHUNK_CHARS:

                prompt = f"""
{action} for the following code:

```text
{code}
```
"""

                show_ai_result_stream(
                    ask_ai_stream(
                        prompt,
                        system=(
                            "You are an expert programming assistant. "
                            "Explain clearly and provide practical fixes."
                        ),
                    ),
                    "Code Assistant 💻",
                )

            else:

                chunks = split_into_chunks(code)

                st.info(
                    f"Your code is long, so it was split into "
                    f"{len(chunks)} parts and each part is being "
                    "analyzed separately (this avoids the free-tier "
                    "token limit and no part of your code is skipped)."
                )

                progress_bar = st.progress(
                    0,
                    text=f"Analyzing part 0 of {len(chunks)}...",
                )

                for i, chunk in enumerate(chunks, start=1):

                    part_prompt = f"""
This is part {i} of {len(chunks)} of a larger code file. The file was
split into parts only because it was too long to send in one request —
keep in mind this part may reference code defined in another part.

{action} for this part of the code:

```text
{chunk}
```
"""

                    progress_bar.progress(
                        (i - 1) / len(chunks),
                        text=f"Analyzing part {i} of {len(chunks)}... 💻",
                    )

                    result = ask_ai(
                        part_prompt,
                        system=(
                            "You are an expert programming assistant. "
                            "Explain clearly and provide practical fixes. "
                            "You are looking at one part of a larger file."
                        ),
                    )

                    show_ai_result(
                        result,
                        f"Code Assistant 💻 — Part {i}/{len(chunks)}",
                    )

                    progress_bar.progress(
                        i / len(chunks),
                        text=f"Finished part {i} of {len(chunks)} ✅",
                    )

        else:

            st.warning(
                "Please paste some code first."
            )


# ============================================================
# DOCUMENTS
# ============================================================

elif st.session_state.page == "Documents":

    st.markdown(
        '<div class="page-header">📄 Documents</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Understand one or more documents with AI.</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=[
            "txt",
            "csv",
            "pdf",
            "docx",
            "xlsx",
        ],
        accept_multiple_files=True,
        key="documents_uploader",
    )

    if uploaded_files:

        new_docs = []
        any_errors = False

        for uploaded_file in uploaded_files:

            text = extract_text(uploaded_file)

            if (
                text.startswith("[Could not read")
                or text == "[Unsupported file type]"
            ):
                st.error(f"{uploaded_file.name}: {text}")
                any_errors = True

            else:
                new_docs.append(
                    {
                        "name": uploaded_file.name,
                        "text": text,
                    }
                )

        if new_docs:
            st.session_state.docs = new_docs

        if new_docs and not any_errors:
            st.success(
                f"Uploaded {len(new_docs)} document(s): "
                + ", ".join(d["name"] for d in new_docs)
            )

    if st.session_state.docs:

        docs = st.session_state.docs

        with st.expander(
            f"Preview {len(docs)} document(s)"
        ):

            for d in docs:
                st.markdown(f"**{d['name']}**")
                st.text(d["text"][:3000])
                st.markdown("---")

        doc_names = [d["name"] for d in docs]

        selected_names = (
            st.multiselect(
                "Which document(s) should the AI use?",
                doc_names,
                default=doc_names,
            )
            if len(docs) > 1
            else doc_names
        )

        action_options = [
            "Summarize document",
            "Explain document",
            "Extract key points",
            "Ask a question",
        ]

        if len(selected_names) > 1:
            action_options.append("Compare documents")

        document_action = st.selectbox(
            "What do you want to do?",
            action_options,
        )

        document_question = ""

        if document_action == "Ask a question":

            document_question = st.text_input(
                "Ask a question",
                placeholder=(
                    "Example: What is the main topic?"
                ),
            )

        if st.button(
            "Analyze Document(s) ✨",
            use_container_width=True,
        ):

            if not selected_names:

                st.warning(
                    "Please select at least one document."
                )
                st.stop()

            selected_docs = [
                d for d in docs if d["name"] in selected_names
            ]

            combined_text = "\n\n".join(
                f"--- Document: {d['name']} ---\n{d['text']}"
                for d in selected_docs
            )

            content_for_ai, doc_was_trimmed = truncate_text(
                combined_text
            )

            if doc_was_trimmed:
                st.warning(
                    f"Your document(s) were long, so only the first "
                    f"{MAX_INPUT_CHARS:,} characters (combined) were sent "
                    "to the AI to stay within the free-tier request limit."
                )

            if document_action == "Summarize document":

                prompt = f"""
Summarize the following document(s) clearly and concisely.
If there is more than one document, summarize each one separately
under its own heading.

{content_for_ai}
"""

            elif document_action == "Explain document":

                prompt = f"""
Explain the following document(s) in simple, easy-to-understand terms.
If there is more than one document, explain each one separately
under its own heading.

{content_for_ai}
"""

            elif document_action == "Extract key points":

                prompt = f"""
Extract the key points from the following document(s) as a clear
bullet-point list. If there is more than one document, group the
points under each document's name.

{content_for_ai}
"""

            elif document_action == "Compare documents":

                prompt = f"""
Compare and contrast the following documents. Point out the main
similarities, differences, and anything that appears in one document
but not the others.

{content_for_ai}
"""

            else:

                if not document_question.strip():

                    st.warning(
                        "Please enter a question."
                    )

                    st.stop()

                prompt = f"""
Answer the following question using only the information in the
document(s) below. If the answer is not in the document(s), say so
clearly. If multiple documents are provided, mention which document
the answer came from.

{content_for_ai}

Question:
{document_question}
"""

            show_ai_result_stream(
                ask_ai_stream(
                    prompt,
                    system=(
                        "You are an intelligent document assistant. "
                        "Use the provided document(s) as the main source. "
                        "Do not invent information that is not supported "
                        "by the document(s)."
                    ),
                ),
                "Document Assistant 📄",
            )

        if st.button(
            "Clear All Documents 🗑️",
            use_container_width=True,
        ):

            st.session_state.docs = []

            st.rerun()

    else:

        st.markdown(
            """
            <div class="info-box">
                📌 Upload one or more TXT, CSV, PDF, DOCX,
                or XLSX files to get started.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# WORKSPACE
# ============================================================

elif st.session_state.page == "Workspace":

    st.markdown(
        '<div class="page-header">✨ Workspace</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Organize your ideas, notes and projects.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        workspace_text = st.text_area(
            "Your workspace",
            placeholder=(
                "Write your notes, ideas, project details, "
                "study topics, or anything you want to work on..."
            ),
            height=400,
        )

    with col2:

        workspace_action = st.selectbox(
            "What should AI do?",
            [
                "Organize my notes",
                "Improve my writing",
                "Make a study plan",
                "Make a project plan",
                "Give me ideas",
            ],
        )

        if workspace_action == "Organize my notes":

            task = (
                "Organize the notes clearly using headings, "
                "subheadings and bullet points."
            )

        elif workspace_action == "Improve my writing":

            task = (
                "Improve the writing while keeping the "
                "original meaning. Make it clear and natural."
            )

        elif workspace_action == "Make a study plan":

            task = (
                "Create a practical and easy-to-follow "
                "study plan from the given content."
            )

        elif workspace_action == "Make a project plan":

            task = (
                "Create a clear step-by-step project plan "
                "from the given content."
            )

        else:

            task = (
                "Give useful, practical and interesting ideas "
                "based on the given content."
            )

        if st.button(
            "Work with AI ✨",
            use_container_width=True,
        ):

            if workspace_text.strip():

                workspace_prompt = f"""
{task}

Content:
{workspace_text}
"""

                show_ai_result_stream(
                    ask_ai_stream(
                        workspace_prompt,
                        system=(
                            "You are a helpful assistant for organizing "
                            "study and project content."
                        ),
                    ),
                    "Workspace Assistant ✨",
                )

            else:

                st.warning(
                    "Please write something in your workspace first."
                )