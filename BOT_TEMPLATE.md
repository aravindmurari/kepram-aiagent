# Kepram AI Bot — Reusable Template

Use this as the starting point for any new client bot. Every section has [CUSTOMIZE] markers where you change things per client.

---

## Stack (do not change per client)

| Layer | Technology | Why |
|-------|-----------|-----|
| LLM | Claude API (`claude-sonnet-4-6`) | Best reasoning, stays in context |
| RAG | LlamaIndex + Pinecone Serverless | Retrieves from client's knowledge base |
| Embeddings | FastEmbed `BAAI/bge-small-en-v1.5` | Free, local, no OpenAI key needed |
| Backend | Python 3.11 + FastAPI | Python-first RAG ecosystem |
| Frontend | Vanilla HTML/CSS/JS | No build step, embeds anywhere |
| Font | Inter (Google Fonts) | Clean, professional |

**Python version**: Always use 3.11. Python 3.14 breaks fastembed and most ML packages.

---

## Folder Structure

```
[client-name]-bot/
├── CLAUDE.md               ← project briefing (copy and fill in)
├── BOT_TEMPLATE.md         ← this file
├── .env                    ← real keys, never commit
├── .env.example            ← template, safe to commit
├── .gitignore
├── context/
│   └── strategy.md         ← business context for this client
├── knowledge/              ← all RAG source files go here
│   ├── overview.md         ← what the business does
│   ├── services.md         ← offerings and pricing
│   ├── faq.md              ← common questions and answers
│   ├── team.md             ← who they are, credentials, background
│   └── [vertical].md       ← industry-specific content
├── backend/
│   ├── requirements.txt
│   ├── main.py             ← FastAPI app
│   ├── rag.py              ← RAG + Claude logic
│   └── ingest.py           ← one-time ingestion script
└── frontend/
    ├── index.html          ← chat widget
    └── style.css
```

---

## .env.example

```
ANTHROPIC_API_KEY=sk-ant-api03-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=[client-name]-knowledge
```

---

## .gitignore

```
.env
venv/
__pycache__/
*.pyc
.DS_Store
```

---

## backend/requirements.txt

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
python-dotenv>=1.0.0
anthropic>=0.40.0
llama-index-core>=0.11.0
llama-index-llms-anthropic>=0.3.0
llama-index-embeddings-fastembed>=0.1.2
llama-index-vector-stores-pinecone>=0.2.0
pinecone>=5.0.0
fastembed>=0.1.0
pymupdf>=1.24.0
python-docx>=1.1.0
```

---

## backend/rag.py (full template)

```python
import os
import anthropic as anthropic_sdk
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# [CUSTOMIZE] — rewrite this entire prompt for each client
SYSTEM_PROMPT = """You are [BOT NAME], the AI assistant for [BUSINESS NAME].

Your role: [1-2 sentences on what the bot does for this business]

Your jobs:
1. [Primary job — e.g. qualify leads]
2. [Secondary job — e.g. answer FAQs]
3. [Tertiary job — e.g. move toward a booking/contact]

Rules:
- Keep responses concise: 2–4 sentences per turn unless detail is genuinely needed
- Ask one clarifying question at a time when qualifying
- Never make up facts — if unsure, offer to connect them with a real person
- Short replies (1–3 words) are complete answers — treat them as such, never ask if the message was cut off
- You have full conversation history — never ask for information already provided
- If the user signals they have booked/completed an action, acknowledge warmly and do not repeat the CTA
- [Any business-specific rules — compliance, tone, restricted topics]

Tone: [e.g. professional but warm / friendly and casual / authoritative]"""


def _configure_settings():
    Settings.llm = Anthropic(
        model="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        system_prompt=SYSTEM_PROMPT,
        max_tokens=1024,
    )
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")


def _build_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    return VectorStoreIndex.from_vector_store(vector_store)


_configure_settings()
_index = _build_index()
_retriever = VectorIndexRetriever(index=_index, similarity_top_k=3)
_claude = anthropic_sdk.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _retrieve_context(message: str) -> tuple[str, list[str]]:
    nodes = _retriever.retrieve(message)
    context = "\n\n---\n\n".join(node.text for node in nodes)
    sources = list({
        node.metadata.get("file_name", "")
        for node in nodes
        if node.metadata.get("file_name")
    })
    return context, sources


def _build_messages(message: str, context: str, history: list) -> list:
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({
        "role": "user",
        "content": f"Knowledge base context:\n{context}\n\nUser message: {message}",
    })
    return messages


def query(message: str, history: list = None) -> dict:
    context, sources = _retrieve_context(message)
    messages = _build_messages(message, context, history or [])
    response = _claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return {"response": response.content[0].text, "sources": sources}


def query_stream(message: str, history: list = None):
    context, _ = _retrieve_context(message)
    messages = _build_messages(message, context, history or [])
    with _claude.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for token in stream.text_stream:
            yield token
```

---

## backend/main.py (copy as-is, no changes needed)

```python
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import rag

app = FastAPI(title="Kepram AI Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[HistoryMessage] = []


class ChatResponse(BaseModel):
    response: str
    sources: list[str] = []


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    history = [h.model_dump() for h in request.history]
    result = rag.query(request.message, history=history)
    return ChatResponse(**result)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    history = [h.model_dump() for h in request.history]

    def generate():
        for token in rag.query_stream(request.message, history=history):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}
```

---

## backend/ingest.py (copy as-is, no changes needed)

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

Settings.llm = Anthropic(model="claude-sonnet-4-6", api_key=os.environ["ANTHROPIC_API_KEY"])
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

existing_names = [i.name for i in pc.list_indexes()]
if INDEX_NAME not in existing_names:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Created Pinecone index: {INDEX_NAME}")
else:
    print(f"Using existing index: {INDEX_NAME}")

pinecone_index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print(f"Loading documents from {KNOWLEDGE_DIR} ...")
docs = SimpleDirectoryReader(str(KNOWLEDGE_DIR)).load_data()
print(f"Loaded {len(docs)} document chunks")

print("Ingesting into Pinecone ...")
VectorStoreIndex.from_documents(docs, storage_context=storage_context)
print("Done. Knowledge base is ready.")
```

---

## frontend customization checklist

In `index.html`, change:
- `<title>` — client/bot name
- `id="bot-name"` text — bot display name
- `id="bot-tagline"` text — one-line description
- `id="header-avatar"` letter — first letter of bot name
- Opening message in `#chat-messages` — the first thing the bot says
- `API_URL` — update port or production URL when deployed
- Calendly regex in `renderMarkdown()` — swap for client's booking link if different

In `style.css`, change:
- Gradient colors (search for `#6366f1` and `#8b5cf6`) — swap to client's brand colors
- Background gradient — adjust the radial-gradient in `body`

---

## Commands (run in backend/ every time)

```bash
# First-time setup
/opt/homebrew/bin/python3.11 -m venv venv
venv/bin/python3 -m pip install -r requirements.txt

# After adding/editing any knowledge file
venv/bin/python3 ingest.py

# Start the server (pick a free port)
venv/bin/python3 -m uvicorn main:app --port 8001

# Open the frontend
open ../frontend/index.html
```

---

## Port allocation (avoid conflicts)

| App | Port |
|-----|------|
| Isha MoM | 8000 |
| Kepram bot | 8001 |
| Trinity CRE bot (next) | 8002 |
| Next client | 8003 |

---

## Gotchas learned building Kepram bot

| Problem | Fix |
|---------|-----|
| Python 3.14 breaks fastembed | Always use `/opt/homebrew/bin/python3.11 -m venv venv` |
| Port 8000 taken by Isha MoM | Start each new bot on the next free port |
| Bot re-asks questions already answered | Pass `history` array with every request |
| Bot says "your message seems cut off" on short answers | Add explicit rule to system prompt |
| Bot offers booking link after user confirms they booked | Add "if user says done/booked, stop showing the CTA" rule |
| Streaming sends full response as one chunk | Bypass LlamaIndex streaming — use `_claude.messages.stream()` directly |
| Calendly URL appears as raw text | Use `renderMarkdown()` with regex to convert to styled button |

---

## Knowledge file writing guide

Each `.md` file in `knowledge/` becomes searchable context. Write them like internal documentation, not marketing copy. The bot retrieves chunks and answers from them, so clarity beats brevity.

**Good knowledge file structure:**
```markdown
# [Topic]

## [Sub-section]
[Specific, factual content. Short paragraphs. No fluff.]

## FAQs
**Q: [Exact phrasing a user might ask]**
A: [Direct answer]
```

**Supported file types:** `.md`, `.txt`, `.pdf`, `.docx`, `.html`, `.csv`, `.pptx`

Drop any supported file into `knowledge/` and re-run `ingest.py`.

---

## Calendly link pattern

The frontend auto-converts any `calendly.com` URL in a bot response into a styled button.
To change the booking link: update it in the knowledge files (services.md, faq.md) and update the regex in `renderMarkdown()` if the domain is different.
