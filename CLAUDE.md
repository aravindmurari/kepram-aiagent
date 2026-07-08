# Kepram AI Bot — Claude Code Briefing

## What We Are Building

A lead qualification and AI advisory chatbot for Kepram LLC — Aravind Murari's AI consulting firm.

**Four jobs the bot does:**
1. Qualify leads (asks about business type, challenge, timeline, budget)
2. Showcase Aravind's expertise by vertical (insurance, legal, CRE, healthcare, education)
3. Answer questions about Kepram's AI services and how the tech works
4. Invite qualified leads to book a 30-minute discovery call

**The meta-demo:** the bot IS the product. Every visitor who talks to it and is impressed is a potential client.

## Full Background Context

Read: `/context/strategy.md`
Covers: full business rationale, 6-layer stack explanation, 1mind competitive analysis, Trinity CRE first client opportunity, go-to-market timeline.

## About Aravind

- 15+ years Senior BSA / Product Owner
- Domains: retail, insurance, payments, healthcare, education
- CAIO (Chief AI Officer) certified
- HCI instructor at Mercer University, Atlanta GA
- Building a vertical AI avatar bot agency for SMBs
- Python venv is always at `venv/bin/python3` — never use system python

## Tech Stack

| Layer | Tool |
|-------|------|
| LLM | Claude API (`claude-sonnet-4-6`) via LlamaIndex |
| RAG | LlamaIndex (llama-index-core) |
| Embeddings | FastEmbed `BAAI/bge-small-en-v1.5` (runs locally, no API key) |
| Vector DB | Pinecone Serverless |
| Backend | Python + FastAPI |
| Frontend | Vanilla HTML/CSS/JS (no build step) |
| Hosting | Railway (backend) + Vercel or GitHub Pages (frontend) |

## Project Structure

```
kepram-bot/
├── CLAUDE.md                    ← you are here
├── .env                         ← never commit this
├── .env.example                 ← template, safe to commit
├── context/
│   └── strategy.md              ← full business + tech context
├── knowledge/                   ← RAG source files (edit these, then re-run ingest.py)
│   ├── company.md               ← overview, two products, Why Kepram, process, guarantee, NO pricing
│   ├── ai-front-desk.md         ← the turnkey product (voice, missed-call text-back, chat, booking)
│   ├── ai-custom-agents.md      ← the built-to-order product (goals/knowledge/guardrails engine)
│   ├── verticals.md             ← the 8 site industries + "any other industry"
│   ├── faq.md                   ← common questions; quote-based, no published prices
│   ├── resume.md                ← founder background (surface only when asked; default to "we"/"Kepram")
│   └── caio.md                  ← AI philosophy + CAIO framework
│   NOTE (July 2026): retrained around the TWO-product site (AI Front Desk + AI Custom Agents).
│   No pricing anywhere on the site or in the bot — always route pricing to a quote. The old
│   three-service framing (AI Assistants / Custom Software / AI Training) and services.md are gone.
├── backend/
│   ├── requirements.txt
│   ├── main.py                  ← FastAPI app (/chat, /health endpoints)
│   ├── rag.py                   ← LlamaIndex + Pinecone + Claude query logic
│   └── ingest.py                ← one-time script: loads knowledge/ into Pinecone
└── frontend/
    ├── index.html               ← standalone chat widget
    └── style.css
```

## Build Sequence

- [x] Phase 1 — Text chatbot MVP (current)
- [ ] Phase 2 — Add ElevenLabs voice (TTS) + Deepgram (STT)
- [ ] Phase 3 — Add D-ID avatar face with real-time lip sync
- [ ] Phase 4 — Lead capture email trigger + Calendly booking

## Running Locally

```bash
# First time setup
cd backend
python3 -m venv venv
venv/bin/python3 -m pip install -r requirements.txt

# Ingest knowledge base into Pinecone (run once, or after editing knowledge files)
venv/bin/python3 ingest.py

# Start the API server
venv/bin/python3 -m uvicorn main:app --reload --port 8000

# Open frontend in browser
open ../frontend/index.html
```

## Key Files to Edit First

1. `.env` holds the API keys (already set)
2. To change what the bot knows: edit `knowledge/*.md`, then re-run `ingest.py` (it clears + re-ingests Pinecone). This updates the LIVE bot immediately — no redeploy needed, since the backend reads Pinecone at runtime.
3. To change how the bot behaves/frames itself (identity, rules, pricing stance, CTA): edit `SYSTEM_PROMPT` in `backend/rag.py`, then redeploy the backend to Railway (`railway up --service kepram-aiagent`).

## Environment Variables Required

```
ANTHROPIC_API_KEY     - from console.anthropic.com (already set in .env)
PINECONE_API_KEY      - from app.pinecone.io (already set in .env)
PINECONE_INDEX_NAME   - kepram-knowledge (already set in .env)
```

No OpenAI key needed — embeddings run locally via FastEmbed.
