# 🍔 FoodBot — n8n Telegram Food Ordering Chatbot

A minimal yet reliable food-ordering chatbot on Telegram, orchestrated with **n8n**, using a local open-source LLM (**Ollama + llama3.1:8b**) for natural language item extraction.

## Features

- **Menu browsing** with inline keyboard buttons
- **Structured ordering**: `BG1 x2, DR1` style codes
- **Free-text ordering**: "I want a chicken burger and fries" → AI parses it
- **3-layer parsing**: Regex → Ollama AI → Keyword fallback
- **Cart persistence** per Telegram user (via n8n workflow static data)
- **Full checkout**: Pickup/Delivery, name, phone validation (Egyptian format), address
- **Order confirmation** with generated Order ID

## Architecture

```
Telegram User
    │
    ▼
[Telegram Bot API]
    │
    ▼
[n8n Workflow]
    ├── Router (Code node)
    ├── Switch (routes to handlers)
    ├── Menu / Cart / Checkout handlers (Code nodes)
    ├── Free-text parser
    │     ├── [1] Regex structured codes
    │     ├── [2] Ollama llama3.1:8b (HTTP Request)
    │     └── [3] Keyword fallback
    ├── Cart Manager (workflow static data)
    └── Send Telegram Message
```

## Prerequisites

- **Docker & Docker Compose** (v2+)
- **8GB+ RAM** (for Ollama model inference)
- A **Telegram Bot Token** from [@BotFather](https://t.me/BotFather)

## Quick Start

### 1. Clone & Configure

```bash
git clone <your-repo-url>
cd n8n-food-bot
cp .env.example .env
```

Edit `.env` and paste your Telegram bot token:
```
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIjKlMnOpQrStUvWxYz
```

### 2. Start Services

```bash
docker compose up -d
```

This starts:
- **n8n** on `http://localhost:5678`
- **Ollama** on `http://localhost:11434`

### 3. Pull the LLM Model

```bash
docker exec -it ollama-food-bot ollama pull llama3.1:8b
```

> ⏱ First pull downloads ~4.7GB. Subsequent starts are instant.

If you're running Ollama on your host machine instead of Docker:
```bash
ollama pull llama3.1:8b
```
Then update `.env`:
```
OLLAMA_BASE_URL=http://host.docker.internal:11434   # Mac/Windows
OLLAMA_BASE_URL=http://172.17.0.1:11434              # Linux
```

### 4. Import the Workflow into n8n

1. Open `http://localhost:5678` in your browser
2. Log in (default: admin / changeme)
3. Go to **Workflows** → **Import from File**
4. Select `workflow.json`
5. **Configure the Telegram credential**:
   - Click on the "Telegram Trigger" node
   - Under Credentials, click "Create New"
   - Name: `Telegram Bot`
   - Access Token: paste your bot token
   - Save
6. Do the same for the "Send Telegram Message" node (use the same credential)
7. Click **Save** then **Activate** the workflow (toggle in top-right)

### 5. Set the Webhook

n8n should automatically register the Telegram webhook when you activate the workflow. If it doesn't work:

```bash
# Manual webhook setup (replace YOUR_TOKEN and YOUR_N8N_URL)
curl "https://api.telegram.org/botYOUR_TOKEN/setWebhook?url=YOUR_N8N_URL/webhook/food-bot-webhook"
```

> For local development, use **ngrok** to expose your n8n:
> ```bash
> ngrok http 5678
> ```
> Then use the ngrok URL as your webhook URL.

### 6. Test It!

Open Telegram → find your bot → send `/start`

## Demo Token Setup (For Reviewers)

During review, the following values need to be pasted:

| Where | What | How |
|-------|------|-----|
| n8n Telegram credential | Bot Token | Workflows → Open workflow → Click Telegram Trigger → Credentials → Edit → Paste token |
| `.env` file (if restarting) | `TELEGRAM_BOT_TOKEN` | Edit `.env` before `docker compose up` |

## Menu

| Code | Item | Price (EGP) |
|------|------|-------------|
| BG1 | Classic Beef Burger | 85 |
| BG2 | Chicken Burger | 75 |
| BG3 | Double Smash Burger | 120 |
| BG4 | Veggie Burger | 70 |
| SD1 | French Fries | 30 |
| SD2 | Onion Rings | 35 |
| SD3 | Coleslaw | 20 |
| SD4 | Cheese Fries | 45 |
| DR1 | Coca-Cola | 25 |
| DR2 | Fresh Lemonade | 35 |
| DR3 | Mango Smoothie | 45 |
| DR4 | Water | 10 |

## File Structure

```
n8n-food-bot/
├── docker-compose.yml      # n8n + Ollama containers
├── .env.example             # Environment variable template
├── menu.json                # Menu data (reference copy)
├── workflow.json             # n8n workflow (import this)
├── README.md                # This file
├── Notes.md                 # AI choice, fallback strategy, assumptions
└── sample-transcript.md     # Demo conversation showing all flows
```

## Troubleshooting

**Bot doesn't respond?**
- Check the workflow is activated (green toggle in n8n)
- Verify webhook: `curl https://api.telegram.org/bot<TOKEN>/getWebhookInfo`
- Check n8n execution logs for errors

**AI extraction not working?**
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Ensure model is pulled: `docker exec ollama ollama list`
- The keyword fallback will still work even if Ollama is down

**"Unauthorized" in n8n?**
- Re-enter the Telegram bot token in the credential settings

## Tech Stack

- **n8n** — Workflow automation (self-hosted)
- **Ollama** — Local LLM inference server
- **llama3.1:8b** — Open-source language model
- **Telegram Bot API** — Chat interface
- **Docker Compose** — Container orchestration
