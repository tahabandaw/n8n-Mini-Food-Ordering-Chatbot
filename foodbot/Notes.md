# Notes.md — Technical Decisions & Assumptions

## AI Model Choice: Ollama + llama3.1:8b

### Why llama3.1:8b?
- **Open-source & free**: No API keys, no usage limits, no vendor lock-in
- **Runs locally**: Inference stays on your machine; no data leaves your network
- **Good at structured output**: The 8B parameter model handles JSON extraction well with proper system prompts
- **OpenAI-compatible API**: Ollama exposes `/v1/chat/completions`, making integration trivial — any HTTP Request node in n8n can call it
- **Resource-friendly**: Runs on 8GB RAM (CPU) or ~6GB VRAM (GPU); no high-end hardware needed
- **Fast enough for chat**: Typical response time is 1-3 seconds for short JSON extraction tasks

### Alternatives Considered
| Model | Why Not |
|-------|---------|
| llama3.1:70b | Too heavy for a demo; requires 40GB+ VRAM |
| mistral:7b | Slightly worse at structured JSON output in testing |
| phi-3 | Good but less widely tested for tool-use/extraction |
| GPT-4 / Claude | Proprietary — explicitly disallowed by requirements |

## AI Extraction Strategy

### Three-Layer Parsing Pipeline

```
User Input
    │
    ├─[1] Structured Code Parser (regex)
    │     Matches: "BG1 x2", "DR1, SD2", "bg3"
    │     Fast, deterministic, no AI needed
    │
    ├─[2] Ollama LLM Extraction (if step 1 fails)
    │     System prompt with full menu context
    │     Strict JSON output: {"items": [{"code": "BG1", "qty": 2}]}
    │     Temperature: 0.1 for consistency
    │
    └─[3] Keyword Fallback (if step 2 fails)
          Simple string matching against known food words
          "chicken burger" → BG2, "fries" → SD1, etc.
          Last resort — always produces some result
```

### Why This Layered Approach?
1. **Speed**: Most power users will type codes (`BG1 x2`) — no need to invoke AI
2. **Reliability**: If Ollama is down or returns garbage, keyword matching still works
3. **Quality**: Natural language inputs like "I want two chicken burgers and a coke" get properly parsed by the LLM
4. **Graceful degradation**: System never fully breaks; worst case = keyword matching

### AI Prompt Design
The system prompt is carefully crafted:
- Full menu is embedded directly in the prompt (no retrieval needed)
- Output format is strictly specified as JSON
- Temperature is set to 0.1 to minimize creative responses
- The model is told to pick the CLOSEST match for ambiguous inputs
- Markdown code block cleanup handles common LLM output quirks

## Persistence: `$getWorkflowStaticData('global')`

### Why Not a Database?
- n8n's workflow static data persists across executions within the same workflow
- Zero additional infrastructure needed
- Sufficient for a demo/interview prototype
- Data survives workflow re-executions but clears on n8n restart

### What's Stored
- **carts**: `{ [chatId]: [{code, name, price, qty}] }`
- **checkouts**: `{ [chatId]: {step, deliveryMethod, name, phone, address} }`
- **orders**: `{ [orderId]: {full order details + timestamp} }`
- **orderCounter**: Sequential counter for order IDs (starts at 1000)

### Production Alternative
For production, replace with:
- Redis (fast, ephemeral carts)
- PostgreSQL via n8n's Postgres node (persistent orders)
- n8n's built-in "Data Store" node (simpler but limited)

## Checkout Flow Design

```
[Checkout] → Pick Delivery/Pickup
    → Enter Name
    → Enter Phone (validated for Egyptian format)
    → Enter Address (delivery only)
    → Order Summary + Confirm
    → Order ID generated → Cart cleared
```

### Validation Rules
- **Name**: 2-50 characters
- **Phone**: Egyptian mobile format (`01[0125]XXXXXXXX` or `+201[0125]XXXXXXXX`)
- **Address**: Minimum 5 characters (delivery only)
- **Cart**: Must not be empty at checkout start

### State Machine
The checkout uses a step-based state machine stored per `chatId`. This prevents race conditions and lets users pick up where they left off if they accidentally tap the wrong button.

## Assumptions

1. **Single-user per chat_id**: No shared Telegram groups (bot works in DMs)
2. **EGP currency**: All prices in Egyptian Pounds as specified
3. **Menu is static**: Hardcoded in Code nodes; for production, externalize to a database
4. **Order fulfillment is out of scope**: No kitchen display, status updates, or payment integration
5. **Ollama is pre-warmed**: The model should be pulled (`ollama pull llama3.1:8b`) before first use
6. **Demo tokens**: Telegram bot token is pasted at demo time; not committed to repo
7. **n8n Data Store vs Static Data**: Used `$getWorkflowStaticData('global')` which is simpler and built into Code nodes. The "Data Store" node could also work but requires additional node configuration.

## Edge Cases Handled

| Scenario | Handling |
|----------|----------|
| Empty cart checkout | Blocked with message + menu link |
| Invalid item code | Error message + menu suggestion |
| AI returns invalid JSON | Falls through to keyword matching |
| Ollama is down | HTTP request has `continueOnFail`, keyword fallback activates |
| Invalid phone number | Re-prompts with format example |
| Double-add same item | Quantities merge (not duplicated) |
| Unknown text input | Friendly error + Help/Menu buttons |
| Callback from unknown button | Fallback handler catches it |

## File Structure
```
n8n-food-bot/
├── docker-compose.yml     # n8n + Ollama services
├── .env.example           # Environment variable template
├── menu.json              # Menu data (reference; also embedded in nodes)
├── workflow.json           # n8n workflow export (import this)
├── README.md              # Setup & usage guide
├── Notes.md               # This file
└── sample-transcript.md   # Demo conversation logs
```
