from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import re

# === INIT LLM & PROMPT ===
llm = Ollama(model="qwen3:14b",
             base_url='http://56.125.123.8:11434')

system_message = (
    "You are Audira, an expert AI onboarding consultant. "
    "You will guide the configuration of an AI voice agent by asking questions, summarizing input, assigning tags, "
    "generating clarifying questions, and preparing the agent for deployment. "
    "Your responses must always be clear, structured, and precise. "
    "If given prior answers, use them to adapt intelligently — don't repeat. "
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt_template | llm
chat_history = []

def clean_response(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# === STAGE 1.5: BUSINESS PROFILE SUMMARY ===
def summarize_business_profile(core_data: dict) -> dict:
    formatted = "\n".join([f"{qa['question']}\nAnswer: {qa['answer']}" for qa in core_data.values()])
    prompt = (
        f"Summarize the business based on the following answers:\n\n{formatted}\n\n"
        "Output as three fields:\n"
        "- Business Objective\n- Audience & Channel\n- Industry Context"
    )
    response = chain.invoke({"input": prompt, "chat_history": chat_history})
    cleaned = clean_response(response)
    chat_history.append(HumanMessage(content=prompt))
    chat_history.append(AIMessage(content=cleaned))

    summary = {}
    for line in cleaned.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            summary[key.strip().lower().replace(" ", "_")] = val.strip()
    return summary

# === STAGE 1.6: TAG ASSIGNMENT ===
def assign_discovery_tags(profile_summary: dict) -> dict:
    formatted = "\n".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in profile_summary.items()])
    prompt = (
        f"Based on this business profile:\n\n{formatted}\n\n"
        "Assign all relevant Audira configuration tags. Each tag must include a short explanation "
        "of why it is being applied based on the input. Use this exact JSON format:\n\n"
        "{\n"
        "  \"#intent_clarity\": \"...\",\n"
        "  \"#channel_behavior\": \"...\",\n"
        "  \"#tone_variation\": \"...\",\n"
        "  \"#fallback_rules\": \"...\",\n"
        "  \"#integration_scope\": \"...\",\n"
        "  \"#training_data_source\": \"...\",\n"
        "  \"#compliance_constraints\": \"...\",\n"
        "  \"#persona_control\": \"...\",\n"
        "  \"#recommendation_logic\": \"...\",\n"
        "  \"#booking_handling\": \"...\"\n"
        "}"
    )
    response = chain.invoke({"input": prompt, "chat_history": chat_history})
    cleaned = clean_response(response)
    chat_history.append(HumanMessage(content=prompt))
    chat_history.append(AIMessage(content=cleaned))

    try:
        return eval(cleaned) if "{" in cleaned else {}
    except Exception:
        return {"error": "Failed to parse tag output", "raw": cleaned}

# === STAGE 2: PER-TAG QUESTION GENERATION ===
def generate_question_for_each_tag(tag_map: dict, context_data: dict) -> dict:
    tag_questions = {}

    for tag, reason in tag_map.items():
        formatted_context = "\n".join([f"{qa['question']}\nAnswer: {qa['answer']}" for qa in context_data.values()])
        prompt = (
            f"You need to finalize the configuration for the tag '{tag}', which was triggered because:\n"
            f"→ {reason}\n\n"
            f"Here is what we know so far:\n\n{formatted_context}\n\n"
            f"Ask one precise question that would clarify or complete the logic for this tag. "
            f"Do not repeat earlier content. Return only the question."
        )

        response = chain.invoke({"input": prompt, "chat_history": chat_history})
        cleaned = clean_response(response)

        chat_history.append(HumanMessage(content=prompt))
        chat_history.append(AIMessage(content=cleaned))

        tag_questions[tag] = {
            "question": cleaned,
            "reason": reason,
            "answer": None
        }

    return tag_questions