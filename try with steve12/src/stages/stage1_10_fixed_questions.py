def get_core_questions():
    return [
        "What is the core problem you want the AI agent to solve for your business?",
        "Who is your primary audience or user interacting with this AI?",
        "Which channel(s) will the AI agent operate on? (e.g., website, WhatsApp, voice IVR, etc.)",
        "What business outcomes do you want the AI to drive? (e.g., reduce support time, increase sales, qualify leads)",
        "What tone or personality should your AI agent reflect? (e.g., formal, casual, friendly, luxury)",
        "Are there any critical workflows or systems the agent must integrate with?",
        "How should the agent handle unknowns or sensitive topics?",
        "Do you want the agent to follow a fixed script, be dynamic, or a mix?",
        "What kind of data privacy or compliance rules should we consider? (e.g., GDPR, HIPAA, local laws)",
        "Do you already have materials (FAQs, scripts, tone guides) we can use to train the AI?"
    ]

def collect_core_answers():
    questions = get_core_questions()
    answers = {}

    for i, question in enumerate(questions, 1):
        # Simulated input for now
        answer = input(f"{i}. {question}\n> ")
        answers[f"Q{i}"] = {
            "question": question,
            "answer": answer
        }

    return answers