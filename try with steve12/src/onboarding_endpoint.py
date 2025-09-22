from orchestrator.audira_agent_orchestrator import (
    summarize_business_profile,
    assign_discovery_tags,
    generate_question_for_each_tag
)

from stages.stage1_10_fixed_questions import collect_core_answers
from stages.stage4_upload_brand_files import save_brand_file_paths
from fastapi import APIRouter, Depends,WebSocket

onboarding_websocket = APIRouter(

)

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:5000/ws/onboarding?project_id=5351");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""


@onboarding_websocket.get("/")
async def get():
    return HTMLResponse(html)

# @onboarding_websocket.websocket("/ws/onboarding")
# async def onboarding_handler(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             data = await websocket.receive_text()
#             await websocket.send_text(f"Echo: {data}")
#     except Exception as e:
#         print(f"WebSocket Error: {e}")
#     finally:
#         await websocket.close()


@onboarding_websocket.websocket("/ws/onboarding")
async def onboarding(websocket: WebSocket, project_id: str):
    await websocket.accept()
    try:
        # Start the onboarding process
        await run_onboarding_process(websocket,project_id)
        await websocket.send_text("✅ Onboarding process completed successfully!")
    except Exception as e:
        await websocket.send_text(f"❌ Error during onboarding: {str(e)}")
    finally:
        await websocket.close()


async def run_onboarding_process(websocket: WebSocket, project_id: str):
    print("🔷 Stage 1: Collecting fixed core answers...\n")
    core_data = collect_core_answers()

    print("\n🔷 Stage 1.5: Summarizing business profile...\n")
    profile_summary = summarize_business_profile(core_data)
    for k, v in profile_summary.items():
        print(f"- {k.replace('_', ' ').title()}: {v}")

    print("\n🔷 Stage 1.6: Assigning discovery tags...\n")
    tag_map = assign_discovery_tags(profile_summary)
    print("📌 Assigned Tags:")
    for tag, reason in tag_map.items():
        print(f"- {tag}: {reason}")


    print("\n🔷 Stage 2: Generating 1 tag-based question per discovery tag...\n")
    tag_questions = generate_question_for_each_tag(tag_map, core_data)

    for tag, data in tag_questions.items():
        print(f"\n🧩 Tag: {tag}")
        print(f"❓ Question: {data['question']}")
        answer = input("Your Answer: ")
        tag_questions[tag]["answer"] = answer

    full_data = {
        "core_questions": core_data,
        "business_profile": profile_summary,
        "tag_mapping": tag_map,
        "tag_questions": tag_questions,

    }
    import requests
    from fastapi import Request
    from fastapi.datastructures import State
    import json   # Save to file
    file_name = f"onboarding_data_{project_id}.txt"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=4, ensure_ascii=False)

 
    

    print(f"\n✅ All data collected successfully. Saved to '{file_name}'. Total questions: {len(tag_questions)}")

