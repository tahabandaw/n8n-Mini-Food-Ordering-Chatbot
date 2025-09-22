# import torch
# from parler_tts  import ParlerTTSForConditionalGeneration
# from transformers import AutoTokenizer
# import soundfile as sf

# async def inference_parlertts(text:str,business_id:str):
        
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"

#     model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1",
#                                                                   attn_implementation="eager"
#                                                                 ).to(device)
    
#     tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

#     prompt = f"{text}"
#     description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

#     input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
#     prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

#     generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
#     audio_arr = generation.cpu().numpy().squeeze()
#     sf.write(f"parler_tts_out{business_id}.wav", audio_arr, model.config.sampling_rate)


import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

async def inference_parlertts(text: str, business_id: str):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1",
        attn_implementation="eager"
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    
    # Fix the tokenizer padding token issue
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = f"{text}"
    description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

    # Tokenize with explicit attention_mask and padding
    input_ids = tokenizer(
        description, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        return_attention_mask=True
    ).to(device)
    
    prompt_input_ids = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        return_attention_mask=True
    ).to(device)

    # Generate audio with attention masks
    generation = model.generate(
        input_ids=input_ids.input_ids,
        prompt_input_ids=prompt_input_ids.input_ids,
        attention_mask=input_ids.attention_mask,
        prompt_attention_mask=prompt_input_ids.attention_mask
    )
    
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(f"parler_tts_out{business_id}.wav", audio_arr, model.config.sampling_rate)

# import torch
# from parler_tts import ParlerTTSForConditionalGeneration
# from transformers import AutoTokenizer
# import soundfile as sf

# async def inference_parlertts(text: str, business_id: str):
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"

#     model = ParlerTTSForConditionalGeneration.from_pretrained(
#         "parler-tts/parler-tts-mini-v1",
#         attn_implementation="eager"
#     ).to(device)
    
#     tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

#     prompt = f"{text}"
#     description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

#     # Add padding and attention mask
#     inputs = tokenizer(
#         description,
#         padding=True,
#         return_tensors="pt",
#         return_attention_mask=True
#     )
#     prompt_inputs = tokenizer(
#         prompt,
#         padding=True,
#         return_tensors="pt",
#         return_attention_mask=True
#     )

#     # Move everything to device
#     input_ids = inputs["input_ids"].to(device)
#     attention_mask = inputs["attention_mask"].to(device)
#     prompt_input_ids = prompt_inputs["input_ids"].to(device)
#     prompt_attention_mask = prompt_inputs["attention_mask"].to(device)

#     # Generate with attention masks
#     generation = model.generate(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         prompt_input_ids=prompt_input_ids,
#         prompt_attention_mask=prompt_attention_mask
#     )
    
#     audio_arr = generation.cpu().numpy().squeeze()
#     sf.write(f"parler_tts_out{business_id}.wav", audio_arr, model.config.sampling_rate)
#     return audio_arr