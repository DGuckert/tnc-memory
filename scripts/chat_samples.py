import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="/home/daniel/tnc/training_runs/chat-135m-lora")
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.adapter)
base_name = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float16).to("cuda")
model = PeftModel.from_pretrained(base, args.adapter).eval()
prompts = [
    ("coding", "You are a coding assistant in a Discord chat. Continue the conversation naturally and help with the code or technical question.", "Recent messages:\nUser: how do I install torch with cuda on windows?\nUser: I keep getting a version mismatch."),
    ("coding", "You are a coding assistant in a Discord chat. Continue the conversation naturally and help with the code or technical question.", "Recent messages:\nUser: my python script crashes when I load the model on the gpu.\nUser: it says CUDA out of memory."),
    ("chat", "You are participating in a Discord chat. Continue the conversation naturally.", "Recent messages:\nUser: hey, how's it going?\nUser: I finally got the new setup working."),
]
for kind, system, user in prompts:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=60, do_sample=False,
                             pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    print("---", kind, flush=True)
    print(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip(), flush=True)
