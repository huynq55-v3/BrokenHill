import os
from fastchat.conversation import conv_templates, get_conv_template, Conversation

def get_conversation_template(model_path: str) -> Conversation:
    model_path_lower = model_path.lower()
    if "llama-2" in model_path_lower or "llama2" in model_path_lower:
        return get_conv_template("llama-2")
    elif "llama-3" in model_path_lower or "llama3" in model_path_lower:
        return get_conv_template("llama-3")
    elif "mistral" in model_path_lower:
        return get_conv_template("mistral")
    elif "qwen" in model_path_lower:
        return get_conv_template("qwen-7b-chat")
    elif "chatglm" in model_path_lower or "glm" in model_path_lower:
        return get_conv_template("chatglm3")
    elif "solar" in model_path_lower:
        return get_conv_template("solar")
    elif "mpt" in model_path_lower:
        return get_conv_template("mpt-7b-chat")
    elif "vicuna" in model_path_lower:
        return get_conv_template("vicuna_v1.1")
    return get_conv_template("zero_shot")
