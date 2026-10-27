from enum import IntEnum
from typing import List, Tuple, Any, Optional, Union
import copy

class SeparatorStyle(IntEnum):
    ADD_COLON_SINGLE = 1
    ADD_COLON_TWO = 2
    NO_COLON_SINGLE = 3
    NO_COLON_TWO = 4
    ADD_NEW_LINE_SINGLE = 5
    LLAMA2 = 6
    CHATGLM = 7
    CHATML = 8
    CHATINTERN = 9
    DOLLY = 10
    RWKV = 11
    PHOENIX = 12
    ROBIN = 13
    FALCON_CHAT = 14
    CHATGLM3 = 15
    INTERNLM2 = 16
    MISTRAL = 17

class Conversation:
    def __init__(
        self,
        name: str = "default",
        system_template: str = "{system_message}",
        system_message: str = "",
        roles: Tuple[str, str] = ("USER", "ASSISTANT"),
        messages: Optional[List[List[Any]]] = None,
        offset: int = 0,
        sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE,
        sep: str = "\n",
        sep2: Optional[str] = None,
        stop_str: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        system_message_vision: str = "",
    ):
        self.name = name
        self.system_template = system_template
        self.system_message = system_message
        self.roles = tuple(roles) if roles is not None else ("USER", "ASSISTANT")
        self.messages = [[m[0], m[1]] for m in messages] if messages is not None else []
        self.offset = offset
        self.sep_style = sep_style
        self.sep = sep
        self.sep2 = sep2
        self.stop_str = stop_str
        self.stop_token_ids = stop_token_ids
        self.system_message_vision = system_message_vision

    def get_prompt(self) -> str:
        system_prompt = self.system_template.format(system_message=self.system_message) if self.system_message else ""
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = (system_prompt + self.sep) if system_prompt else ""
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2 or self.sep]
            ret = (system_prompt + seps[0]) if system_prompt else ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt + self.sep if system_prompt else ""
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2 or self.sep]
            ret = system_prompt + seps[0] if system_prompt else ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
            for role, message in self.messages:
                if message:
                    ret += f"{role}\n{message}<|im_end|>\n"
                else:
                    ret += f"{role}\n"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            if self.system_message:
                ret = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n"
            else:
                ret = "<s>[INST] "
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i == 0:
                        ret += f"{message} [/INST] "
                    else:
                        ret += f"<s>[INST] {message} [/INST] " if i % 2 == 0 else f"{message} </s>"
                else:
                    ret += ""
            return ret
        elif self.sep_style == SeparatorStyle.MISTRAL:
            ret = f"<s> [INST] {system_prompt}\n" if system_prompt else "<s> [INST] "
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i == 0:
                        ret += f"{message} [/INST] "
                    else:
                        ret += f"[INST] {message} [/INST] " if i % 2 == 0 else f"{message}</s> "
                else:
                    ret += ""
            return ret
        else:
            ret = (system_prompt + self.sep) if system_prompt else ""
            for role, message in self.messages:
                if message:
                    ret += f"{role}: {message}{self.sep}"
                else:
                    ret += f"{role}:"
            return ret

    def set_system_message(self, system_message: str):
        self.system_message = system_message

    def append_message(self, role: str, message: Optional[str]):
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        if self.messages:
            self.messages[-1][1] = message

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
            system_message_vision=self.system_message_vision,
        )

    def dict(self):
        return {
            "name": self.name,
            "system_template": self.system_template,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "sep_style": self.sep_style.name if hasattr(self.sep_style, "name") else str(self.sep_style),
            "stop_str": self.stop_str,
            "stop_token_ids": self.stop_token_ids,
        }

conv_templates = {
    "zero_shot": Conversation(
        name="zero_shot",
        system_template="{system_message}",
        roles=("### Instruction", "### Response"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
    ),
    "one_shot": Conversation(
        name="one_shot",
        system_template="{system_message}",
        roles=("### Instruction", "### Response"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
    ),
    "llama-2": Conversation(
        name="llama-2",
        system_template="<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
    ),
    "llama-3": Conversation(
        name="llama-3",
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        roles=("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="<|eot_id|>",
        stop_str="<|eot_id|>",
    ),
    "mistral": Conversation(
        name="mistral",
        system_template="<s> [INST] {system_message}\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.MISTRAL,
        sep=" ",
        sep2="</s>",
    ),
    "qwen-7b-chat": Conversation(
        name="qwen-7b-chat",
        system_template="{system_message}",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_str="<|im_end|>",
    ),
    "chatglm3": Conversation(
        name="chatglm3",
        system_template="<|system|>\n{system_message}",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
    ),
    "solar": Conversation(
        name="solar",
        system_template="### System:\n{system_message}",
        roles=("### User:", "### Assistant:"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="</s>\n",
    ),
    "mpt-7b-chat": Conversation(
        name="mpt-7b-chat",
        system_template="<|im_start|>system\n{system_message}<|im_end|>",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
    ),
    "mpt-30b-instruct": Conversation(
        name="mpt-30b-instruct",
        system_template="{system_message}",
        roles=("### Instruction:\n", "### Response:\n"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n\n",
    ),
    "vicuna_v1.1": Conversation(
        name="vicuna_v1.1",
        system_template="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    ),
}

def register_conv_template(template: Conversation, override: bool = False):
    if not override and template.name in conv_templates:
        raise ValueError(f"{template.name} has already been registered.")
    conv_templates[template.name] = template

def get_conv_template(name: str) -> Conversation:
    if name not in conv_templates:
        # Fallback to zero_shot if template name not found
        return conv_templates["zero_shot"].copy()
    return conv_templates[name].copy()
