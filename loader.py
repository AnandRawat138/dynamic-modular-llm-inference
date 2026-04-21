import torch
import os
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from utils.config import MODEL_NAME, LOAD_IN_8BIT, MODULE_DIR, CACHE_SIZE


# ✅ BASE MODEL LOADER (THIS WAS MISSING)
def load_base_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = {}
    if device == "cuda" and LOAD_IN_8BIT:
        kwargs["load_in_8bit"] = True
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        **kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return model, tokenizer


# ✅ MODULE LOADER (MULTI-ADAPTER SYSTEM)
class ModuleLoader:
    def __init__(self, base_model):
        self.base_model = base_model
        self.cache = OrderedDict()
        self.active_adapters = []

    def load_module(self, module_name):
        if module_name in self.cache:
            self.cache.move_to_end(module_name)
            return

        path = os.path.join(MODULE_DIR, module_name)

        if len(self.cache) == 0:
            # First adapter wraps base model
            self.base_model = PeftModel.from_pretrained(
                self.base_model,
                path,
                adapter_name=module_name
            ).to(self.base_model.device)
        else:
            # Add adapter without replacing
            self.base_model.load_adapter(
                path,
                adapter_name=module_name
            )

        self.cache[module_name] = True

        if len(self.cache) > CACHE_SIZE:
            removed, _ = self.cache.popitem(last=False)
            print(f"Evicting adapter: {removed}")
            self.base_model.delete_adapter(removed)

    def activate_adapters(self, module_names):
        # Always use only ONE adapter
        adapter = module_names[0]
        self.active_adapters = [adapter]

        self.base_model.set_adapter(adapter)

    def get_model(self):
        return self.base_model