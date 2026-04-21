from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from utils.config import MODEL_NAME, MODULE_DIR

def create_lora_adapter(name="qa_adapter"):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    peft_model = get_peft_model(model, lora_config)

    save_path = f"{MODULE_DIR}/{name}"
    peft_model.save_pretrained(save_path)

    print(f"Adapter saved at {save_path}")

if __name__ == "__main__":
    create_lora_adapter("qa_adapter")
    create_lora_adapter("summarization_adapter")
    create_lora_adapter("code_adapter")
