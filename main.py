from inference import ModularInference

if __name__ == "__main__":
    system = ModularInference()

    prompts = [
        "What is overfitting in machine learning?",
        "Summarize the importance of renewable energy",
        "Write a Python function to reverse a linked list"
    ]

    for p in prompts:
        print("\n============================")
        print("PROMPT:", p)

        output = system.generate(p)

        print("OUTPUT:", output[:300])