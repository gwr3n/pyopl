from enum import Enum

def test_generative_solve():
    prompt = (
        "A small inventory routing problem involves a company that must deliver a single product "
        "from a central warehouse to several retail stores over a planning horizon. "
        "Each store has a limited storage capacity and a known demand for each period. "
        "The company must decide how much inventory to deliver to each store and when, "
        "while minimizing the total cost of transportation and inventory holding, "
        "and ensuring that no store runs out of stock or exceeds its storage capacity."
    )
    model_file = "tmp/gen_pyopl_model.mod"
    data_file = "tmp/gen_pyopl_data.dat"
    assessment = generative_solve(prompt, model_file, data_file)
    print("Assessment of alignment:", assessment)


def test_generative_feedback():
    prompt = "Can you suggest improvements to the model to better handle variable transportation costs between the warehouse and each store?"
    model_file = "tmp/gen_pyopl_model.mod"
    data_file = "tmp/gen_pyopl_data.dat"
    feedback_result = generative_feedback(prompt, model_file, data_file)
    print("Feedback:", feedback_result.get("feedback", "No feedback"))
    print("Revised Model:\n", feedback_result.get("revised_model", ""))
    print("Revised Data:\n", feedback_result.get("revised_data", ""))

class GenerativeProvider(Enum):
        OPENAI = "openai"
        GEMINI = "gemini"
        OLLAMA = "ollama"

if __name__ == "__main__":

    genai = GenerativeProvider.OPENAI

    if genai == GenerativeProvider.GEMINI:
        # Use Gemini
        from pyopl.pyopl_generative_gemini import generative_feedback, generative_solve
    elif genai == GenerativeProvider.OPENAI:
        # Use OpenAI
        from pyopl.pyopl_generative_openai import generative_feedback, generative_solve
    elif genai == GenerativeProvider.OLLAMA:
        # Use Ollama
        from pyopl.pyopl_generative_ollama import generative_feedback, generative_solve

    test_solve = True
    test_feedback = False
    if test_solve:
        test_generative_solve()
    if test_feedback:
        test_generative_feedback()
