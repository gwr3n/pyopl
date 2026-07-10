from pathlib import Path

from pyopl.genai.pyopl_generative import generative_feedback, generative_solve

TMP_DIR = Path(__file__).resolve().parents[1] / "tmp"


def test_generative_solve():
    prompt = (
        "A small inventory routing problem involves a company that must deliver a single product "
        "from a central warehouse to several retail stores over a planning horizon. "
        "Each store has a limited storage capacity and a known demand for each period. "
        "The company must decide how much inventory to deliver to each store and when, "
        "while minimizing the total cost of transportation and inventory holding, "
        "and ensuring that no store runs out of stock or exceeds its storage capacity."
    )
    model_file = str(TMP_DIR / "gen_pyopl_model.mod")
    data_file = str(TMP_DIR / "gen_pyopl_data.dat")
    assessment = generative_solve(prompt, model_file, data_file)
    print("Assessment of alignment:", assessment)


def test_generative_feedback():
    prompt = "Can you suggest improvements to the model to better handle variable transportation costs between the warehouse and each store?"
    model_file = str(TMP_DIR / "gen_pyopl_model.mod")
    data_file = str(TMP_DIR / "gen_pyopl_data.dat")
    feedback_result = generative_feedback(prompt, model_file, data_file)
    print("Feedback:", feedback_result.get("feedback", "No feedback"))
    print("Revised Model:\n", feedback_result.get("revised_model", ""))
    print("Revised Data:\n", feedback_result.get("revised_data", ""))


if __name__ == "__main__":
    test_solve = True
    test_feedback = False
    if test_solve:
        test_generative_solve()
    if test_feedback:
        test_generative_feedback()
