import os
import argparse
from dotenv import load_dotenv

from agent import ReActAgent


load_dotenv()


BENCHMARK_QUESTIONS = [
    "What fraction of Japan's population is Taiwan's population as of 2025?",
    "Compare the main display specs of iPhone 15 and Samsung S24.",
    'Who is the CEO of the startup "Morphic" AI search?'
]


def run_interactive_mode(agent: ReActAgent) -> None:
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Assistant: Goodbye!")
            break

        if not user_input:
            continue

        try:
            response = agent.run(user_input)
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"Assistant Error: {str(e)}\n")


def run_benchmark_mode(model_name: str, debug_mode: bool) -> None:
    print("===================================")
    print(" Assignment 2 Benchmark Mode ")
    print("===================================\n")

    agent = ReActAgent(
        model_name=model_name,
        max_steps=5,
        debug=debug_mode
    )

    for idx, question in enumerate(BENCHMARK_QUESTIONS, start=1):
        print(f"********** BENCHMARK TASK {idx} **********")
        print(f"Question: {question}\n")

        try:
            answer = agent.run(question)
            print(f"\nFinal Answer: {answer}")
        except Exception as e:
            print(f"\nAssistant Error: {str(e)}")

        print("\n" + "=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Assignment 2 ReAct CLI Agent")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logs"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the 3 required benchmark questions"
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=5,
        help="Maximum number of ReAct steps in interactive mode"
    )

    args = parser.parse_args()

    debug_mode = args.debug
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in .env")

    print("===================================")
    print(" ReAct CLI Agent ")
    print("===================================")

    if debug_mode:
        print("[DEBUG MODE ENABLED]")

    if args.benchmark:
        print("[MODE] Benchmark\n")
        run_benchmark_mode(model_name=model_name, debug_mode=debug_mode)
        return

    print("[MODE] Interactive\n")

    agent = ReActAgent(
        model_name=model_name,
        max_steps=args.max_steps,
        debug=debug_mode
    )

    run_interactive_mode(agent)


if __name__ == "__main__":
    main()