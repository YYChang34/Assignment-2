import os
import re
from typing import Optional, Tuple

from openai import OpenAI

from tools import search_web


SYSTEM_PROMPT = """
You are a general-purpose ReAct agent.

You must solve user questions by using a Thought -> Action -> Observation loop.

You have access to exactly one tool:

Search["query"]
- Use this tool to search the web for current factual information.

Rules:
1. Always think step by step.
2. For multi-part or numerical questions, break the task into smaller sub-problems first.
3. If the previous search result is missing, weak, or irrelevant, reflect and try a better query.
4. Do not repeat the exact same search query unless there is a very strong reason.
5. If you already have enough evidence, output a final answer.
6. Never invent an Observation. Observations only come from the tool.
7. For comparison questions, retrieve ALL key specs for each item before answering.
   For display comparisons, always include: screen size, panel type, resolution, refresh rate, and peak brightness.
8. For questions asking about a specific person holding a role (CEO, founder, director, etc.):
   if the search results mention MORE THAN ONE individual in any leadership capacity
   for that company, you MUST perform one additional targeted search to confirm the
   correct person. Do NOT skip this step by reasoning "one seems more likely" —
   always verify with an explicit follow-up search.
9. Action is ONLY for Search. If you need to calculate or reason through numbers,
   perform ALL calculations inside your Thought, then output Final Answer directly.
   Never write "Action: Calculate" or any non-Search action.
10. Use exactly one of these two output formats:

Format A:
Thought: your reasoning here
Action: Search["your search query here"]

Format B:
Thought: your reasoning here
Final Answer: your final response here

Example 1:
User Question: What fraction of Japan's population is Taiwan's population?
Thought: I should first find Japan's population, then Taiwan's population, and finally compute the fraction.
Action: Search["Japan population 2025"]

Observation: {"ok": true, "query": "Japan population 2025", "results": [{"title": "Japan Population 2025", "content": "Japan population is about 123,103,479 in 2025."}]}

Thought: Now I have Japan's population. Next I need Taiwan's population.
Action: Search["Taiwan population 2025"]

Observation: {"ok": true, "query": "Taiwan population 2025", "results": [{"title": "Taiwan Population 2025", "content": "Taiwan population is about 23,404,000 in 2025."}]}

Thought: I now have both populations. I should calculate Taiwan / Japan and present the result clearly.
Final Answer: Taiwan's population is about 0.19 of Japan's population in 2025, or roughly 19%.

Example 2:
User Question: Who is the CEO of the AI startup "Perplexity"?
Thought: I will search for the CEO of Perplexity directly.
Action: Search["Perplexity AI CEO"]

Observation: {"ok": true, "results": [{"content": "Aravind Srinivas is the CEO of Perplexity AI."}, {"content": "Andy Konwinski is a co-founder of Perplexity."}]}

Thought: The results mention both Aravind Srinivas (CEO) and Andy Konwinski (co-founder). Since more than one individual appears in a leadership capacity, I must perform a follow-up search before concluding — I cannot skip this step even if one name seems more prominent.
Action: Search["Perplexity AI founder CEO Aravind Srinivas"]

Observation: {"ok": true, "results": [{"content": "Aravind Srinivas is the co-founder and CEO of Perplexity AI, an AI-powered answer engine."}]}

Thought: Confirmed. Aravind Srinivas is both co-founder and CEO of Perplexity AI.
Final Answer: The CEO of Perplexity AI is Aravind Srinivas, who co-founded the company.

Be careful:
- For company leadership questions, if the first search is weak or conflicting, try alternative queries such as founders, leadership, about page, or site-specific searches.
- Be honest. If you cannot find reliable evidence after several attempts, say you do not know.
""".strip()


class ReActAgent:
    def __init__(self, model_name: str = "gpt-4o-mini", max_steps: int = 5, debug: bool = False):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.max_steps = max_steps
        self.debug = debug

        self.history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def _log(self, message: str) -> None:
        if self.debug:
            print(message)

    def _extract_action(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract Action: Search["..."]
        Returns (tool_name, tool_input) or (None, None)
        """
        pattern = r'Action:\s*Search\["(.*?)"\]'
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None, None
        return "Search", match.group(1).strip()

    def _extract_final_answer(self, text: str) -> Optional[str]:
        pattern = r"Final Answer:\s*(.*)"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None
        return match.group(1).strip()

    def _call_llm(self) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.history,
            temperature=0,
            stop=["Observation:"]
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    def run(self, user_input: str) -> str:
        self._log(f"\n[USER INPUT] {user_input}")

        self.history.append({
            "role": "user",
            "content": f"User Question: {user_input}"
        })

        for step in range(1, self.max_steps + 1):
            self._log(f"\n========== STEP {step} ==========")

            llm_output = self._call_llm()
            self._log(f"[LLM OUTPUT]\n{llm_output}")

            final_answer = self._extract_final_answer(llm_output)
            if final_answer:
                self.history.append({
                    "role": "assistant",
                    "content": llm_output
                })
                self._log(f"[FINAL ANSWER] {final_answer}")
                return final_answer

            action_name, action_input = self._extract_action(llm_output)
            if action_name == "Search" and action_input:
                self.history.append({
                    "role": "assistant",
                    "content": llm_output
                })

                self._log(f"[ACTION] {action_name}")
                self._log(f"[ACTION INPUT] {action_input}")

                observation = search_web(action_input)

                self._log(f"[OBSERVATION]\n{observation}")

                self.history.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
                continue

            fallback_answer = (
                "I could not parse a valid Action or Final Answer from the model output."
            )

            self.history.append({
                "role": "assistant",
                "content": llm_output
            })

            self._log(f"[PARSER ERROR] {fallback_answer}")
            return fallback_answer

        fallback_answer = (
            f"I reached the maximum number of steps ({self.max_steps}) without finishing."
        )

        self._log(f"[MAX STEP REACHED] {fallback_answer}")

        self.history.append({
            "role": "assistant",
            "content": f"Final Answer: {fallback_answer}"
        })

        return fallback_answer