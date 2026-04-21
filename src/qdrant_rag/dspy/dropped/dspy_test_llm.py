import dspy
from dotenv import load_dotenv
import litellm
import os
from typing import Literal

api_key = os.getenv("OPENAI_API_KEY")

print(f"Using OpenAI API key: {api_key[:6]}...")  # Print only the first 8 characters for security

if not api_key:

    raise RuntimeError("Missing OPENAI_API_KEY in the environment.")



if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    print(f"Using OpenAI API key: {api_key[:6]}...")  # Print only the first 8 characters for security
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in the environment.")

    # litellm.openrouter_key = api_key

    # lm = dspy.LM(
    #     model="openrouter/openai/gpt-4o-mini",
    #     api_key=api_key,
    #     api_base="https://openrouter.ai/api/v1",
    #     headers={
    #         "Authorization": f"Bearer {api_key}",
    #         "HTTP-Referer": "https://8bitoracle.ai",
    #         "X-Title": "8-Bit Oracle"
    #     }
    # )

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    lm = dspy.LM(
        model="openrouter/google/gemini-2.5-pro-preview",
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    dspy.configure(lm=lm)

        # Create and use a DSPy module with failover
    qa = dspy.ChainOfThought('question -> answer')
    response = qa(question="What is DSPy? Please give a brief explanation.")
    print(response)

    class Classify_response_type(dspy.Signature):
        """Classify wether the user is asking for a scientific rapport, a summary, a synthesis rapport or an opinion."""

        sentence: str = dspy.InputField()
        classification: Literal["scientific_rapport", "summary", "synthesis_rapport", "opinion"] = dspy.OutputField()
        confidence: float = dspy.OutputField()

    classify = dspy.Predict(Classify_response_type)
    classified_response = classify(sentence="Qu'est ce que R2D2 ?")
    print(classified_response)
    classified_response = classify(sentence="Rapport de Synthèse C24DST")
    print(classified_response)
