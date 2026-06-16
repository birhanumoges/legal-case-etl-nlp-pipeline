import os
from openai import OpenAI
from rag.prompt_templates import LEGAL_QA_TEMPLATE
from utils.logger import get_logger

logger = get_logger(__name__)

class LegalGenerator:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set."
            )

        logger.info(f"Using API key: {api_key[:10]}...")

        self.client = OpenAI(api_key=api_key)

    def generate(self, question: str, context: str) -> str:
        prompt = LEGAL_QA_TEMPLATE.format(
            context=context,
            question=question
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert legal assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.exception("FULL LLM ERROR")
            return f"OpenAI Error: {str(e)}"