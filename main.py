#!.\.venv python

from groq import Groq
from datetime import datetime
from typing import Any, NoReturn
import json
from difflib import get_close_matches


class VitazAssistant:
    def __init__(self) -> None:
        self.client = Groq()
        self.history: list[str] = []
        self.similarity_threshold: float = 0.75
        with open("answer_database.json", "r") as f:
            self.answer_database = json.load(f)["questions_and_answers"]
    
    
    def generate_thoughts(self, message: str) -> str | None:
        return self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": 
                        f"""
                        You are the thoughts processing unit of Vitaz AI. 
                        Analyze the user's message and provide detailed thoughts 
                        about how to respond. Focus only on analysis and plannin
                        Use the history when relevant.
                        """
                },
                {
                    "role": "user",
                    "content": 
                        f"""
                        Generate thoughts about how to respond to: {message}
                        Context: {self.history}, {datetime.now():%c}
                        """
                }
            ],
            temperature=0.7,
            max_tokens=2048
        ).choices[0].message.content

    def generate_response(self, message: str, thoughts: str | None) -> str | None:
        return self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content":
                    """
                    You are Vitaz AI. Use the provided thoughts to create
                    an appropriate response. Output only the final response,
                    respond like a normal AI without quotes around your message.
                    """
                },
                {
                    "role": "user",
                    "content": 
                    f"""
                    Create a response based on these thoughts:
                    {thoughts}
                    For message: {message}
                    """
                }
            ],
            temperature=1.3,
            max_tokens=4096
        ).choices[0].message.content
    
    def find_answ(self, query: str) -> str | None:
    
        closest_matches = get_close_matches(query, self.answer_database.keys(), n=1, cutoff=self.similarity_threshold)
        if closest_matches:
            return self.answer_database[closest_matches[0]]
        return None

    def process_message(self, message: str) -> dict[str, Any]:
        try:
            thoughts: str | None = None
            response: str | None = None
            
            if not self.find_answ(message):
                thoughts: str | None = self.generate_thoughts(message)
                response: str | None = self.generate_response(message, thoughts)
                
                self.history.append(f"user: {message}, assistant: {response}")
                
                # Update the answer database
                if message:
                    self.answer_database[message] = response
                    with open("answer_database.json", "w") as f:
                        json.dump({"questions_and_answers": self.answer_database}, f, indent=4)
                else:
                    response = "Message cannot be empty."
            else:
                response = self.find_answ(message)
                thoughts = None
                
            return {
                "thoughts": thoughts,
                "response": response
            }
            
        except Exception as e:
            return {
                "thoughts": "Error processing request",
                "response": f"Oops! Something went wrong: {str(e)}"
            }


def main() -> NoReturn:
    vitaz: VitazAssistant = VitazAssistant()
    
    while True:
        response: dict[str, Any] = vitaz.process_message(input(": "))
        print(response["response"])

if __name__ == "__main__":
    main()
























































































