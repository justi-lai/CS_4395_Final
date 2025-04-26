from openai import OpenAI

MODEL_ID = "gemma-2-9b-it"
BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. You will be given a question and a context. Your task is to answer the question based on the context provided."

class OpenAIClient:
    def __init__(self, system_prompt=None, model_id=MODEL_ID, base_url=BASE_URL, api_key=API_KEY):
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
    
    def set_system_prompt(self, prompt):
        self.system_prompt = prompt
    
    def get_system_prompt(self):
        return self.system_prompt

    def generate_response(self, question, context):
        prompt = f"{self.system_prompt}\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content