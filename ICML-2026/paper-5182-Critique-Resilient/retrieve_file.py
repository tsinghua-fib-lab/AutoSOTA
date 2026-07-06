
import os
import openai
FILE_ID = 'file-CHqc2ySz4GcusaiDVHvmAf'

import dotenv
dotenv.load_dotenv()

def retrieve_openai_file(file_id):
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    file_response = client.files.content(file_id)
    print(file_response.text)

if __name__ == "__main__":
    retrieve_openai_file(FILE_ID)