import os
from dotenv import load_dotenv

# this file will be used to load all the api keys from the .env file
load_dotenv()

class Keys:
    pinecone_api_key = os.getenv("pinecone_api_key")
    pinecone_index = os.getenv("pinecone_index")
    hackrx_token = os.getenv("hackrx_token")
    gemini_api_key = os.getenv("gemini_api_key")
    pinecone_env = "asia-southeast1-gcp" 
    oss_api_key = os.getenv("openrouter_api_key")

key = Keys()