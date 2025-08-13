import google.generativeai as genai
import os


genai.configure(api_key=os.getenv("GEMINI_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

def gemini_generate(prompt, simple=False):
    if simple:
        return model.generate_content(prompt).text
    
    return model.generate_content(prompt)
