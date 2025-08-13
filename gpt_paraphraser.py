import openai
import os
from tf_idf import *
from tqdm import tqdm

openai.api_key = os.getenv("GPT_KEY")
model_name = "gpt-4o-mini-2024-07-18"

prompt = """
Your task is to paraphrase the provided text at the {level_name} level of modification.

{level_instructions}

**Universal Requirements:**
• **Meaning Preservation:** Maintain the exact same meaning, tone, and intent as the original text
• **Grammatical Fluency:** Produce natural, grammatically correct, and fluent English
• **JSON Format Compliance:** The input is a JSON object with 4 keys, each containing a list of text lines. Paraphrase ONLY the text content within each list while preserving the exact JSON structure, key names, and formatting

**Your Task:** Paraphrase each text line according to the {level_name} level instructions while maintaining the identical JSON structure.

**Text to paraphrase:**
{text}
"""

LOW_LEVEL = """
**Low Level Paraphrasing:**
• **Primary Action:** Replace key words with appropriate synonyms and alternative expressions
• **Structure Constraint:** Preserve the original sentence structure, word order, and grammatical patterns
• **Scope:** Make minimal, targeted word substitutions without altering sentence flow or organization
• **Example:** "The quick brown fox jumps over the lazy dog" → "The fast brown fox leaps over the sluggish dog"
"""

MEDIUM_LEVEL = """
**Medium Level Paraphrasing:**
• **Primary Action:** Substantially rewrite using varied vocabulary and moderate structural changes
• **Structure Flexibility:** Reorder clauses, rephrase expressions, and modify sentence components while maintaining similar logical progression
• **Scope:** Create noticeable differences through phrase restructuring, clause reordering, and varied grammatical constructions
• **Example:** "The quick brown fox jumps over the lazy dog" → "A fast, brown-colored fox leaps across a sluggish canine"
"""

HIGH_LEVEL = """
**High Level Paraphrasing:**
• **Primary Action:** Complete conceptual reconstruction with maximum structural freedom
• **Structure Flexibility:** Split/combine sentences, change voice (active ↔ passive), alter grammatical constructions, and reorganize information flow
• **Scope:** Create entirely different sentence patterns while conveying identical meaning through sophisticated rephrasing
• **Example:** "The quick brown fox jumps over the lazy dog" → "A sluggish canine was cleared by the swift leap of a brown fox"
"""

def paraphrase(text, level, level_name):
    response = openai.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert linguist AI, specializing in paraphrasing."},
                    {"role": "user", "content": prompt.format(
                        level_instructions = level,
                        text = text,
                        level_name = level_name
                    )}
                ],
    )

    return response.choices[0].message.content.strip()

# eval is to be done later on
def do_eval(text):
    try:
        return eval(text[7:-3])
    except Exception as e:
        print(e)
        print("Error")
        print(text)
        return text

def revw_paraphrase(input_folder, output_folder):
    input_ids = os.listdir(input_folder)
    output_ids = os.listdir(output_folder)

    for id in tqdm(input_ids):
        if id not in output_ids:
            try:
                print(f"Processing {id}")
                revw = json.load(open(f"{input_folder}/{id}"))
                if type(revw[id[:-5]])==dict:
                    para = {
                        "original":revw,
                        "low": paraphrase(revw[id[:-5]]['aspects'], LOW_LEVEL, "Low Level"),
                        "med": paraphrase(revw[id[:-5]]['aspects'], MEDIUM_LEVEL, "Medium Level"),
                        "high": paraphrase(revw[id[:-5]]['aspects'], HIGH_LEVEL, "High Level")
                    }
                    output_ids.append(id)
                    json.dump(para, open(f"{output_folder}/{id}","w"))

                elif type(revw[id[:-5]])==list:
                    para = {
                        "original":revw[id[:-5]][0],
                        "low": paraphrase(revw[id[:-5]][0]['aspects'], LOW_LEVEL, "Low Level"),
                        "med": paraphrase(revw[id[:-5]][0]['aspects'], MEDIUM_LEVEL, "Medium Level"),
                        "high": paraphrase(revw[id[:-5]][0]['aspects'], HIGH_LEVEL, "High Level")
                    }
                    output_ids.append(id)
                    json.dump(para, open(f"{output_folder}/{id}","w"))
            except:   
                print(f"Error at {id}")
        else:
            print(f"Skipping {id}")


if __name__=='__main__':
    input_path = "/Data/sandeep/Vardhan/Journal/Dataset/Aspects"
    output_path = "/Data/sandeep/Vardhan/Journal/Dataset/Aspects/paraphrase"
    revw_paraphrase(f"{input_path}/ai_iclr",f"{output_path}/ai_iclr")
    revw_paraphrase(f"{input_path}/human_iclr",f"{output_path}/human_iclr")
    # revw_paraphrase(f"{input_path}/ai_neur",f"{output_path}/ai_neur")
    revw_paraphrase(f"{input_path}/ai_org",f"{output_path}/ai_org")
    # revw_paraphrase(f"{input_path}/human_neur",f"{output_path}/human_neur")
    revw_paraphrase(f"{input_path}/human_org",f"{output_path}/human_org")