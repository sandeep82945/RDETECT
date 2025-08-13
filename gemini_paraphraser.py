from gemini import *
import json
from tqdm import tqdm
import os

prompt = """
Your task is to paraphrase the provided text at the {level_name} level of modification.

{level_instructions}

**Universal Requirements:**
• **Meaning Preservation:** Maintain the exact same meaning, tone, and intent as the original text
• **Grammatical Fluency:** Produce natural, grammatically correct, and fluent English

**Your Task:** Paraphrase each text line according to the {level_name} level instructions.

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

def paraphrase(text):
    """Paraphrase the text in 3 different levels: low, medium, high"""


    # Just do the high paraphrasing for now


    return {
        # "low": gemini_generate(prompt=prompt.format(
        #                     level_instructions = LOW_LEVEL,
        #                     text = text,
        #                     level_name = "Low Level"
        #                 ),simple=True),
        # "med": gemini_generate(prompt=prompt.format(
        #                     level_instructions = MEDIUM_LEVEL,
        #                     text = text,
        #                     level_name = "Medium Level"
        #                 ),simple=True),
        "high": gemini_generate(prompt=prompt.format(
                        level_instructions = HIGH_LEVEL,
                        text = text,
                        level_name = "High Level"
                    ),simple=True)
    }

def domain_paraphraser(data,folder_path):
    done = [a[:-5] for a in os.listdir(folder_path)]

    for id in tqdm(list(data.keys())):
        if id not in done:
            print(f"Processing {id}")
            if (type(data[id])==list):
                revw = data[id][0]
            else:
                revw = data[id]
            revw_para = paraphrase(revw)
            json.dump(revw_para, open(f"{folder_path}/{id}.json","w"))

        else:
            print(f"Skipping {id}")

def paraphrase_(text):
    """Paraphrase the text in 3 different levels: low, medium, high"""
    return {
        "low": gemini_generate(prompt=prompt.format(
                            level_instructions = LOW_LEVEL,
                            text = text,
                            level_name = "Low Level"
                        ),simple=True),
        "med": gemini_generate(prompt=prompt.format(
                            level_instructions = MEDIUM_LEVEL,
                            text = text,
                            level_name = "Medium Level"
                        ),simple=True)
    }

def paraphrase_multiple(text):
    """To paraphrase the reviews multiple times"""
    return gemini_generate(f"""You are a highly advanced AI specializing in paraphrasing and text rewriting. Your task is to paraphrase the given review line by line \n {text}. """, simple=True)

def domain_paraphraser_(data,folder_path, domain):
    done =[a[:-5] for a in os.listdir(folder_path)]

    test_keys = json.load(open("/Data/sandeep/Vardhan/Journal/Dataset/test_keys.json"))

    for id in tqdm(test_keys[domain]):
        yup_id = f"{id}"
        # if len(json.load(open(f"{folder_path}/{id}.json")))<1500:
        if yup_id not in done:
            print(f"Processing {id}")
            if (type(data[id])==list):
                revw = data[id][0]
            else:
                revw = data[id]
            # revw_para = paraphrase_(revw)   # for paraphrasing in low, med, high
            revw_para = paraphrase_multiple(revw)
            json.dump(revw_para, open(f"{folder_path}/{id}.json","w"))


def load(path):
    data = {}
    for file in os.listdir(path):
        data[file[:-5]] = json.load(open(os.path.join(path, file)))
    return data

if __name__=='__main__':
    # ai_org = json.load(open("/Data/sandeep/Vardhan/Journal/Dataset/org_gpt.json"))
    # human_org = json.load(open("/Data/sandeep/Vardhan/Journal/Dataset/org_human.json"))

    # folder_path = "/Data/sandeep/Vardhan/Journal/AI-Review-Detection/Dataset"
    # folders =["ai_iclr","ai_neur","human_iclr","human_neur"]

    # ai_iclr = json.load(open(f"{folder_path}/{folders[0]}.json"))
    # ai_neur = json.load(open(f"{folder_path}/{folders[1]}.json"))
    # human_iclr = json.load(open(f"{folder_path}/{folders[2]}.json"))
    # human_neur = json.load(open(f"{folder_path}/{folders[3]}.json"))

    
    # domain_paraphraser(human_org, "/Data/sandeep/Vardhan/Journal/Dataset/Aspects/gemini_paraphrase/human_org")
    # domain_paraphraser(ai_org, "/Data/sandeep/Vardhan/Journal/Dataset/Aspects/gemini_paraphrase/ai_org")

    # domain_paraphraser(human_iclr, "/Data/sandeep/Vardhan/Journal/Dataset/Aspects/gemini_paraphrase/human_iclr")
    # domain_paraphraser(ai_iclr, "/Data/sandeep/Vardhan/Journal/Dataset/Aspects/gemini_paraphrase/ai_iclr")

    # domain_paraphraser(human_neur, "/Data/sandeep/Vardhan/Journal/Dataset/Aspects/gemini_paraphrase/human_neur")
    # domain_paraphraser(ai_neur, "/Data/sandeep/Vardhan/Journal/Dataset/Aspects/gemini_paraphrase/ai_neur")

    # folder_path = "/Data/sandeep/Vardhan/Journal/AI-Review-Detection/Dataset"
    # folders =["ai_iclr","ai_neur","human_iclr","human_neur"]

    # # # Paraphrasing both low and med
    # domain_paraphraser_(human_org, "/Data/sandeep/Vardhan/Journal/Dataset/Aspects/gemini_paraphrase/med_low/human_org", domain="org")
    # domain_paraphraser_(ai_org, "/Data/sandeep/Vardhan/Journal/Dataset/Aspects/gemini_paraphrase/med_low/ai_org", domain="org")

    # domain_paraphraser_(human_iclr, "/Data/sandeep/Vardhan/Journal/Dataset/Aspects/gemini_paraphrase/med_low/human_iclr", domain="iclr")
    # domain_paraphraser_(ai_iclr, "/Data/sandeep/Vardhan/Journal/Dataset/Aspects/gemini_paraphrase/med_low/ai_iclr", domain="iclr")

    # domain_paraphraser_(human_neur, "/Data/sandeep/Vardhan/Journal/Dataset/Aspects/gemini_paraphrase/med_low/human_neur", domain="neur")
    # domain_paraphraser_(ai_neur, "/Data/sandeep/Vardhan/Journal/Dataset/Aspects/gemini_paraphrase/med_low/ai_neur", domain="neur")

    # # Paraphrasing both multiple times
    folder_path = "/Data/sandeep/Vardhan/Journal/Dataset/Aspects/multiple_paraphrase"

    ai_org = load(f"{folder_path}/ai_org/4")
    human_org = load(f"{folder_path}/human_org/4")

    folders =["ai_iclr","ai_neur","human_iclr","human_neur"]

    ai_iclr = load(f"{folder_path}/{folders[0]}/4")
    ai_neur = load(f"{folder_path}/{folders[1]}/4")
    human_iclr = load(f"{folder_path}/{folders[2]}/4")
    human_neur = load(f"{folder_path}/{folders[3]}/4")

    domain_paraphraser_(human_org, f"{folder_path}/human_org/5", domain="org")
    domain_paraphraser_(ai_org, f"{folder_path}/ai_org/5", domain="org")

    domain_paraphraser_(human_iclr, f"{folder_path}/human_iclr/5", domain="iclr")
    domain_paraphraser_(ai_iclr, f"{folder_path}/ai_iclr/5", domain="iclr")

    domain_paraphraser_(human_neur, f"{folder_path}/human_neur/5", domain="neur")
    domain_paraphraser_(ai_neur, f"{folder_path}/ai_neur/5", domain="neur")