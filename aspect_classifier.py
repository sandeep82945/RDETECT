from gemini import *
import json
import os
from tqdm import tqdm
import tempfile

PROMPT_TEMPLATE = """
You are an expert at analyzing scientific peer reviews. Your task is to extract specific parts of a review that discuss a given aspect.

**Target Aspect:** {aspect_name}
**Aspect Definition:** {aspect_definition}

**What to look for:**
{aspect_indicators}

**Examples of this aspect:**
{aspect_examples}

### Instructions:
1. Read the review carefully and identify sentences or phrases that directly address the specified aspect
2. Extract complete sentences or coherent phrases (not just fragments)
3. Include context if a sentence references the aspect indirectly
4. Maintain the original wording exactly as written
5. If multiple sentences discuss the aspect, extract each as a separate list item

### Response Format:
Return a JSON array containing the extracted text segments. If no relevant content is found, return an empty array `[]`.

**Review to analyze:**
"{review}"
"""

MOTIVATION = {
  "aspect_name": "Motivation/Impact",
  "aspect_definition": "Comments about the importance, significance, or potential impact of the research problem, methodology, or findings. This includes statements about practical applications, theoretical contributions, or relevance to the field.",
  "aspect_indicators": [
    "Importance or significance of the problem",
    "Practical applications or usefulness",
    "Theoretical contributions or insights", 
    "Relevance to practitioners or researchers",
    "Potential for future work or building upon",
    "Impact on the field or domain",
    "Novelty or originality of the approach"
  ],
  "aspect_examples": [
    "The issue researched in this work is of significance because understanding the predictive uncertainty of a deep learning model has both theoretical and practical value.",
    "The method seems limited in both practical usefulness and enlightenment to the reader.",
    "This work addresses an important gap in the literature by providing a unified framework for...",
    "The proposed approach has significant implications for real-world applications in autonomous systems.",
    "While the technical contribution is sound, the practical impact appears limited given existing solutions."
  ]
}

ORIGINALITY = {
  "aspect_name": "Originality",
  "aspect_definition": "Comments about the novelty, originality, or uniqueness of the research topic, technique, methodology, or insights presented in the paper.",
  "aspect_indicators": [
    "Novel or new research topics",
    "Innovative techniques or methodologies",
    "Unique insights or contributions",
    "Comparison with existing work regarding novelty",
    "Originality of the approach or solution",
    "Incremental vs. significant contributions",
    "Similarity to previous work",
    "Creative or innovative aspects"
  ],
  "aspect_examples": [
    "Novel addressing scheme as an extension to NTM.",
    "The reviewer believes that the idea of the paper is similar to the one in [1].",
    "The proposed approach introduces a completely new paradigm for handling sequential data.",
    "While the individual components are known, their combination is novel and well-motivated.",
    "The work lacks novelty as it merely combines existing techniques without significant innovation.",
    "The paper presents an original solution to a well-known problem in the field."
  ]
}

SOUNDNESS = {
  "aspect_name": "Soundness/Correctness",
  "aspect_definition": "Comments about the technical soundness, correctness, and validity of the proposed approach, methods, or claims made in the paper.",
  "aspect_indicators": [
    "Technical soundness of the approach",
    "Correctness of methods or algorithms",
    "Validity of claims and conclusions",
    "Convincing evidence or support for claims",
    "Logical consistency of arguments",
    "Rigor of theoretical analysis",
    "Identification of flaws or errors",
    "Reliability of the proposed solution"
  ],
  "aspect_examples": [
    "Illustrations using simulated data and real data are also very clear and convincing.",
    "The proposed method is sensible and technically sound.",
    "The theoretical analysis contains several logical gaps that undermine the main claims.",
    "The experimental validation provides strong evidence supporting the authors' hypothesis.",
    "There are concerns about the correctness of the mathematical derivations in Section 3.",
    "The approach is theoretically grounded and the implementation appears to be correct."
  ]
}

CLARITY = {
  "aspect_name": "Clarity",
  "aspect_definition": "Comments about the clarity of presentation, writing quality, organization, and overall readability of the paper for a well-prepared reader.",
  "aspect_indicators": [
    "Writing quality and style",
    "Organization and structure of the paper",
    "Clarity of explanations and descriptions",
    "Readability and comprehension",
    "Presentation of figures, tables, and results",
    "Logical flow of ideas",
    "Accessibility to the target audience",
    "Ambiguous or unclear sections"
  ],
  "aspect_examples": [
    "The paper is well-written and easy to follow.",
    "The presentation of the results is not very clear.",
    "The authors should improve the clarity of Section 4 as it is difficult to understand the main contributions.",
    "The figures are well-designed and effectively illustrate the proposed method.",
    "The mathematical notation is inconsistent and makes the paper hard to follow.",
    "The paper would benefit from better organization and clearer section transitions."
  ]
}

ASPECTS = [MOTIVATION, ORIGINALITY, SOUNDNESS, CLARITY]

def get_aspect(revw, aspect_template, simple=False):
    response = gemini_generate(PROMPT_TEMPLATE.format(
        aspect_name = aspect_template['aspect_name'],
        aspect_definition = aspect_template[ 'aspect_definition'],
        aspect_indicators = aspect_template[ 'aspect_indicators'],
        aspect_examples = aspect_template[ 'aspect_examples'],
        review = revw
    ),
        simple=simple)

    return response

def get_all_aspects(revw, simple=False):
    asp_dict = {}
    for asp in ASPECTS:
        asp_dict[asp['aspect_name']] = get_aspect(revw, asp, simple)

    return asp_dict

def revw_aspects(input_file,output_path , isHuman=False):
    reviews = json.load(open(input_file))
    print(f"+++++++++++++++++ Processing File {input_file} ++++++++++++++++++++=")
    i=0

    for (id, revw) in tqdm(reviews.items()):
        new_revw = {}
        done = os.listdir(output_path)
        
        if f"{id}.json" not in done:
            print(f"Processing {id}")
            try:
              new_revw[id] = {"review":revw, "aspects":get_all_aspects(revw, True)}

              if isHuman:  # just take the first review
                  new_revw[id] = {"review":revw[0], "aspects":get_all_aspects(revw[0], True)}

              
              json.dump(new_revw, open(f"{output_path}{id}.json","w"))
            except:
                print(f"Error at {id}")
        else:
            print(f"Skipping {id}")     

        i+=1
        # if i==200:
            # break      

def safe_save_json(data, path):
    temp_path = path + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(temp_path, path)

if __name__=='__main__':
    input_folder = "/Data/sandeep/Vardhan/Journal/AI-Review-Detection/Dataset/"
    input_folder = "/Data/sandeep/Vardhan/Journal/Dataset/"
    output_folder = "/Data/sandeep/Vardhan/Journal/Dataset/Aspects/"

    revw_aspects(f"{input_folder}org_gpt.json", f"{output_folder}ai_org/")
    revw_aspects(f"{input_folder}org_human.json", f"{output_folder}human_org/", True)

    # revw_aspects(f"{input_folder}ai_iclr.json", f"{output_folder}ai_iclr/")
    # revw_aspects(f"{input_folder}human_iclr.json", f"{output_folder}human_iclr/", True)
    
    # revw_aspects(f"{input_folder}ai_neur.json", f"{output_folder}ai_neur/")
    # revw_aspects(f"{input_folder}human_neur.json", f"{output_folder}human_neur/", True)
