def extract_aspects(index, docs):
    aspect= ["Motivation/Impact", "Originality", "Soundness/Correctness","Clarity"][index]
    asp = {}
    
    for (id,revw) in docs.items():
        try:
            asp[id] = " ".join(eval(revw['aspects'][aspect][7:-3]))
        except: 
            asp[id] = revw['aspects'][aspect]
            print(f"Error at {id}")
        
    return asp