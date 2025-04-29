from utility import retrieval
from transformers import pipeline

# Correct pipeline usage
generator = pipeline("text2text-generation", model="facebook/bart-large", max_new_tokens=300)

def run(input_query):
    context_chunks = retrieval.retrieve(input_query)
    context = " ".join(context_chunks)

    # 🔐 Fallback if no context is usable
    if not context.strip() or context.strip() == "No relevant context found.":
        return "I don't know. This topic is not covered in the provided paper."

    # ✅ Correct prompt
    prompt = f"""Answer the question below using ONLY the information provided in the context. 
If the context does not contain the answer, say "I don't know." 

Context:
{context}

Question: {input_query}
Answer:"""

    response = generator(prompt)[0]['generated_text']
    return response

