import ollama

def answer_question(chunks, user_query):
    context_text = "\n\n---\n\n".join([
        f"Source: {chunk['source']} (Page {chunk['page']})\n{chunk['text']}" 
        for chunk in chunks
    ])
    print(context_text)
    print("DLL")
    system_prompt = f"""
    You are a precise and helpful AI assistant. Your task is to answer the user's question using ONLY the provided context below.
    
    Rules:
    - If the answer is not contained in the context, do not guess or use outside knowledge. Simply say, "I cannot answer this based on the provided documents."
    - Always cite the source file and page number when you provide facts.
    
    CONTEXT:
    {context_text}
    """
    
    print("🤖 Generating answer via local Ollama (llama3.1:8b)...")
    response = ollama.chat(
        model='llama3.1:8b',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_query}
        ],
        stream=True
    )

    print("Answer: ", end="", flush=True)
    
    full_answer = "" 
    
    for chunk in response:
        word = chunk['message']['content']
        print(word, end='', flush=True)
        full_answer += word 
        
    print("\n") 
    
   
    return full_answer
    
