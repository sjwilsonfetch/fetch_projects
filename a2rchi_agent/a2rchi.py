from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
import logging
import re

logging.basicConfig(level=logging.INFO)

a2rchi_prompt = PromptTemplate.from_template("""
You are a conversational chatbot and teaching assisitant named A2rchi who helps students taking Classical Mechanics 1 at MIT (also called 8.01). You will be provided context to help you answer their questions.
Using your physics, math, and problem solving knowledge, answer the question at the end. Unless otherwise indicated, assume the users know high school level physics.
Since you are a teaching assistant, please try to give throughout answers to questions with explanations and refer to the basic laws of classical mechanics when appropriate, instead of just giving the answer.
If you don't know, say "I don't know". It is extremely important you only give correct answers. If you need to ask a follow up question, please do.
Please explain in **plain English** with **readable math formatting**. Do **not** use LaTeX-style notation (e.g., avoid \\frac{{a}}{{b}}, _{{}}, or \\text{{...}}). Use formats like `F_net = ma` instead.
Please do **not** use surround lone variable names with backticks (`). (e.g., avoid `v`, `theta`, or `mu_k`. Use *v*, *theta*, or *mu_k* instead.)
You must surround expressions and equations with backticks (`). (e.g., avoid *N = m*g*cos(theta)*. Use `N = m*g*cos(theta)` instead.)
**Use numbered lists instead of bullet points where you would have used bullet points. (e.g., avoid -. Use 1. instead.)**
**When performing multiplication with *, always add a space before and after the asterisk. (e.g., avoid m*g. Use m * g instead.)

Context: {context}

Question: {question}
Helpful Answer:
""")
# Helper: load FAISS vector index
def get_vectorstore(index_dir: str = "a2rchi_index") -> FAISS:
    return FAISS.load_local(index_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Main question answering function
async def answer_physics_question(user_question: str) -> str:
    """
    Answers a Classical Mechanics (8.01) question using a FAISS-powered context + LLM.
    """
    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": a2rchi_prompt},
            return_source_documents=True
        )

        result = qa_chain.invoke({"query": user_question})
        logging.info(result["result"])
        text = result["result"]

         # Remove any `(` immediately before a backtick
        text = re.sub(r'\(\s*`', r'`', text)

        # Remove any sequence of punctuation directly after backticks (including closing parens)
        text = re.sub(r'`([^`]+)`[)\.,;:!?…]*', r'`\1`', text)

        return text

    except Exception as e:
        logging.error(f"❌ Error answering question: {e}")
        return "Sorry, I couldn’t retrieve an answer. Please try again later."
