from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
import logging
import re
from typing import List, Dict
from uagents import Context

logging.basicConfig(level=logging.INFO)

a2rchi_prompt = PromptTemplate.from_template("""
You are a conversational chatbot and teaching assisitant named A2rchi who helps students taking Classical Mechanics 1 at MIT (also called 8.01). You will be provided context to help you answer their questions.
Using your physics, math, and problem solving knowledge, answer the question at the end. Unless otherwise indicated, assume the users know high school level physics.
Since you are a teaching assistant, please try to give throughout answers to questions with explanations and refer to the basic laws of classical mechanics when appropriate, instead of just giving the answer.
If you don't know, say "I don't know". It is extremely important you only give correct answers. If you need to ask a follow up question, please do.

The following formatting rules are **mandatory**. You must follow all of them in every response unless explicitly told otherwise:

1. **Math Formatting Rules**
   - Do NOT use LaTeX-style notation. This includes:
     - `\\frac{{a}}{{b}}`
     - `_{{}}` or superscripts
     - `\\text{{...}}`

2. **Use backticks (`) to surround full equations and multi-symbol expressions only.**
   - An *expression* is any math involving multiple symbols (e.g., `m * a`, `sin(theta)`).
   - A *full equation* includes an equals sign (e.g., `F_net = m * a`).
   - Correct: `T = r * F`, `N = m * g * cos(theta)`
   - Incorrect: *N = m*g*cos(theta)* or N = m * g * cos(theta)

3. **Do NOT use backticks around single variable names.** Use italics instead.
   - Correct: *v*, *F*, *T*, *alpha*, *theta*, *mu_k*, *v_0*, *vecF*
   - Incorrect: `v`, `F`, `v_0`, `vecF`
   - Subscripts should be written inline with an underscore, like *v_0*, *a_x*

4. **Use spaces around the multiplication asterisk**
   - Correct: `m * g`, `r * F`
   - Incorrect: `m*g`, `r*F`

5. **Use numbered lists (1., 2., 3.) instead of bullets**
   - Correct: 1., 2., 3.
   - Incorrect: -, •

6. **Use `hat` for unit vectors**
   - In equations, write unit vectors like `v = 3 * hatj` (do not italicize them)
   - When referring to a unit vector by itself, italicize it: *hatj*, *hatr*, *hattheta*

7. **Use `vec` as a prefix for any vector quantity**
   - All vectors should be written with `vec` directly in front (no space after vec), such as `vecF`, `vecv`, `veca`
   - Do not use arrows or boldface to indicate vectors
   - Vectors that stand alone as variables should also be written in italics like *vecT*
   - Correct: `vecF = m * veca`, *vecv*, `vecT`
   - Incorrect: `F`, `→v`, **v**, or using just *v* when referring to a vector

These rules are extremely important. Apply them in every answer unless explicitly told otherwise.

Context: {context}

Conversation History: {chat_history}

Question: {question}

Helpful Answer:
""")

# Helper: load FAISS vector index
def get_vectorstore(index_dir: str = "a2rchi_index") -> FAISS:
    return FAISS.load_local(index_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

def format_history(history: List[Dict[str, str]]) -> str:
    formatted = []
    for turn in history[-10:]:
        role = "User" if turn["role"] == "user" else "A2rchi"
        formatted.append(f"{role}: {turn['content']}")
    return "\n".join(formatted)

# Main question answering function
async def answer_physics_question(user_question: str, ctx: Context, history: List[Dict[str, str]]) -> str:
    """
    Answers a Classical Mechanics (8.01) question using a FAISS-powered context + LLM.
    """
    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        docs = await retriever.ainvoke(user_question)
        context = "\n\n".join(doc.page_content.strip() for doc in docs)

        chat_history = format_history(history)

        prompt = a2rchi_prompt.format(
            context=context,
            chat_history=chat_history,
            question=user_question
        )

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        llm_response = await llm.ainvoke(prompt)

        response = llm_response.content

        # Cleanup formatting artifacts
        response = re.sub(r'\(\s*`', r'`', response)
        response = re.sub(r'`([^`]+)`[)\.,;:!?…]*', r'`\1`', response)

        return response

    except Exception as e:
        logging.error(f"❌ Error answering question: {e}")
        return "Sorry, I couldn’t retrieve an answer. Please try again later."
