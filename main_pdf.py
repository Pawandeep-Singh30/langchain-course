import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.1:8b", temperature=0.0)

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME_PDF"],
    embedding=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history, rewrite the follow-up question as a standalone question "
            "that can be used to search a document. Resolve pronouns like 'him', 'her', and 'that'. "
            "Output only the rewritten question, nothing else."
            "This is a follow-up. Answer about the same person/topic from the previous message. Do not introduce other historians unless the user asks.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_chain = contextualize_prompt | llm | StrOutputParser()

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant answering questions about a history lecture PDF.
Answer using only the provided PDF context and chat history.
If the PDF context does not contain enough information, say you don't know.
Keep answers clear and concise (2-5 sentences unless the user asks for more detail).
Use chat history for follow-up questions like "tell me more about that".""",
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            """PDF context:
{pdf_context}

Question: {question}""",
        ),
    ]
)
answer_chain = answer_prompt | llm | StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_search_query(question: str, chat_history: list) -> str:
    if not chat_history:
        return question
    return contextualize_chain.invoke(
        {"question": question, "chat_history": chat_history}
    )


def stream_answer(chain_input: dict) -> str:
    print("\nAnswer: ", end="", flush=True)
    full_response = ""
    for chunk in answer_chain.stream(chain_input):
        print(chunk, end="", flush=True)
        full_response += chunk
    print()
    return full_response


if __name__ == "__main__":
    chat_history = []

    print("\nHistory PDF Q&A")
    print("Commands: 'quit' to exit | 'clear' to reset chat history")
    print("Tip: ask follow-ups like 'Who is Thucydides?' then 'Tell me more about him'\n")

    while True:
        query = input("Ask a question: ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "clear":
            chat_history = []
            print("Chat history cleared.\n")
            continue

        search_query = get_search_query(query, chat_history)
        pdf_context = format_docs(retriever.invoke(search_query))

        chain_input = {
            "question": query,
            "chat_history": chat_history,
            "pdf_context": pdf_context,
        }

        answer = stream_answer(chain_input)

        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))

        if len(chat_history) > 12:
            chat_history = chat_history[-12:]
