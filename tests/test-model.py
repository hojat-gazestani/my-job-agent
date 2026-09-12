from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="AliBaba/Qwen3.6-27B")

print(llm.invoke("Hello"))
