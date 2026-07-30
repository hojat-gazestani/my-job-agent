from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="Google/Gemma-4-31B-it")

print(llm.invoke("Hello"))
