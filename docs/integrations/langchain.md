# LangChain Integration

Use LongParser as a document loader in any LangChain RAG pipeline.

## Install

```bash
pip install "longparser[langchain]"
```

## DocumentLoader

```python
from longparser.integrations.langchain import LongParserLoader

loader = LongParserLoader(
    file_path="report.pdf",
    chunk_max_tokens=512,
)

documents = loader.load()
# Returns List[langchain_core.documents.Document]
```

Each document has:
- `page_content` — chunk text
- `metadata.chunk_id` — unique chunk ID
- `metadata.page_numbers` — source page numbers
- `metadata.chunk_type` — `section` | `table` | `equation` | etc.
- `metadata.section_path` — hierarchy path

## With FAISS

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from longparser.integrations.langchain import LongParserLoader

docs = LongParserLoader("report.pdf").load()
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

results = vectorstore.similarity_search("key findings", k=5)
```

## With Chroma

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

docs = LongParserLoader("report.pdf").load()
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
```

## Streaming QA Chain

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-5.3"),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
)

answer = qa.invoke("What are the revenue figures?")
print(answer["result"])
```
