#!/usr/bin/env python
# coding: utf-8

# LangChain's document loader, embedding, and vector store abstractions.

# In[2]:


# pip install langchain-community pypdf


# In[3]:


import getpass
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ["LANGSMITH_TRACING"] = "true"
if os.environ["LANGSMITH_API_KEY"] not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass()


# ### Documents and Document Loaders
# 
# LangChain implements a Document abstraction, which is intended to represent a unit of text and associated metadata. It has three attributes:
# 
# ```page_content```: a string representing the content;
# 
# ```metadata```: a dict containing arbitrary metadata;
# 
# ```id```: (optional) a string identifier for the document.

# In[4]:


from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata = {"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata = {"source": "mammal-pets-doc"},
    )
]


# Loading documents

# In[5]:


from langchain_community.document_loaders import PyPDFLoader

file_path = "nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
print(len(docs))


# In[6]:


print(f"{docs[0].page_content[:200]}\n")
print(f"{docs[0].metadata}")


# Splitting

# In[7]:


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, chunk_overlap = 200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
len(all_splits)

# set add_start_index=True so that the character index where each split Document
# starts within the initial Document is preserved as metadata attribute “start_index”.


# ### Embeddings
# Vector search is a common way to store and search over unstructured data (such as unstructured text). The idea is to store numeric vectors that are associated with the text. Given a query, we can embed it as a vector of the same dimension and use vector similarity metrics (such as cosine similarity) to identify related text.

# In[8]:


# There can be these warning

# IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
# from .autonotebook import tqdm as notebook_tqdm

# UserWarning: Could not download mistral tokenizer from Huggingface for calculating batch sizes. Set a Huggingface token via the HF_TOKEN environment variable to download the real tokenizer. Falling back to a dummy tokenizer that uses `len()`.
#   warnings.warn(

# To resolve these issue
# pip install --upgrade jupyter ipywidgets
# pip install transformers


# In[9]:


if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for MistralAI: ")

from langchain_mistralai import MistralAIEmbeddings

embeddings = MistralAIEmbeddings(model="mistral-embed")


# In[10]:


vector1 = embeddings.embed_query(all_splits[0].page_content)
vector2 = embeddings.embed_query(all_splits[1].page_content)
# embeddings.embed_query(text) converts the text into an embedding
# (a numerical vector representation)
assert len(vector1) == len(vector2)
# ensures that both embeddings have the same length.
print(f"Generated vectors of Length {len(vector1)}\n")
print(vector1[:10])


# ### Vector Stores
# LangChain VectorStore objects contain methods for adding text and Document objects to the store, and querying them using various similarity metrics. They are often initialized with embedding models, which determine how text data is translated to numeric vectors.

# In[11]:


# If not experinced with postgres , just use inmemoryDB
from langchain_postgres import PGVector

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection="postgresql+psycopg://postgres:104050@localhost:5432/pgvector",
)


# Having instantiated our vector store, we can now index the documents.

# In[12]:


ids  = vector_store.add_documents(documents=all_splits)


# ### Usage
# Embeddings typically represent text as a "dense" vector such that texts with similar meanings are geometrically close. This lets us retrieve relevant information just by passing in a question, without knowledge of any specific key-terms used in the document.

# Return documents based on similarity to a string query:

# In[13]:


results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

print(results[0])


# In[ ]:





# Async query:

# In[ ]:


async def perform_search():
    results = await vector_store.asimilarity_search("when was Nike incorporated")
    print(results[0])

# Call the async function
await perform_search()


# Return scores:

# In[18]:


# Note that providers implement different scores; the score here
# is a distance metric that varies inversely with similarity.

results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)


# Return documents based on similarity to an embedded query:

# In[19]:


embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")
results = vector_store.similarity_search_by_vector(embedding)
print(results[0])


# ### Retrievers
# 
# LangChain VectorStore objects do not subclass Runnable. LangChain Retrievers are Runnables, so they implement a standard set of methods (e.g., synchronous and asynchronous invoke and batch operations). Although we can construct retrievers from vector stores, retrievers can interface with non-vector store sources of data, as well (such as external APIs).

# In[20]:


from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1) #k denotes numbers of result to return

retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ]
)


# Vectorstores implement an as_retriever method that will generate a Retriever, specifically a VectorStoreRetriever. These retrievers include specific search_type and search_kwargs attributes that identify what methods of the underlying vector store to call, and how to parameterize them

# In[21]:


retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs={"k":1},
)


retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)

