#!/usr/bin/env python
# coding: utf-8

# In[16]:


import getpass
import os 

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(prompt="Enter your Langsmith API key (optional):")

if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(prompt="Enter your Langsmith Project Name(defautl = 'default'):")

    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = getpass.getpass(prompt="Enter your Mistral API key:")


# In[5]:


from langchain.chat_models import init_chat_model
model = init_chat_model("mistral-large-latest",model_provider="mistralai")


# In[8]:


from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the follwoing from English to french"),
    HumanMessage("Good Evening")
]

model.invoke(messages)


# LangChain also supports chat model inputs via strings or OpenAI format. The following are equivalent:

# In[9]:


model.invoke("Hello")
model.invoke([{"role":"user", "content":"Hello"}])
model.invoke([HumanMessage("Hello")])


# Streaming

# In[10]:


for tokens in model.stream(messages):
    print(tokens.content, end="|")


# ### Prompt Templates
# Prompt templates help to translate user input and parameters into instructions for a language model. This can be used to guide a model's response, helping it understand the context and generate relevant and coherent language-based output.

# In[11]:


from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}" # provides context

prompt_template = ChatPromptTemplate.from_messages(
    [("system",system_template),("user","{text}")] # user,system -> different role
)


# In[13]:


prompt = prompt_template.invoke({"language": "Italian","text":"Bruh"})
prompt


# In[ ]:


prompt.to_messages() # If we want to access the messages directly


# In[15]:


response = model.invoke(prompt)
print(response.content)

