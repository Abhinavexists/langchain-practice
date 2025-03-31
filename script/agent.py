#!/usr/bin/env python
# coding: utf-8

# ### Install the required dependencies
# i am using Mistral AI and Sqlite

# In[ ]:


# %pip install -U langchain-community langgraph langchain-mistralai tavily-python langgraph-checkpoint-sqlite


# Setup [Langsmith](https://www.langchain.com/langsmith) and [Tavily API Key](https://tavily.com/)

# In[ ]:


import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter API key for LangSmith: ")
os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter API key for Tavily: ")


# Tavily search engine as tool.

# In[ ]:


from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=2)
search_results = search.invoke("what is weather in Delhi")
# print(search_results) - give results in JSON
tools = [search] 
# Assigning search to tools allows you to pass this list to an 
# LLM-powered agent so it can decide which tool to use.

json_to_string_results = "\n".join([
    f"{idx+1}.{result['title']}-{result['content']}" 
    for idx,result in enumerate(search_results)
    ])
print(json_to_string_results)


# ### Using LLM

# Install the language model (can change based on your preference)

# In[ ]:


# pip install -qU "langchain[mistralai]"


# In[ ]:


from langchain.chat_models import init_chat_model
# checks if key is avaliable  and if not, ask user to enter one
if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

model = init_chat_model("mistral-large-latest", model_provider="mistralai")
# 


# You can call the language model by passing in a list of messages. By default, the response is a content string.

# In[10]:


from langchain_core.messages import HumanMessage

response = model.invoke([HumanMessage(content="hi!")])
response.content


# In[11]:


# to enable this model to do tool calling we use .bind_tools
# to give the language model knowledge of these tools
model_with_tools = model.bind_tools(tools)


# In[12]:


response = model_with_tools.invoke([HumanMessage(content="Hi!")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")
# there was no tool called for this input to process so it returned empty


# In[13]:


# Now, let's try calling it with some input that would expect a tool to be called.
response = model_with_tools.invoke([HumanMessage(content="What's the weather in Delhi?")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")

# Many LLMs (including Mistral and OpenAI's models) 
# will decide whether to call a tool only if they think it's necessary.


# Using ReAct(Reasoning + Acting) agent

# In[14]:


from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(model, tools)


# In[15]:


response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})
response["messages"]


# In[16]:


response = agent_executor.invoke({"messages": [HumanMessage(content="What is the weather in Delhi?")]})
response["messages"]


# invkoing tool calls

# In[17]:


# If the agent executes multiple steps, this may take a while. 
# To show intermediate progress, we can stream back messages as they occur.

for step in agent_executor.stream(
    {"messages": [HumanMessage(content="What is the weather in Delhi?")]},
    stream_mode="values", 
):
    step["messages"][-1].pretty_print()


# Streaming Messages

# In[18]:


for step, metadata in agent_executor.stream(
    {"messages": [HumanMessage(content="What is the weather in Delhi?")]},
    stream_mode="messages", 
):
    if metadata["langgraph_node"] == "agent" and (text := step.text()):
        print(text, end="|")


# Adding in Memory

# In[19]:


# this agent is stateless.
# This means it does not remember previous interactions.
# To give it memory we need to pass in a checkpointer.

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()


# Setting checkpoint

# In[25]:


agent_executor = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}


# In[27]:


for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob!")]}, config
    ):
        print(chunk)
        print("----")


# If you want to start a new conversation, all you have to do is change the ```thread_id``` used

# In[28]:


config = {"configurable": {"thread_id": "xyz123"}} # New Thread ID
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="What is my name?")]}, config
    ):
       print(chunk)
       print("----")

