from langchain.agents import create_agent
# from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search , scrape_url 
from dotenv import load_dotenv
import os

load_dotenv()

#model setup 
# llm = ChatOpenAI(model = "gpt-4o-mini",temperature=0)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0
# )
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


# #1st agent 
# def build_search_agent():
#     return create_agent(
#         model = llm,
#         tools= [web_search]
#     )

def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="""
You are ONLY a web search agent.

Your ONLY available tool is `web_search`.

You MUST use exactly this tool for web research.

Tool schema:
web_search(query: str)

Rules:
1. Call ONLY `web_search`.
2. Never call `web_open`.
3. Never call `browser_search`.
4. Never call any tool other than `web_search`.
5. Never provide cursor, id, URL, or other parameters to web_search.
6. Every web_search call must contain exactly one argument:
   {"query": "<natural language search query>"}
7. Perform at most 3 searches.
8. After receiving the search results, STOP and return them.
9. Do not attempt to open, visit, scrape, browse, or inspect URLs.
"""
    )
#2nd agent 

def build_reader_agent():
    return create_agent(
        model = llm,
        tools = [scrape_url]
    )


#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()

