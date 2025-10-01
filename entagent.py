import os 
from langchain_voyageai import VoyageAIEmbeddings
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_cohere import CohereRerank
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langchain_community.vectorstores import FAISS
from langchain_core.tools import StructuredTool
from typing import Sequence, TypedDict, Annotated, Literal
from llm_MOA import mixture_of_agents
from dotenv import load_dotenv
load_dotenv()

#Load the API key
google_api_key = os.getenv('GOOGLE_API_KEY')
google_cse_id = os.getenv('GOOGLE_CSE_ID')
os.environ["LANGCHAIN_PROJECT"] = 'llm_reflexion_MCQ'
langchain_api = os.getenv("LANGCHAIN_API_KEY")
api_key = os.getenv("LANGCHAIN_COHERE_KEY")
voyageai_api_key = os.getenv("VOYAGEAI_API_KEY")
co_api_key = os.getenv("CO_API_KEY")

general = 'llama3.1:8b'
tool_use = 'llama3.1:8b'

#Reflection Critiques
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")
#Function to choose different models based on the needs
def model_select(model):
    llm = ChatOllama(model=model,format='json')
    return llm

#Building RAG node 
embedding = VoyageAIEmbeddings(
    voyage_api_key=voyageai_api_key, model="voyage-large-2-instruct"
)
vectorstore = FAISS.load_local(
    "FAISS_index", embedding, allow_dangerous_deserialization=True
)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval(keywords: list[str]) -> dict:
    """Run the search_query by searching similar documents related to the embedded vector of the search_query.""" 
    retrieved_docs = [format_docs(vectorstore.similarity_search_by_vector(
        embedding.embed_query(query), k=1)) for query in keywords
        ]
    return retrieved_docs

rag = StructuredTool.from_function(retrieval, handle_tool_error=True)
rag_node = ToolNode([rag])

#The format structure of the response from Drafter and Revisor
class Keywords(BaseModel):
    """Find main key medical concepts in the question"""
    identified_concepts: list[str] = Field(description="A list of main key factors and medical concepts found in the question.") 

question_identifier_prompt = PromptTemplate(template=""" You are given a question. Perform the task and follow the output format exactly.
                                Input
                                Question: {question}

                                Task — Key factors and concepts

                                Identify main key factors and medical concepts necessary to solve the question (otolaryngology-focused).
                                Output format (strict): a plain bullet list only (no explanations, no numbering).
                                
                                

                                """
)

#Drafter's chain
identifier_chain = question_identifier_prompt|model_select(general).with_structured_output(Keywords)
def RAG_agent(state):
    keywords = identifier_chain.invoke({'question':state['question']})
    rag_docs = retrieval(keywords.identified_concepts)
    return {'question':state['question'], 'retrieved_docs':rag_docs}

#Agents used
#Supervisor's prompt
examiners_prompt = PromptTemplate(template="""
                                You are an expert reviewer. You will receive:
                                - Question: the user’s query
                                - Answer: an draft answer to the question

                                Your tasks:
                                Review for Quality
                                - Evaluate and grade the answer very strictly for the following criteria:
                                    a) Relevance: directly answer the Question; avoids unnecessary tangents, contradictions or hallucinations
                                    c) Detail: includes essential specifics to resolve the question.
                                    d) Clarity: concise, well-structured, and unambiguous.
                                  Each criterion has 0-100 marks, higher marks mean higher quality, you should grade the answer for each criterion

                                3) Output Decision and Format
                                    a) Give a score: {{score:'Relevance':your score, 'Detail':your score, 'Clarity': your score}}
                                    b) For each criterion, if its score is below than 80 marks, reflect on the answer in that criterion and list precise deficiencies
                                    c) Queries: propose targeted search queries to address your reflection.
                                    d) Respond using the {imperfect_answer} function. 
                                  
                                    Arguments:
                                        - reflection: your critiques
                                        - search_query: the list of search queries
                                        - answer: the draft answer

                                Question:
                                {question}

                                Answer:
                                {answer}
""")

class BestAnswer(BaseModel):
    """Using this output format when the answer is the best, provide a detailed and accurate answer to the question"""
    answer: str = Field(description="A detailed answer to the question.")
    score: dict[str,int] = Field(description="A list of scores of each criterion")
class ImperfectAnswer(BaseModel):
    """Answer the question. Provide an detailed answer, reflection, and then follow up with search query related to the critique of missing to improve the answer."""
    answer: str = Field(description="A detailed answer to the question.")
    score: dict[str,int] = Field(description="A dictionary of scores of each criterion")
    reflection: list[str] = Field(description="Your reflection on your generated answer.")
    search_query: list[str] = Field(description="A list of search queries.")

#Supervisor. 
def examiners_agent(state):
    examiners_chain = (
        examiners_prompt.partial(
        imperfect_answer=ImperfectAnswer.__name__
        )
        |model_select(general).bind_tools([ImperfectAnswer])
    )
    input = {"question": state['question'],
             "answer": state['answer']
            }
    response = examiners_chain.invoke(input)
    scores = response.tool_calls[0]['args']['score']
    quality = 'High' if all(score>=80 for score in scores.values()) else 'Low'
    output = {
        'answer': response.tool_calls[0]['args']['answer'],     
        'quality': quality,
        'score': response.tool_calls[0]['args']['score'],
        'search_query': response.tool_calls[0]['args'].get('search_query', ''),
        'reflection': response.tool_calls[0]['args'].get('reflection', '')
    }


    return output

#Multiple agents with tools from Google, ArXiv, PubMed
google_search = GoogleSearchAPIWrapper(
      google_api_key = google_api_key,
      google_cse_id = google_cse_id,
      k=2
)
pubmed = PubmedQueryRun(top_k_results=2, verbose=True)
arxiv = ArxivAPIWrapper(top_k_results=2)

from extract import flatten_summaries
def research_agent(state, **kwargs):
    infos = []
    search_query = state['search_query']
    del state['search_query']
    """Run the search tools with search query as the input.""" 
    for query in search_query:
        info_pubmed = pubmed.invoke(query)
        #info_google = google_search.run(query)
        info_arxiv = arxiv.run(query)
        info_dict = {"pubmed": info_pubmed,
                     #"google": info_google,
                     "arxiv": info_arxiv}
        info_dict = flatten_summaries(info_dict)
        best_info = CohereRerank(model='rerank-v3.5').rerank(documents=info_dict,query=query,top_n=1)
        infos.append(info_dict[best_info[0]['index']])
    
    state['retrieved_docs'] = infos
    return state

#Building the Agent Graph, all the LLMs and agents used will be connected
from langgraph.graph import END, START,  StateGraph

class AgentState(TypedDict):
    question: str
    quality: str
    score: list[str]
    answer:str
    retrieved_docs: list[str]
    reflection: list[str]
    search_query: list[str]

builder = StateGraph(AgentState)
builder.add_node("RAG", RAG_agent)
builder.add_node('mixture_of_agents',mixture_of_agents)
builder.add_node("examiners", examiners_agent)
builder.add_node('research',research_agent)

builder.add_edge(START, "RAG")
builder.add_edge("RAG", "mixture_of_agents")
builder.add_edge("mixture_of_agents", "examiners")
builder.add_edge("research", "mixture_of_agents")

conditional_map = {'High':END, 'Low': 'research'}
builder.add_conditional_edges("examiners", lambda x:  x["quality"], conditional_map)
graph = builder.compile()

#Question-Answering
#while True:
        #user_input = input("You: ")
        #if user_input.lower() in ["exit", "quit"]:
        #    print("Exiting the chatbot. Goodbye!")
        #    break
        
for state in graph.stream({'question':"What nerve is injured during thyroidectomy that reduces supraglottic sensation?"},{"recursion_limit": 100},
): 
    if "__end__" not in state:
        print(state)
        print("----")




