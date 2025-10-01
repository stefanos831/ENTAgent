from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()
#'deepseek-r1:70b','qwen3:32b','mistral-large:123b'

class BestResponse(BaseModel):
    """Aggregate the best response from numerous outputs by the model"""
    answer: str = Field(description="The aggregated response")

def mixture_of_agents(state):
    responses = []
    models = ['llama3.1:8b']
    answer = state.get('answer', '')
    for _ in range (0,3):

        answer_generation_prompt = PromptTemplate(template="""You are an expert in answering question. You have received:
                                                            - Retrieved Information:{information} (information relevant to the question)
                                                            - Question:{question} the user’s query
                                                            - Answer:{answer} an optional draft answer (may be empty)
                                                            If the answer is empty, you should produce a very detailed answer to solve the question using the retrieved information
                                                            If answer is provided, you MUST refine the answer using the retrieved information
                                                            
                                    """)
        llm = ChatOllama(model=models[0], format='json')
        answer_generation_chain = answer_generation_prompt|llm|JsonOutputParser()
        output = answer_generation_chain.invoke({'question':state['question'],'information':state['retrieved_docs'],'answer':answer})
    
        responses.append(output)
    
    aggregator_prompt=PromptTemplate(template="""You have recevied a question and multiple responses from a large lanaguage model.
                                                 You are an expert synthesis engine. Your task is to read multiple responses, compare them, resolve conflicts, and integrate them to produce the best possible final answer. 
                                                 Prioritize correctness, explicit reasoning, verifiable claims, and clarity.
                                                 Produce your response using 
                                                 Question:{question}
                                                 Responses:{responses}
                                              """
    )
    aggregator = ChatOllama(model='llama3.1:8b', format='json', temperature=0)

    aggregator_chain = aggregator_prompt|aggregator.with_structured_output(BestResponse)
    best_responses = aggregator_chain.invoke({'question':state['question'],'responses':responses})
    return {'question':state['question'],'answer':best_responses.answer}

# llm = ChatOllama(model='gemma3:4b', format="json", temperature=0)
# chain = llm | JsonOutputParser()
# output = chain.invoke('what is otolaryngology?')
# print(output)
