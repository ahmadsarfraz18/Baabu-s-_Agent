from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig, AsyncOpenAI, set_tracing_disabled
from dotenv import load_dotenv
import os

load_dotenv()
set_tracing_disabled(disabled=True)

gemini_api_key= os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in environment variables. ")

external_client= AsyncOpenAI(
    api_key= gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model= OpenAIChatCompletionsModel(
    model= "gemini-2.5-flash",
    openai_client= external_client
)

english_teacher= Agent(
    name= "English Teacher",
    instructions= "You are a helpful english teacher and your task is to resolve the queries regarding the English. ",
    model= model,
)

math_teacher= Agent(
    name= "Math Teacher",
    instructions= "You are a helpful math teacher and your task is to resolve the queries regarding the Math. ",
    model= model,
)

urdu_teacher= Agent(
    name= "urdu Teacher",
    instructions= "You are a helpful urdu teacher and your task is to resolve the queries regarding the Urdu .",
    model= model,
)

general_knowledge_teacher= Agent(
    name= "General Knowledge Teacher",
    instructions= "You are a helpful general knowledge teacher and your task is to resolve the queries regarding the Gk .",
    model= model,
)

islamiyat_teacher= Agent(
    name= "Islamiyat Teacher",
    instructions= "You are a helpful islamiyat teacher and your task is to resolve the queries regarding the Islamiyat .",
    model= model,
)

triage_agent= Agent(
    name= "Manager",
    instructions= "You are a Manager agent and your task is to transfer the task to the relavant agent, ",
    model= model,
    handoffs= [english_teacher, math_teacher, urdu_teacher, general_knowledge_teacher, islamiyat_teacher] 
)

config= RunConfig(
    model= model,
    model_provider= external_client,
)

result= Runner.run_sync(
    triage_agent,
    "hamaray pyaray nabi Hazrat Muhammad (SAW) k uswa e hasna pay aik easy easy likh k do roman urdu mn. ",
    run_config= config,
) 
print("last agent:", result.last_agent)  
print(result.final_output)


