import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain.chains.llm_math.base import LLMMathChain
from langchain.agents import Tool,initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
load_dotenv()


### Set up the streamlit app
st.set_page_config(page_title="Text to math problem solver and data search assistant")
st.title("Text to math problem solver and data search assistant uasing LLm")

groq_api_key=st.sidebar.text_input(label="Groq api Key",type="password")

if not groq_api_key:
    st.info("Please add Groq api Key to continue")
    st.stop()
llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

## Initialize the tool
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for search the internate and solve the Problem "
)

## Intialize the Math Tool
math_chain=LLMMathChain.from_llm(llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions,Only input mathematical expression"


)
prompt="""   Your a agent taked for solution users mathematical questions,logically arrive add the solution detailed explaination and explain poin wise for the question below
Question:{question}
Answer"""
prompt_template=PromptTemplate(input_variables=['question'],
                               template=prompt
                               )

## Mathe problem tool combine all the tool

chain=LLMMathChain(llm=llm,prompt=prompt_template)

resoning_tool=Tool(
    name="Resoning",
    func=chain.run,
    description="A tool for answer logica-based and resoning questions."
)
## Initialize the agent
assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,resoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handal_parsing_error=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi,I'M a Math Chatbot who can answore all your mathe questions. "}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


## function to genrate the response

def genrate_response(question):
    response=assistant_agent.invoke({'input':question})
    return response

## Start the intraction
question=st.text_area("Enter your Questions:","I have 5 bananas and 3 grapes. I bought 2 more bananas and 4 more grapes. How many do I have now?")

if st.button("Find my answer"):
    if question:
        with st.spinner("Genrate Response"):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages ,callbacks=[st_cb])
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write("### Response")
            st.success(response)

    else:
        st.warning("please enter the Questions")
