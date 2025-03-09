import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
import time

load_dotenv()

# Define the state type for our travel planner
class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "The messages in the conversation"]
    city: str
    interests: List[str]
    itinerary: str

# Initialize the language model (ensure you have the proper credentials/config)
llm = ChatGroq(
    model="deepseek-r1-distill-qwen-32b"
)

# Create the itinerary prompt
itinerary_prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user's interests: {interests}. Provide a brief, bulleted itinerary.Also Suggest some places based on {city} and {interests}"
    ),
    ("human", "Create an itinerary for my day trip."),
])

# This node creates the itinerary using the provided city and interests.
def create_itinerary(state: PlannerState) -> PlannerState:
    st.write(f"Creating an itinerary for **{state['city']}** based on interests: **{', '.join(state['interests'])}**...")
    response = llm.invoke(
        itinerary_prompt.format_messages(
            city=state['city'],
            interests=", ".join(state['interests'])
        )
    )
    st.write("\n**Final Itinerary:**")
    # st.write(response.content)
    return {
        **state,
        "messages": state['messages'] + [AIMessage(content=response.content)],
        "itinerary": response.content,
    }

# A helper function that builds the initial state using the inputs and then calls create_itinerary.
def generate_itinerary(city: str, interests: str) -> str:
    state: PlannerState = {
        "messages": [HumanMessage(content="I want to plan a day trip.")],
        "city": city,
        "interests": [i.strip() for i in interests.split(',')],
        "itinerary": "",
    }
    state = create_itinerary(state)
    return state["itinerary"]

# -------------------------
# Streamlit Front End Code
# -------------------------

# Set page configuration
st.set_page_config(page_title="Travel Planner", page_icon="✈️", layout="wide")

st.title("Travel Planner Agent 🚀")
st.subheader("Plan your perfect day trip!")

# Text inputs for the required parameters
city = st.text_input("Enter the city you want to visit:")
interests = st.text_input("Enter your interests (comma-separated):")

if st.button("Generate Itinerary"):
    if city and interests:
        with st.spinner("Creating your itinerary..."):
            itinerary = generate_itinerary(city, interests)
            time.sleep(1)  # Simulate processing delay if needed
        st.success("Itinerary generated!")
        st.markdown("### Your Day Trip Itinerary:")
        st.markdown(itinerary)
    else:
        st.error("Please enter both a city and your interests.")
