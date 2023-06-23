import os
import streamlit as st
import openai
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI

# Set OpenAI API key


def app():
    # Title and description
    st.title("Ask Your CSV Files")
    st.write(
        "Enter your OPEN AI API KEY, Upload a CSV file and enter a query to get an answer.")
    api_key = st.text_input("Enter OpenAI API key:")

    file = st.file_uploader("Upload CSV file", type=["csv"])
    if not file:
        st.stop()

    data = pd.read_csv(file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    agent = create_pandas_dataframe_agent(
        OpenAI(temperature=0, openai_api_key=api_key), data, verbose=True)

    query = st.text_input("Enter a query:")

    if st.button("Execute"):
        answer = agent.run(query)
        st.write("Answer:")
        st.write(answer)


if __name__ == "__main__":
    app()
