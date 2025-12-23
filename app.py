import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import validators
#######
# Streamlit App
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Get the Groq API Key and URL
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Define the LLM (only if key is present)
if groq_api_key:
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

# Define Prompt
prompt_template = """
Provide a summary of the following content in 300 words:
Content:{documents}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["documents"])

if st.button("Summarize the Content from YT or Website"):
    
    # Validate inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube or website).")
    else:
        try:
            with st.spinner("Fetching and summarizing content..."):

                # Load data from Youtube / website
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url, 
                        add_video_info=False,
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent":
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        },
                    )

                docs = loader.load()
                
                # 1. Combine doc content into a single string (Correct fix for Groq)
                formatted_docs = "\n\n".join([doc.page_content for doc in docs])
                
                # 2. Direct Chain (LCEL) - simpler and works with string input
                chain = prompt | llm
                
                # 3. Invoke the chain
                output_summary = chain.invoke({"documents": formatted_docs})

                # 4. Display result (.content extracts the text from the AIMessage)
                st.success(output_summary.content)

        except Exception as e:

            st.exception(f"Exception: {e}")
