import os
from dotenv import load_dotenv 
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.document_loaders import UnstructuredURLLoader
from llama_index import download_loader
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import requests
import ast
import nltk
import json


load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


# query = 'how did jeff bezos get rich?'
def search(query,SERPAPI_API_KEY):
    api_endpoint = 'https://serpapi.com/search'
    params = {
        'q': query,
        'tbm': 'nws',
        'api_key': SERPAPI_API_KEY,
        'Content-Type': 'application/json'

    }

    response = requests.get(api_endpoint, params=params)
    data = response.json()

    if 'news_results' in data:
        news_list = data['news_results']
        #print("search_results: " ,news_list)

        headline=[]
        url=[]
        for news in news_list:
          headline.append(news['title'])
          headline.append(news['link'])
          url.append(news['link'])
        # print(f"Headline: {headline}")
        # print(f"URL: {url}")
        # print()
        return headline
    else:
        
        return []
    



def find_best_article_urls(data , query):
    response_str = data
    print(response_str)


    llm = OpenAI(model_name ='gpt-3.5-turbo', temperature= 0)

    template = """
    You are a world class journalist & researcher, you are extremely good at find most relevant articles to certain topic;
    {response_str}
    Above is the list headline and urls of search results for the query {query}.
    Please choose the best 3 articles from the list, return ONLY a list of the urls, do not include anything else; 
    """
    prompt_template = PromptTemplate(
        input_variables=["response_str", "query"], template=template)
 
    article_picker_chain = LLMChain(
        llm=llm, prompt=prompt_template, verbose=True)
    
    urls = article_picker_chain.predict(response_str=response_str, query=query)
    #url_list = json.loads(urls)
    # print("10000000##")
    print(urls)
    print(type(urls))
    b = ast.literal_eval(urls)
    # urls = (urls)
    # print(urls)
    # print(urls[5])
    # a="".join(urls)
    # print (a)
    print(type(b))


    return b

def get_content_from_urls(urls):   
    # use unstructuredURLLoader
    # nltk.download('punkt')
    # loader = UnstructuredURLLoader(urls=urls)
    # data = loader.load()
    UnstructuredURLLoader = download_loader("UnstructuredURLLoader")
    loader = UnstructuredURLLoader(urls=urls, continue_on_failure=True, headers={"User-Agent": "value"})
    data=loader.load()
    datas=data[0].text
    print(datas)
    return datas
  


def summarise(data, query):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=3000, chunk_overlap=200, length_function=len)
    text = text_splitter.split_text(data)    

    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=.7)
    template = """
    {text}
    You are a world class journalist, and you will try to summarise the text above in order to create a twitter thread about {query}
    Please follow all of the following rules:
    1/ Make sure the content is engaging, informative with good data
    2/ Make sure the content is not too long, it should be no more than 3-5 tweets
    3/ The content should address the {query} topic very well
    4/ The content needs to be viral, and get at least 1000 likes
    5/ The content needs to be written in a way that is easy to read and understand
    6/ The content needs to give audience actionable advice & insights too

    SUMMARY:
    """

    prompt_template = PromptTemplate(input_variables=["text", "query"], template=template)

    summariser_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    summaries = []

    for chunk in enumerate(text):
        summary = summariser_chain.predict(text=chunk, query=query)
        summaries.append(summary)

    # print(summaries)
    return summaries

def generate_thread(summaries, query):
    summaries_str = str(summaries)

    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=.7)
    template = """
    {summaries_str}

    You are a world class journalist & twitter influencer, text above is some context about {query}
    Please write a viral twitter thread about {query} using the text above, and following all rules below:
    1/ The thread needs to be engaging, informative with good data
    2/ The thread needs to be around than 3-5 tweets
    3/ The thread needs to address the {query} topic very well
    4/ The thread needs to be viral, and get at least 1000 likes
    5/ The thread needs to be written in a way that is easy to read and understand
    6/ The thread needs to give audience actionable advice & insights too

    TWITTER THREAD:
    """

    prompt_template = PromptTemplate(input_variables=["summaries_str", "query"], template=template)
    twitter_thread_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    twitter_thread = twitter_thread_chain.predict(summaries_str=summaries_str, query=query)
    return twitter_thread

def main():
    load_dotenv()   


    st.set_page_config(page_title="Autonomous researcher - Twitter threads", page_icon=":bird:")

    st.header("Autonomous researcher - Twitter threads :bird:")
    openaiapi = st.text_input("OpenAI API Key")
    query = st.text_input("Topic of twitter thread")

    OPENAI_API_KEY = openaiapi
    if query:
        print(query)
        st.write("Generating twitter thread for: ", query)
        
        search_results = search(query,SERPAPI_API_KEY)
        urls = find_best_article_urls(search_results, query)
        data = get_content_from_urls(urls)
        summaries = summarise(data, query)
        thread = generate_thread(summaries, query)


        with st.expander("search results"):
            st.info(search_results)
        with st.expander("best urls"):
            st.info(urls)
        with st.expander("data"):
            st.info(data)
        with st.expander("summaries"):
            st.info(summaries)
        with st.expander("thread"):
            st.info(thread)

if __name__ == '__main__':
    main()
