from config.config import OPENAI_API_KEY, DEBUG
# print('test')

#! Lesson 1: Router Engine

#! Define AI model and embedding model
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
# Settings.llm = OpenAI(model="gpt-3.5-turbo")
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

#! Load Data from file
from llama_index.core import SimpleDirectoryReader 
documents = SimpleDirectoryReader(input_files=["data/Python_Wikipedia.txt"]).load_data()

#! Sentence Splitter into nodes
from llama_index.core.node_parser import SentenceSplitter 
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

#! Define Vector Index/Summary Index
from llama_index.core import SummaryIndex, VectorStoreIndex
summary_index = SummaryIndex(nodes)         # returns *all* nodes (via summaries, not embeddings)
vector_index = VectorStoreIndex(nodes)      # returns only nodes similar in embedding space

#! Turn indexes into query engines
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()

#! Turn query engines into tools (same as query engines, but with meta data)
from llama_index.core.tools import QueryEngineTool
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for retrieving general details from the Python article."
    ),
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Use me to retrieve specific facts, definitions, dates, or code details from the Python article."
    ),
)

#! Define Router Query Engine
# Two selector types available - 
# LLM selectors -> use LLM --> parsed JSON --> indexes queried
# Pydantic selectors --> OpenAI function calling API -> pydantic selection objects
# verbose=False --> removes the selection logs
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector #have the capability of selecting one index or multiple
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=False
)

#! Test some questions
response = query_engine.query("What is the summary of the document?")
print(str(response))
# Tells us how many chunks were called, from this number, can tell which tool was used. i.e. summary if num equals all chunks
# print(len(response.source_nodes))

response = query_engine.query("Is python a high-level language?")
print(str(response))


# ! The afore-mentioned, can be created using one simple helper function:
# from utils import get_router_query_engine
# query_engine = get_router_query_engine("data/Python_Wikipedia.txt")

# response = query_engine.query("Tell me about the python language?")
# print(str(response))