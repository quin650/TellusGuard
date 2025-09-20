from config.config import OPENAI_API_KEY, DEBUG
# print('test')

#! Lesson 1: Router Engine
#! Load Data from file
from llama_index.core import SimpleDirectoryReader 
documents = SimpleDirectoryReader(input_files=["data/Python_Wikipedia.txt"]).load_data()

#! Sentence Splitter into nodes
from llama_index.core.node_parser import SentenceSplitter 
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

#! Define AI model and embedding model
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
# Settings.llm = OpenAI(model="gpt-3.5-turbo")
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
Settings.llm = OpenAI(model="gpt-5-nano")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# vector index queries based on similarities in embeddings
# summary index queries through a hierarchy of summaries
#! Define Indexes
from llama_index.core import SummaryIndex, VectorStoreIndex
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

#! Turn indexes into query engines
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()
#! Turn query engines into tools (same as query engines, but with descriptions)
from llama_index.core.tools import QueryEngineTool
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for summarization questions related to MetaGPT"
    ),
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from the MetaGPT paper."
    ),
)

