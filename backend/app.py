import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter 
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

#? Workflow - 
#? Data -> Embedding Model -> Finds similar Embeddings -> Returns relevant Chunks of Text 
#? Query + Chunks of Text -> LLM -> Answer

#! Load .env no matter where you run from
load_dotenv(find_dotenv(), override=True)

#! Retrieve Open AI Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY in environment variables.")

#! load documents
documents = SimpleDirectoryReader(input_files=["backend/data/Python_Wikipedia.txt"]).load_data()

#! Sentence Splitter into nodes
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=150)
nodes = splitter.get_nodes_from_documents(documents)

#! Define AI model and embedding model
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

#! Define Summary Index/Vector Index
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

#! Turn indexes into query engines
summary_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_engine = vector_index.as_query_engine(similarity_top_k=5)

#! Turn query engines into tools (same as query engines, but with meta data)
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_engine,
    description=(
        "Use for high-level summaries, themes, or overviews"
    ),
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_engine,
    description=(
        "Use for specific facts, definitions, quotes, or pinpoint answers"
    ),
)

#! Define Router Query Engine
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=True
)

#! Test some questions
response = query_engine.query("What is small summary of the document?")
print(str(response))
response = query_engine.query(
    "When was python invented and by who?"
)
print(str(response))