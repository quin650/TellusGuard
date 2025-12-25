import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter 
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters

#! Load .env no matter where you run from
load_dotenv(find_dotenv(), override=True)

#! Retrieve Open AI Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY in environment variables.")

#! Load documents
documents = SimpleDirectoryReader(input_files=["backend/data/Python_Wikipedia.txt"]).load_data()

#! Sentence Splitter into nodes
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=150)
nodes = splitter.get_nodes_from_documents(documents)

#! prints content + metadata
# print(nodes[0].get_content(metadata_mode="all"))

#! Define Vector Index
vector_index = VectorStoreIndex(nodes)

#! Turn indexes into query engines with meta filters
#! i.e. the query will base response on page 2
vector_engine = vector_index.as_query_engine(
    similarity_top_k=2,
    filters=MetadataFilters.from_dicts(
        [
            {"key": "file_name", "value": "Python_Wikipedia.txt"}
        ]
    )
)

#! metadata 
# print(nodes[0].metadata)

#! 
response = vector_engine.query(
    "What year did python gain popularity?", 
)

#! 
print("1: ", str(response))

#! print the metadata for each chunk
for n in response.source_nodes:
    print("2", n.metadata)