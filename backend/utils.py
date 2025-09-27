from typing import Optional, List
from config.config import OPENAI_API_KEY, DEBUG
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader 
from llama_index.core.node_parser import SentenceSplitter 
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector 
from llama_index.core.vector_stores import MetadataFilters, FilterCondition

def get_router_query_engine(
    filepath: str,
    llm_model: str = "gpt-3.5-turbo",
    embed_model: str = "text-embedding-3-small",
    api_key: str = OPENAI_API_KEY,
    chunk_size: int = 1024,
    verbose: bool = False,
) -> RouterQueryEngine:

    Settings.llm = OpenAI(model=llm_model, api_key=api_key)
    Settings.embed_model = OpenAIEmbedding(model=embed_model, api_key=api_key)
    documents = SimpleDirectoryReader(input_files=[filepath]).load_data()
    splitter = SentenceSplitter(chunk_size=chunk_size, separator=" ")
    nodes = splitter.get_nodes_from_documents(documents)
    summary_index = SummaryIndex(nodes)    
    vector_index = VectorStoreIndex(nodes)   

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name="summary_tool",
        query_engine=summary_query_engine,
        description=(
            "Useful for retrieving specific details from the Python article."
        ),
    )
    
    vector_query_engine = vector_index.as_query_engine()
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Use me to retrieve specific facts, definitions, dates, or code details from the Python article."
        ),
    )

    def vector_query(
        query: str, 
        page_numbers: List[str]
    ) -> str:
        """Perform a vector search over an index.
        query (str): the string query to be embedded.
        page_numbers (List[str]): Filter by set of pages. Leave BLANK if we want to perform a vector search
            over all pages. Otherwise, filter by the set of specified pages.
        """

        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )
        )
        response = query_engine.query(query)
        return response
    vector_query_tool = FunctionTool.from_defaults(
        name="vector_tool",
        fn=vector_query
    )

    router = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[summary_tool, vector_tool, vector_query_tool],
            verbose=verbose,
        )
    return router

def getLLMObject(
    llm_model: str = "gpt-3.5-turbo",
    api_key: str = OPENAI_API_KEY,
) -> OpenAI:
    Settings.llm = OpenAI(model=llm_model, api_key=api_key)
    return Settings.llm