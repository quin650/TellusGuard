from utils import get_router_query_engine, getLLMObject
from functions  import add_tool, mystery_tool

def main():

    #! Query Engine
    # query_engine = get_router_query_engine("data/Python_Wikipedia.txt", verbose=True)
    # response = query_engine.query("What is the summary of the document?")
    # print(str(response))
    # response = query_engine.query("Is Python a high-level language?")
    # print(str(response))

    #! Tool Calling
    llm = getLLMObject()
    response = llm.predict_and_call(
            [add_tool, mystery_tool], 
            "Tell me the output of the mystery function on 2 and 9", 
            verbose=True
    )
    print(str(response))

if __name__ == "__main__":
    main()