from utils import get_router_query_engine

def main():
    query_engine = get_router_query_engine("data/Python_Wikipedia.txt", verbose=True)

    response = query_engine.query("What is the summary of the document?")
    print(str(response))

    response = query_engine.query("Is Python a high-level language?")
    print(str(response))

if __name__ == "__main__":
    main()