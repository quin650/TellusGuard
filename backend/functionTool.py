import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

#! Load .env no matter where you run from
load_dotenv(find_dotenv(), override=True)

#! Retrieve Open AI Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY in environment variables.")

#! Define AI model
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

#! FunctionTool uses Type annotations: x, y + Doc String """...""" as part of prompt
def add(x: int, y: int) -> int:
    """Adds two integers together."""
    return x + y
def mystery(x: int, y: int) -> int: 
    """Mystery function that operates on top of two numbers."""
    return (x + y) * (x + y)
add_tool = FunctionTool.from_defaults(fn=add)
mystery_tool = FunctionTool.from_defaults(fn=mystery)

#! predict_and_call: tools + prompt -> decides which tool to call -> response
response = Settings.llm.predict_and_call(
    [add_tool, mystery_tool], 
    "Tell me the output of the mystery function on 2 and 9", 
    verbose=True
)
print(str(response))