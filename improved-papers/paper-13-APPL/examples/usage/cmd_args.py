from loguru import logger

import appl
from appl import gen, ppl

# * option 1: update part of the configs in a dict
# appl.init(default_servers={"default": "gpt-4o", "small": "gpt-4o-mini"})
# !! update the default server to `gpt-4o`, and the small server to `gpt-4o-mini`
# !! used for gen(...) and gen(server="small", ...)

# * option 2: get options from command line or environment variables
parser = appl.get_parser()
parser.add_argument("--name", type=str, default="APPL")
args = parser.parse_args()
appl.update_appl_configs(args.appl)

# * Both change the default server to `gpt-4o`
# python cmd_args.py --appl.default_servers.default gpt-4o
# _APPL__DEFAULT_SERVERS__DEFAULT=gpt-4o python cmd_args.py


@ppl  # the @ppl decorator marks the function as an `APPL function`
def greeting(name: str):
    f"Hello World! My name is {name}."  # Add text to the prompt
    return gen()  # call the default LLM with the current prompt


print(greeting(args.name))  # call `hello_world` as a normal Python function
