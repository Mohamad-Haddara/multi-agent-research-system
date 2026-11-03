
from argparse import Namespace
from langchain.chat_models import init_chat_model
from typing import Any, List, Dict

"""
ARBITRARY Arguments: means we don't know how many arguments the user will pass in when they invoke the function.
To accept many amount of arguments - We use *args, **kwargs

*args = allows us to pass multiple non-key arguments 
**kwargs = allows us pass multiple keyword-arguments 
I want to prefix each of these parameter with (*) unpacking operator
*arg -> pack the arguments into tuple 
**kwags -> pack the arguments into dictionary
"""

# most important thing is unpacking operator
def add_function(*args):
    # using unpacking operator - we pack all the parameters in functions into tuple
    print(type(args))
    total = 0
    for arg in args:
        total += arg

    return arg

add_function(1,2,3)


def display_name(*args):
    for arg in args:
        print(arg, end=" ")

display_name("moe", "haddara")


def print_address(**kwargs):
    # unpack the dictionary
    print(kwargs.get('street'))
    


print_address(street = "123 Fake St." , city = "Detroit" , state= "MI", zip="54321") #When I pass in these keyword argument -> packed into dictionary 


from langchain.chat_models import init_chat_model

def is_openai_model(model: str) -> bool:

    return model.startswith("openai:") or model.startswith("gpt-")

def _build_llm(model: str, **kwargs: Any): #collect keyword arguments (name = value)


    return init_chat_model(model, **kwargs)
        
LLM_CONFIGS: List[Dict[str, Any]] = [

    {
            "name": "gpt-5.6-luna",
            "model": "gpt-5.6-luna",
            "kwargs": {
                "max_completion_tokens": 2000,
                "reasoning": {"effort": "medium"},
            },
        },
        {
            "name": "gpt-5.4",
            "model": "gpt-5.4",
            "kwargs": {
                "max_completion_tokens": 2000,
                "reasoning": {"effort": "medium"},
            },
        },
        {
            "name": "gpt-5.4-mini",
            "model": "gpt-5.4-mini",
            "kwargs": {
                "max_completion_tokens": 2000,
                "reasoning": {"effort": "low"},
            },
        },
        {
            "name": "gpt-5.4-nano",
            "model": "gpt-5.4-nano",
            "kwargs": {
                "max_completion_tokens": 2000,
                "reasoning": {"effort": "low"},
            },
        },
]

def testing(**kwargs):
    print(kwargs)

for config in LLM_CONFIGS:
    print(config.get("kwargs")) #unpacks dictionary into keyword args
    

# I should take name and the llm
#llms: List[Dict[str, Any]] =  [{"name": config["name"] , "llm": _build_llm(config["name"], config.get("kwargs"))}for config in LLM_CONFIGS]

#print(llms)

# class methods - Allow operations related to the class itself
# They take (cls) as the first parameter, which represents the class itself
# Instance method take self
# self refer to class and not any object


class Student:

    # Class variable ~ count how many student we create
    count = 0
    total_gpa = 0

    # All student will have a name and GPA
    def __init__(self, name, gpa):
        self.name = name
        self.gpa = gpa
        #Access Student class
        Student.count +=1
        Student.gpa += gpa

    # INSTANCE METHOD
    # self refer to object we are working with
    def get_info(self):
        return f"{self.name} {self.gpa}"

    # CLASS METHOD
    # To work with class data
    @classmethod
    def get_count(cls):
        return f"Total nb of student: {cls.count}"


    @classmethod
    def get_average_gpa(cls):
        # count variable of my class - no student
        if cls.count == 0:
            return 0
        else:

            avg = cls.total_gpa/cls.count
            return f"Average: {avg}"

        


print("="*100)

config = next((c for c in LLM_CONFIGS if c["name"]== "gpt"), None)

print("Config:")
print(config)

if not config:
    # show the available model
    availabe = " ".join(c["name"] for c in LLM_CONFIGS)
    print(availabe)