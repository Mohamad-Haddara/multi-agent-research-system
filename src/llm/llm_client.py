"""
Start with Internal types ~ dataclass. They only pass between our code and the LLM client.

Dataclasses is used to organize, store, transfer data across our application.
Having dataclass allow us to organize our data in a defined structure and pass it as a single item to functions.

"""


from dataclasses import dataclass, field
import logging
from typing import Literal


logger = logging.getLogger(__name__)





# Step 1: Types
# Agents get these objects, and not SDK provider, so chaning the provider ot upgrading the SDL doesn't affect our code
# Dataclass vs Pydantic
# Pydantic validate data that come from OUTSIDE (env vars, HTTP bodies)
# dataclass for internal ~ frozen dataclasses are enough ~ immutable

Role = Literal["system", "user", "assistant"]

# Use dataclasses to decorate the Message and Usage classes as special classes for holding data
@dataclass(frozen=True, slots=True) # Frozen: no body can change it by accident
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Usage:
    input_token: int = 0
    output_token: int = 0


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str # model that actually answered: useful for logs and cost
    finish_reason: str # Why generation stopped 
    usage: Usage = field(default_factory=Usage)




# step 2 Error
# Add RateLimitError
# Every failure is now one of our exceptions
# retryable - answers one question for the caller
# The SDK has Already retried 'max_retries' times before I see retryable error
# SDK is a software development kit. Here means openai package. It is a library that does the HTTP work for us: it builds the request, sends it, reads the JSON reply, and retries by itself when the server is busy.
# In our code it's AsyncOpenAI, client.chat.completion.create(...).
# Azure, vLLM, ollama, and openRouter all accepts the same requests so this one SDK works with them all.
# When something fails, SDK raises its own exceptions such as openai.RateLimitError ~ Those belong to the openai package, not me


# One place of inheritance - family of errors
class LLMError(Exception):
    """Base class: 'except LLMError' catches every (all) LLM failure."""
    retryable: bool = False




class LLMTimeoutError(LLMError):
    """
    No reply in time. Model overloaded or very long prompt
    """

    retryable: bool = True # a subclass changes the answer


class LLMConnectionError(LLMError):
    """
    Server unreachable. The server can't be reached (down, DNS, network). vLLM down, wrong URL. firewall
    """
    retryable: bool = True



class LLMRateLimitError(LLMError):
    """
    Too many request(429). Azure or OpenRouter quota used up
    """
    retryable: bool = True


class LLMServerError(LLMError):
    """
    Provider failure (5xx). Outage, GPU out of memory
    """
    retryable: bool = True



class LLMResponseError(LLMError):
    """
    A reply came back, but it is empty or broken. Unreliable provider.
    """
    retryable: bool = True



class LLMAuthError(LLMError):
    """
    Not allowed (401/403). Wrong key or no access. Retrying won't help: fix the config
    """
    retryable: bool = True


class LLMRequestError(LLMError):
    """
    The request is invalid (400,404, 422). We sent a bad request (wrong deployment name, wrong model name, prompt longer than the context model window, invalid parameter).
    Fix the code
    """
    retryable: bool = True



class LLMContentFilterError(LLMError):
    """
    Safety filter blocked the prompt or reply. Azure content filter
    """
    retryable: bool = True


def is_retryable(error: BaseException) -> bool:
    """
    For retry policies, e.g. LangGraph's RetryPolicy(retry_on=is_retryable).
    """

    return isinstance(error, LLMError) and error.retryable


"""
- raise _to_llm_error(error) from error keeps the original SDK error attached, so the traceback still shows the full HTTP details when you debug.
- is_retryable() returns False for errors that are not ours. A KeyError is a bug in your code and should never be retried.

Then using them in an agent
"""


# step 3 Interface
# 