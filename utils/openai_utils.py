import asyncio
import logging
import os
from typing import Any
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio
import random
from time import sleep
import sys
import aiolimiter

import openai
from openai import AsyncOpenAI, OpenAIError


def prepare_message(SYSTEM_INPUT, USER_INPUT):
    cur_message = [
        {
            "role": "system",
            "content": SYSTEM_INPUT
        },
        {
            "role": "user",
            "content": USER_INPUT,
        }
    ]
    return cur_message

def prepare_remove_message(USER_INPUT):
    cur_message = [
        {
            "role": "system",
            "content": "Remove sentences about experimental design and results: "
        },
        {
            "role": "user",
            "content": USER_INPUT,
        }
    ]
    return cur_message

def prepare_generation_input(title, abstract, sections, filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        SYSTEM_INPUT=file.read()
    return SYSTEM_INPUT,f"Paper title: {title}\n\nPaper abstract: {abstract}\n\nPaper Sections: {sections}"

def prepare_remove_input(title, abstract, introduction, filepath):
    with open(filepath,'r',encoding='utf-8') as file:
        SYSTEM_INPUT=file.read()
        print(SYSTEM_INPUT)
    return SYSTEM_INPUT,f"Paper title: {title}\n\nPaper abstract: {abstract}\n\nIntroduction: {introduction}\n\n"


async def _throttled_openai_chat_completion_acreate(
    client: AsyncOpenAI,
    model: str,
    messages,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
    response_format: dict = {},
):
    async with limiter:
        for _ in range(10):
            try:
                if response_format["type"] == "text":
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
                else: 
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        response_format=response_format,
                    )
            except openai.BadRequestError as e:
                print(e)
                return None
            except OpenAIError as e:
                print(e)
                sleep(random.randint(5, 10))
        return None


async def generate_from_openai_chat_completion(
    client,
    messages,
    engine_name: str,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    requests_per_minute: int = 100,
    response_format: dict = {"type":"text"},
):
    """Generate from OpenAI Chat Completion API.

    Args:
        messages: List of messages to proceed.
        engine_name: Engine name to use, see https://platform.openai.com/docs/models
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """    
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            client,
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
            response_format=response_format,
        )
        for message in messages
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses, file=sys.stdout)
    
    outputs = []
    for response in responses:
        if response:
            outputs.append(response.choices[0].message.content)
        else:
            outputs.append("Invalid Message")
    return outputs


# Example usage
if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "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" # Set your OpenAI API key here
    
    client = AsyncOpenAI()
    AsyncOpenAI.api_key = os.getenv('OPENAI_API_KEY')
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the purpose of life? Output result in json format."},
    ]
    responses = asyncio.run(
        generate_from_openai_chat_completion(
            client,
            messages=[messages]*50, 
            engine_name="gpt-3.5-turbo-0125",
            max_tokens=256,
            response_format={"type":"json_object"},
        )
    )
    print(responses)