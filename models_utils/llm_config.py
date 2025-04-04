import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

import pathlib
pyfile = pathlib.Path(__file__).parent.resolve()

class LLMConfig:
    def __init__(self, provider: str = "azure"):

        ##### CONSTANTS #######
        self.AZURE_OPENAI_BASE_URL = os.environ["AZURE_OPENAI_BASE_URL"]
        self.OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
        self.OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"]

        ##### VARIABLES #######
        self.provider = provider
        self.DEFAULT_AZURE_MODEL = "gpt-4o"
        self.DEFAULT_AWS_MODEL = "claude-3-5-sonnet"


    def get_llm(self, **kwargs):

        if self.provider == "azure":
            return self.get_azure_llm(**kwargs)

        elif self.provider == "aws_bedrock":
            return self.get_aws_bedrock_llm(**kwargs)

        else:
            raise ValueError(f"Provider {self.provider} not supported")

    def get_azure_llm(self, **kwargs):
        from langchain_openai import AzureChatOpenAI

        params = {
            "openai_api_key": self.OPENAI_API_KEY,
            "openai_api_version": self.OPENAI_API_VERSION,
            "azure_endpoint": self.AZURE_OPENAI_BASE_URL,
        }
        
        if kwargs.get("model", None) is None: kwargs["model"] = self.DEFAULT_AZURE_MODEL

        if kwargs.get("model").startswith("o1") or kwargs.get("model").startswith("o3"):
            # constraints of o1 and o3 family
            kwargs["temperature"] = 1
            kwargs["disabled_params"] = {"parallel_tool_calls": None}

        return AzureChatOpenAI(**params, **kwargs)

    def get_aws_bedrock_llm(self, **kwargs):
        pass
