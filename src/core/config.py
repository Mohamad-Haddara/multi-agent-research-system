"""
Application configuration management.

This module handles everything comes from environment variables (or .env files) loading, parsing and
management for the application. It includes environment detection, .env file loading, and configuration
value parsing.

"""

from ntpath import isfile
import os

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Any

from dotenv import load_dotenv

import logging

from pydantic import Field, ValidationInfo, field_validator, model_validator

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)



# Define environment types
class Environment(str, Enum):
    """
    Application environment types.

    Defines the possible environments the application can run in:
    development, staging, production and test.

    """
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


# Determine environment
def get_environment() -> Environment:
    """
    Get the current environment.

    Returns:
        Environment: The current environment (development, staging, production, or test)
    """

    match os.getenv("APP_ENV", "development").lower():

        case "production" | "prod":
            return Environment.PRODUCTION

        case "staging" | "stag":
            return Environment.STAGING

        case "development" | "dev":
            return Environment.DEVELOPMENT

        case _:
            return Environment.TEST


# load appropriate .env file based on environment
def load_env_file():
    """Load environment-specific .env file."""
    env = get_environment()

    logger.info(f"Loading Environment: {env}")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


    # Define env files in priority order
    env_files = [
        os.path.join(base_dir, f".env.{env.value}.local"),
        os.path.join(base_dir, f".env.{env.value}"),
        os.path.join(base_dir, ".env.local"),
        os.path.join(base_dir, ".env")
    ]


    # load the first env file that exists
    for env_file in env_files:
        if os.path.isfile(env_file):
            load_dotenv(dotenv_path=env_file)
            logger.info(f"Loaded environment from {env_file}")
            return env_file

    
    # fall back to default if no env file found
    return None

ENV_FILE = load_env_file()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive= False,
        extra="ignore",
    )
    

    # ----- APP -----
    app_env: Literal["development", "production", "staging", "test"] = "development"
    project_name: str = "Multi Agent Research System"
    version: str = "1.0.0"
    description: str = "A production-read Multi-Agent Research System with LangGraph and Langfuse integration"
    
    # ----- OpenAI -----
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    # ----- LangGraph Configuration -----
    max_tokens: int = 2000
    
    # ----- Logging -----
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    # ----- API -----
    api_prefix: str = "/api/v1"









@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings instance

    """

    return Settings()