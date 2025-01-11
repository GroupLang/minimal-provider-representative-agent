import os

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    openai_api_key: str = Field(..., description="The API key for OpenAI.")
    github_pat: str = Field(..., description="The personal access token for GitHub.")
    github_username: str = Field(..., description="The GitHub username.")
    github_email: str = Field(..., description="The GitHub email.")

    market_url: str = Field("https://api.agent.market", description="The URL for the market.")
    market_api_key: str = Field(..., description="The API key for the market.")

    market_open_instance_code: int = Field(
        0, description="The code for an open instance in the market."
    )
    market_resolved_instance_code: int = Field(
        3, description="The code for a resolved instance in the market."
    )
    market_awarded_proposal_code: int = Field(
        1, description="The code for an awarded proposal in the market."
    )

    max_bid: float = Field(0.01, gt=0, description="The maximum bid for a proposal.")

    class Config:
        case_sensitive = False

    @classmethod
    def load_settings(cls) -> "Settings":
        return cls()


SETTINGS = Settings.load_settings()
