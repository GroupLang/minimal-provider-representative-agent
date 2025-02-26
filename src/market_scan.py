import asyncio

import httpx
from loguru import logger

from src import utils
from src.config import SETTINGS, Settings
from src.agents.chatbot import invoke_openai_model

TIMEOUT = httpx.Timeout(10.0)


async def _is_background_decomposable(background: str) -> bool:
    """Check if a background text can be decomposed into small tasks using ChatGPT.

    Args:
        background: The background text to analyze

    Returns:
        bool: True if the background can be decomposed into small tasks, False otherwise
    """
    prompt = f"""
    Analyze the following request and determine if it can be decomposed into small, manageable tasks:
    
    {background}
    
    Respond with only 'YES' if the request can be broken down into small tasks, or 'NO' if it cannot.
    """
    
    try:
        response = await invoke_openai_model(prompt)
        return response.strip().upper() == "YES"
    except Exception as e:
        logger.error("Error checking if background is decomposable: {}", str(e))
        return False


async def _create_proposal_for_instance(instance: dict, settings: Settings) -> None:
    instance_id = instance["id"]
    if utils.find_github_repo_url(instance["background"]):
        logger.info("Instance id {} have a github repo url", instance_id)
        return
    
    if not await _is_background_decomposable(instance["background"]):
        logger.info("Instance id {} cannot be decomposed into small tasks", instance_id)
        return

    logger.info("Creating proposal for instance id: {}", instance_id)

    headers = {
        "x-api-key": settings.market_api_key,
        "Accept": "application/json",
    }
    url = f"{settings.market_url}/v1/proposals/create/for-instance/{instance_id}"
    data = {
        "max_bid": settings.max_bid,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)

    response.raise_for_status()
    logger.info(f"Proposal for instance id {instance_id} created successfully")


async def async_market_scan_handler() -> None:
    headers = {
        "x-api-key": SETTINGS.market_api_key,
        "Accept": "application/json",
    }
    url = f"{SETTINGS.market_url}/v1/instances/"
    params = {"instance_status": SETTINGS.market_open_instance_code}

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(url, headers=headers, params=params)

    response.raise_for_status()
    open_instances = response.json()

    if not open_instances:
        logger.debug("No open instances found")
        return

    logger.debug(f"Found {len(open_instances)} open instances")
    url = f"{SETTINGS.market_url}/v1/proposals/"
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(url, headers=headers)

    response.raise_for_status()
    proposals = response.json()

    filled_instances = set(proposal["instance_id"] for proposal in proposals)
    tasks = [
        _create_proposal_for_instance(instance, SETTINGS)
        for instance in open_instances
        if instance["id"] not in filled_instances
    ]
    await asyncio.gather(*tasks)


def market_scan_handler() -> None:
    asyncio.run(async_market_scan_handler())
