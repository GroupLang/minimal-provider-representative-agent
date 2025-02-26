from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import httpx
import openai
from loguru import logger

from src.agents import process_message
from src.config import SETTINGS, Settings
from src.enums import ModelName

TIMEOUT = httpx.Timeout(10.0)

openai.api_key = SETTINGS.openai_api_key
WEAK_MODEL = "gpt-4o-mini"


@dataclass
class InstanceToSolve:
    instance: dict
    messages_history: Optional[str] = None


def _get_instance_to_solve(instance_id: str, settings: Settings) -> Optional[InstanceToSolve]:
    try:
        headers = {
            "x-api-key": settings.market_api_key,
        }
        with httpx.Client(timeout=TIMEOUT) as client:
            instance_endpoint = f"{settings.market_url}/v1/instances/{instance_id}"
            response = client.get(instance_endpoint, headers=headers)
            instance = response.json()

            if not instance.get("status") or instance["status"] != settings.market_resolved_instance_code:
                return None

        with httpx.Client(timeout=TIMEOUT) as client:
            chat_endpoint = f"{settings.market_url}/v1/chat/{instance_id}"
            response = client.get(chat_endpoint, headers=headers)

            chat = response.json()
            if isinstance(chat, dict) and chat.get("detail"):
                return None

            if not chat:
                return InstanceToSolve(instance=instance)

            sorted_messages = sorted(chat, key=lambda m: m["timestamp"])
            
            if not sorted_messages or sorted_messages[-1]["sender"] != "requester":
                return None

            messages_history = "\n\n".join(
                [f"{message['sender']}: {message['message']}" for message in sorted_messages]
            )

            return InstanceToSolve(
                instance=instance,
                messages_history=messages_history,
            )
    except Exception:
        return None


def _process_instance(instance_to_solve: InstanceToSolve) -> Optional[str]:
    logger.info("Processing instance id: {}", instance_to_solve.instance["id"])
    
    try:
        background = instance_to_solve.instance.get("background", "")
        last_message = ""
        
        if instance_to_solve.messages_history:
            messages = instance_to_solve.messages_history.split("\n\n")
            last_message = messages[-1].replace("requester: ", "")
        
        full_context = f"Initial message {background}\n\nLast message: {last_message}" if last_message else background
        
        workflow_tasks = process_message(
            message=full_context,
            chat_history=instance_to_solve.messages_history
        )
        
        if not workflow_tasks:
            logger.info("Could not generate response for instance")
            return None

        return workflow_tasks

    except Exception as e:
        logger.error(
            "Error processing instance {}: {}",
            instance_to_solve.instance["id"],
            str(e),
            exc_info=True,
        )
        return None


def get_awarded_proposals(settings: Settings) -> Optional[list[dict]]:
    try:
        headers = {
            "x-api-key": settings.market_api_key,
            "Accept": "application/json",
        }
        url = f"{settings.market_url}/v1/proposals/"

        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            all_proposals = response.json()

        current_time = datetime.utcnow()
        one_day_ago = current_time - timedelta(days=1)

        awarded_proposals = [
            p
            for p in all_proposals
            if p["status"] == settings.market_awarded_proposal_code
            and datetime.fromisoformat(p["creation_date"]) > one_day_ago
        ]
        logger.info(f"Found {len(awarded_proposals)} awarded proposals in the last 24 hours")
        return awarded_proposals
    except Exception as e:
        logger.error(f"Failed to get awarded proposals: {str(e)}", exc_info=True)
        return None


def _send_message(instance_id: str, message: str, settings: Settings) -> Optional[bool]:
    try:
        headers = {
            "x-api-key": settings.market_api_key,
        }
        url = f"{settings.market_url}/v1/chat/send-message/{instance_id}"
        data = {"message": message}

        response = httpx.post(url, headers=headers, json=data)
        response.raise_for_status()
        return True
    except Exception:
        return None

def _send_instances_from_workflow(instance_to_solve: InstanceToSolve, workflow_tasks: str) -> None:
    """
    Create a new instance based on workflow tasks.
    
    Args:
        instance_to_solve: The original instance that triggered the workflow
        workflow_tasks: The workflow tasks to include in the new instance
    """
    logger.info(f"Creating new instance from workflow for instance {instance_to_solve.instance['id']}")
    
    try:
        # Prepare the data for the new instance
        instance_data = {
            "background": workflow_tasks,
            "max_credit_per_instance": SETTINGS.max_bid,  # Use the max bid from settings
            "percentage_reward": 1,  # Default percentage reward
            "side_effect_free": True,  # Set as side-effect free
            "representative_agent": True,  # Mark as representative agent instance
            "max_providers": 1,  # Default to 1 provider
        }
        
        # Make the API request to create a new instance
        headers = {
            "x-api-key": SETTINGS.market_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        url = f"{SETTINGS.market_url}/v1/instances"
        
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.post(url, headers=headers, json=instance_data)
            response.raise_for_status()
            new_instance = response.json()
            
        logger.info(f"Successfully created new instance: {new_instance.get('id')}")
        
    except httpx.HTTPStatusError as e:
        logger.error(
            f"HTTP error creating instance from workflow: {e.response.status_code} - {e.response.text}"
        )
    except Exception as e:
        logger.error(
            f"Error creating instance from workflow: {str(e)}",
            exc_info=True,
        )


def solve_instances_handler() -> None:
    logger.info("Processing instances handler")
    awarded_proposals = get_awarded_proposals(SETTINGS)

    if not awarded_proposals:
        return

    logger.info(f"Found {len(awarded_proposals)} awarded proposals")

    for p in awarded_proposals:
        instance_to_solve = _get_instance_to_solve(p["instance_id"], SETTINGS)
        if not instance_to_solve:
            continue

        workflow_tasks = _process_instance(instance_to_solve)
        if not workflow_tasks:
            continue

        _send_instances_from_workflow(instance_to_solve, workflow_tasks)

        joined_tasks = "\n\n".join(workflow_tasks)
    
        _send_message(instance_to_solve.instance["id"], joined_tasks, SETTINGS)
        logger.info(f"Sent message to instance {instance_to_solve.instance['id']}")
