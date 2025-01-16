from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import httpx
import openai
from loguru import logger

from src.agents import estimate_reward
from src.config import SETTINGS, Settings
from src.enums import ModelName

TIMEOUT = httpx.Timeout(10.0)

openai.api_key = SETTINGS.openai_api_key
WEAK_MODEL = "gpt-4o-mini"


@dataclass
class InstanceToSolve:
    instance: dict
    messages_history: Optional[str] = None


def _submit_reward(instance_id: str, settings: Settings, reward: float) -> Optional[bool]:
    try:
        headers = {
            "x-api-key": settings.market_api_key,
            "Accept": "application/json",
        }
        
        url = f"{settings.market_url}/v1/instances/{instance_id}/report-reward"
        data = {"gen_reward": reward}
        
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.put(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
    except Exception:
        logger.error(f"Failed to submit reward for instance {instance_id}")
        return None


def _get_instance_to_solve(instance_id: str, settings: Settings) -> Optional[InstanceToSolve]:
    try:
        headers = {
            "x-api-key": settings.market_api_key,
        }
        with httpx.Client(timeout=TIMEOUT) as client:
            instance_endpoint = f"{settings.market_url}/v1/instances/{instance_id}"
            response = client.get(instance_endpoint, headers=headers)
            instance = response.json()

            if instance.get("reward_estimation_id") is None:
                return None

            if (
                not instance.get("status")
                or instance["status"] != settings.market_resolved_instance_code
            ):
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

            messages_history = "\n\n".join(
                [f"{message['sender']}: {message['message']}" for message in sorted_messages]
            )

            return InstanceToSolve(
                instance=instance,
                messages_history=messages_history,
            )
    except Exception:
        return None


def _solve_instance(instance_to_solve: InstanceToSolve) -> Optional[str]:
    logger.info("Solving instance id: {}", instance_to_solve.instance["id"])
    
    try:
        reward = estimate_reward(
            background=instance_to_solve.instance.get("background", ""),
            chat_messages=instance_to_solve.messages_history
        )
        
        if reward is None:
            logger.info("Could not estimate reward for instance")
            return None, None
            
        if reward < 0:
            logger.info("Negative reward value, skipping instance")
            return None, None

        return f"Estimated reward value: {reward}", reward

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


def solve_instances_handler() -> None:
    logger.info("Solve instances handler")
    awarded_proposals = get_awarded_proposals(SETTINGS)

    if not awarded_proposals:
        return

    logger.info(f"Found {len(awarded_proposals)} awarded proposals")

    for p in awarded_proposals:
        instance_to_solve = _get_instance_to_solve(p["instance_id"], SETTINGS)
        if not instance_to_solve:
            continue

        message, reward = _solve_instance(instance_to_solve)
        if not message:
            continue

        _send_message(instance_to_solve.instance["id"], message, SETTINGS)

        logger.info(f"Sent message to instance {instance_to_solve.instance['id']}")

        if isinstance(reward, float):
            _submit_reward(instance_to_solve.instance["id"], SETTINGS, reward)
            logger.info(f"Submitted reward for instance {instance_to_solve.instance['id']}")
