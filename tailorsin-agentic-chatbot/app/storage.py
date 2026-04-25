from __future__ import annotations

import logging
from datetime import datetime, timezone
import json
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError


logger = logging.getLogger(__name__)

TIMESTAMP_FIELDS = {
    "last_activity_at",
    "handoff_requested_at",
    "handoff_last_human_message_at",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


def _deserialize_value(key: str, value: Any) -> Any:
    if isinstance(value, dict):
        return {child_key: _deserialize_value(child_key, child_value) for child_key, child_value in value.items()}
    if isinstance(value, list):
        return [_deserialize_value(key, item) for item in value]
    if key in TIMESTAMP_FIELDS and isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return value


class RedisConversationStore:
    def __init__(self, redis_url: str):
        self.redis = Redis.from_url(redis_url, decode_responses=True)
        self.conversation_index_key = "tailorsin:conversations"
        self._memory_store: dict[str, dict[str, Any]] = {}
        self._memory_index: set[str] = set()
        self._memory_mode_logged = False

    def _serialize_conversation(self, conversation: dict[str, Any]) -> dict[str, Any]:
        return {key: _serialize_value(value) for key, value in conversation.items()}

    def _deserialize_conversation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {key: _deserialize_value(key, value) for key, value in payload.items()}

    def _log_memory_fallback(self, exc: Exception, operation: str) -> None:
        if not self._memory_mode_logged:
            logger.warning(
                "Redis unavailable during %s (%s). Falling back to in-memory conversation store.",
                operation,
                exc,
            )
            self._memory_mode_logged = True

    def _log_redis_recovered(self) -> None:
        if self._memory_mode_logged:
            logger.info("Redis connection restored. Continuing with Redis-backed persistence.")
            self._memory_mode_logged = False

    def _conversation_key(self, customer_id: str) -> str:
        return f"tailorsin:conversation:{customer_id}"

    def default_conversation(self) -> dict:
        return {
            "messages": [],
            "handoff_active": False,
            "customer_profile": None,
            "agent_context": {},
            "last_activity_at": None,
            "handoff_summary": None,
            "handoff_requested_at": None,
            "handoff_assigned_to": None,
            "handoff_last_human_message_at": None,
            "audit_log": [],
        }

    async def ping(self) -> bool:
        try:
            ok = bool(await self.redis.ping())
            self._log_redis_recovered()
            return ok
        except (RedisError, OSError) as exc:
            self._log_memory_fallback(exc, "ping")
            return False

    async def get_conversation(self, customer_id: str) -> dict | None:
        try:
            raw = await self.redis.get(self._conversation_key(customer_id))
            if raw is None:
                return None
            payload = json.loads(raw)
            self._log_redis_recovered()
            return self._deserialize_conversation(payload)
        except (RedisError, OSError) as exc:
            self._log_memory_fallback(exc, "get_conversation")
            payload = self._memory_store.get(customer_id)
            if payload is None:
                return None
            return self._deserialize_conversation(payload)

    async def get_or_create_conversation(self, customer_id: str) -> dict:
        conversation = await self.get_conversation(customer_id)
        if conversation is not None:
            return conversation
        conversation = self.default_conversation()
        await self.save_conversation(customer_id, conversation)
        return conversation

    async def save_conversation(self, customer_id: str, conversation: dict) -> None:
        serialized = self._serialize_conversation(conversation)
        try:
            await self.redis.set(self._conversation_key(customer_id), json.dumps(serialized))
            await self.redis.sadd(self.conversation_index_key, customer_id)
            self._log_redis_recovered()
            self._memory_store.pop(customer_id, None)
            self._memory_index.discard(customer_id)
        except (RedisError, OSError) as exc:
            self._log_memory_fallback(exc, "save_conversation")
            self._memory_store[customer_id] = serialized
            self._memory_index.add(customer_id)

    async def list_customer_ids(self) -> list[str]:
        try:
            ids = await self.redis.smembers(self.conversation_index_key)
            self._log_redis_recovered()
            return sorted(ids)
        except (RedisError, OSError) as exc:
            self._log_memory_fallback(exc, "list_customer_ids")
            return sorted(self._memory_index)

    async def list_open_handoffs(self) -> list[dict]:
        handoffs = []
        for customer_id in await self.list_customer_ids():
            conversation = await self.get_conversation(customer_id)
            if not conversation or not conversation.get("handoff_active"):
                continue
            handoffs.append({"customer_id": customer_id, "conversation": conversation})
        return handoffs