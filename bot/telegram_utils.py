"""Small Telegram transport helpers shared by bot handlers.

The Bot API rejects messages whose markup is malformed. Model output and
runtime status values are dynamic, so MarkdownV2 formatting must never be a
single point of failure for a user-facing response.
"""

from __future__ import annotations

import logging
from typing import Any

from telegram.error import BadRequest

logger = logging.getLogger(__name__)


async def reply_text_safe(message: Any, text: str, **kwargs: Any) -> Any:
    """Reply with the requested formatting and fall back to plain text.

    This preserves existing formatting when it is valid while ensuring a
    malformed Markdown/MarkdownV2 response does not prevent the user from
    receiving the underlying message. If the plain-text retry also fails, its
    exception is allowed to propagate so genuine Telegram failures remain
    observable to the caller.
    """
    try:
        return await message.reply_text(text, **kwargs)
    except BadRequest:
        parse_mode = kwargs.get("parse_mode")
        if not parse_mode:
            raise

        logger.warning(
            "Telegram rejected formatted reply; retrying as plain text",
            exc_info=True,
        )
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("parse_mode", None)
        return await message.reply_text(text, **fallback_kwargs)
