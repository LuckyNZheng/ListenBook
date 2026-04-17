"""提示词模块"""

from listen_book.prompts.query_prompt import (
    BOOK_RECOGNITION_PROMPT,
    ANSWER_GENERATION_PROMPT,
    QUERY_REWRITE_PROMPT,
)
from listen_book.prompts.import_prompt import (
    BOOK_NAME_EXTRACT_PROMPT,
    CONTENT_TYPE_CLASSIFY_PROMPT,
)

__all__ = [
    "BOOK_RECOGNITION_PROMPT",
    "ANSWER_GENERATION_PROMPT",
    "QUERY_REWRITE_PROMPT",
    "BOOK_NAME_EXTRACT_PROMPT",
    "CONTENT_TYPE_CLASSIFY_PROMPT",
]