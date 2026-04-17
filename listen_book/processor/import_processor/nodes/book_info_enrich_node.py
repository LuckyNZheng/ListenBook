"""书籍信息补充节点

通过 LLM 和可选的联网搜索补充缺失的书籍信息。
"""

import json
import re
from typing import Dict, Any, List, Optional

from listen_book.processor.import_processor.base import BaseNode
from listen_book.processor.import_processor.state import ImportGraphState
from listen_book.utils.client.ai_clients import AIClients
from listen_book.core.config import get_settings

# 信息补充提示词
BOOK_INFO_ENRICH_PROMPT = """你是一位书籍信息专家。请根据以下信息，补充和完善书籍的元数据。

已知信息：
- 书名：{book_name}
- 文件标题：{file_title}
- 内容片段：
{content_sample}

请读取得到以下信息：
1. 作者名（author）
2. 书籍类别（category，如：科幻、悬疑、历史、文学、教育、儿童等）
3. 有声书时长（audio_duration，格式如：12小时30分钟）
4. 出版社（publisher）
5. 出版年份（publish_year）
6. 书籍简介摘要（summary，100字以内）
7. 适合人群（target_audience）
8. 推荐理由（recommend_reason）

如果内容片段没有以上信息，那么通过你自己的知识进行补充，实在没有的就填写空字符串

最后请以 JSON 格式返回：
{
    "author": "",
    "category": "",
    "audio_duration": "",
    "publisher": "",
    "publish_year": "",
    "summary": "",
    "target_audience": "",
    "recommend_reason": ""
}

只返回 JSON，不要其他内容。"""


class BookInfoEnrichNode(BaseNode):
    """书籍信息补充节点"""

    name = "book_info_enrich_node"

    # 需要补充的字段列表
    ENRICH_FIELDS = [
        "author", "category", "audio_duration",
        "publisher", "publish_year", "summary",
        "target_audience", "recommend_reason"
    ]

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("step1", "书籍信息补充开始")

        # settings = get_settings()
        book_name = state.get("book_name", "")
        file_title = state.get("file_title", "")
        md_content = state.get("md_content", "")

        # 检查哪些字段需要补充
        missing_fields = self._get_missing_fields(state)

        # 如果所有字段都已填写，跳过补充
        if not missing_fields and book_name:
            self.log_step("step2", "所有字段已填写，跳过补充")
            return state

        # 取内容样本
        content_sample = md_content[:3000] if md_content else ""

        if not content_sample and not book_name:
            self.log_step("step2", "缺少内容样本和书名，无法补充")
            return state

        try:
            llm = AIClients.get_llm_client(json_mode=True)
            prompt = BOOK_INFO_ENRICH_PROMPT.format(
                book_name=book_name or file_title,
                file_title=file_title,
                content_sample=content_sample[:2000] if content_sample else "无内容"
            )
            response = llm.invoke(prompt)
            result = json.loads(response.content)

            # 补充缺失字段
            enriched_count = 0
            for field in self.ENRICH_FIELDS:
                current_value = state.get(field, "")
                new_value = result.get(field, "")

                # 只补充空字段
                if not current_value and new_value:
                    state[field] = new_value
                    enriched_count += 1
                    self.log_step("enrich", f"{field}: {new_value}")

            self.log_step("step3", f"补充完成，共补充 {enriched_count} 个字段")

        except Exception as e:
            self.logger.warning(f"LLM 信息补充失败: {e}")

        return state

    def _get_missing_fields(self, state: ImportGraphState) -> List[str]:
        """获取缺失的字段列表"""
        missing = []
        for field in self.ENRICH_FIELDS:
            if not state.get(field):
                missing.append(field)
        return missing


if __name__ == '__main__':
    llm = AIClients.get_llm_client(json_mode=True)
    res= llm.invoke(["你是什么模型"])
    print(res)


    node = BookInfoEnrichNode()

    state = {
        "book_name": "蛤蟆先生去看心理医生",
        "file_title": "蛤蟆先生去看心理医生简介",
        "md_content": "书名：蛤蟆先生去看心理医生,作者名：罗伯特·戴博德,条目名称：全书 类别/标签：心理学, 情绪, 自我疗愈,《蛤蟆先生去看心理医生》有声书整体时长约 5 小时适合入门"
    }

    state = node.process(state)
    print(state)
