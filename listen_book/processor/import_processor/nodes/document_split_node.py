"""文档切分节点"""

import os
import re
import json
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from listen_book.processor.import_processor.base import BaseNode
from listen_book.processor.import_processor.state import ImportGraphState
from listen_book.core.config import get_settings


class DocumentSplitNode(BaseNode):
    """文档切分节点"""

    name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("step1", "文档切分开始")

        settings = get_settings()
        max_length = settings.max_chunk_length
        min_length = settings.min_chunk_length

        md_content = state.get("md_content", "")
        file_title = state.get("file_title", "")

        if not md_content:
            # 如果没有 md_content，尝试从文件读取
            md_path = state.get("md_path", "")
            if md_path and os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    md_content = f.read()

        if not md_content:
            raise ValueError("缺少文档内容")

        # 统一换行符
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 第一次切分：按标题切分
        sections = self._split_by_headings(md_content, file_title)

        # 第二次切分：处理过长/过短的章节
        final_sections = self._split_and_merge(sections, max_length, min_length)

        # 组装 chunks
        chunks = self._assemble_chunks(final_sections, state)

        state["chunks"] = chunks
        self.log_step("step2", f"切分完成，共 {len(chunks)} 个切片")

        # 备份
        self._backup(chunks, state)

        return state

    def _split_by_headings(self, md_content: str, file_title: str) -> List[Dict]:
        """按标题切分"""
        current_title = ""
        level = 0
        body_lines = []
        sections = []
        in_fence = False
        hierarchy = [""] * 7

        def flush():
            body = "\n".join(body_lines)
            if current_title or body:
                parent_title = ""
                for i in range(level - 1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break
                if not parent_title:
                    parent_title = current_title if current_title else file_title
                sections.append({
                    "title": current_title if current_title else file_title,
                    "body": body,
                    "parent_title": parent_title,
                    "file_title": file_title,
                })

        lines = md_content.split("\n")
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        for line in lines:
            if line.strip().startswith("```") or line.strip().startswith("~~~"):
                in_fence = not in_fence

            match = heading_re.match(line) if not in_fence else None
            if match:
                flush()
                current_title = line
                level = len(match.group(1))
                for i in range(level + 1, 7):
                    hierarchy[i] = ""
                body_lines = []
            else:
                body_lines.append(line)

        flush()
        return sections

    def _split_and_merge(self, sections: List[Dict], max_len: int, min_len: int) -> List[Dict]:
        """切分过长章节，合并过短章节"""
        new_sections = []
        for section in sections:
            new_sections.extend(self._split_long_section(section, max_len))

        return self._merge_short_sections(new_sections, min_len)

    def _split_long_section(self, section: Dict, max_len: int) -> List[Dict]:
        """切分过长章节"""
        body = section.get("body", "")
        title = section.get("title", "")
        parent_title = section.get("parent_title", "")
        file_title = section.get("file_title", "")

        if len(title) > 100:
            title = title[:100]

        title_prefix = f"{title}\n\n"
        total_len = len(body) + len(title_prefix)

        if total_len <= max_len:
            return [section]

        body_len = max_len - len(title_prefix)
        if body_len <= 0:
            return [section]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_len,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "；", "？", ".", "?", "!", " ", ""],
        )

        parts = splitter.split_text(body)
        if len(parts) == 1:
            return [section]

        sub_sections = []
        for i, part in enumerate(parts):
            sub_sections.append({
                "body": part,
                "title": f"{title}_{i + 1}",
                "parent_title": parent_title,
                "file_title": file_title,
            })
        return sub_sections

    def _merge_short_sections(self, sections: List[Dict], min_len: int) -> List[Dict]:
        """合并同源短章节"""
        if len(sections) <= 1:
            return sections

        result = []
        current = sections[0]

        for next_section in sections[1:]:
            same_parent = current.get("parent_title") == next_section.get("parent_title")
            if same_parent and len(next_section.get("body", "")) < min_len:
                current["body"] = current.get("body", "").rstrip() + "\n\n" + next_section.get("body", "").lstrip()
                current["title"] = current.get("parent_title", "")
            else:
                result.append(current)
                current = next_section

        result.append(current)
        return result

    def _assemble_chunks(self, sections: List[Dict], state: ImportGraphState) -> List[Dict]:
        """组装最终 chunks"""
        chunks = []
        book_info = {
            "book_name": state.get("book_name", ""),
            "author": state.get("author", ""),
            "content_type": state.get("content_type", "书籍简介"),
            "entry_name": state.get("entry_name", ""),
            "category": state.get("category", ""),
            "audio_duration": state.get("audio_duration", ""),
            "source_file": state.get("file_name", ""),
            "source_url": state.get("source_url", ""),
        }

        for section in sections:
            body = section.get("body", "")
            title = section.get("title", "")
            content = f"{title}\n\n{body}"

            chunk = {
                "content": content,
                "title": title,
                "parent_title": section.get("parent_title", ""),
                "file_title": section.get("file_title", ""),
            }
            chunk.update(book_info)
            chunks.append(chunk)

        return chunks

    def _backup(self, chunks: List[Dict], state: ImportGraphState):
        """备份切片到 JSON 文件"""
        file_dir = state.get("file_dir", "")
        if not file_dir:
            return
        try:
            output_path = os.path.join(file_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            self.log_step("backup", f"切片已备份到 {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")