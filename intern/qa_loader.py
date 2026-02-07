from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterator, Sequence, TypedDict, overload

from .markers import user, assistant, assistant_start
from .model import ContentBlocks, MaskedContentBlocks

ResolvedItem = tuple[ContentBlocks, int]

class QAItem(TypedDict, total=False):
    question: str
    answer: str

    messages: list[dict[str, Any]]
    images: list[str]

    question_type: str


class QADialogLoader(Sequence[ResolvedItem]):
    resolver: Callable[[str], ContentBlocks]
    question_type_to_model: dict[str, int]
    is_open_ending: bool = False
    
    _indices: Sequence[int] | None = None
    _data: Sequence[QAItem]
    
    def __init__(
        self, 
        data: Sequence[QAItem], 
        resolver: Callable[[str], ContentBlocks],
        question_type_to_model: dict[str, int],
        is_open_ending: bool = False,
        indices: Sequence[int] | None = None
    ) -> None:
        self._data = data
        self.resolver = resolver
        self.question_type_to_model = question_type_to_model
        self.is_open_ending = is_open_ending
        
        if indices is not None and not \
            all(0 <= index < len(self._data) for index in indices):
                
            raise ValueError("All indices must be within the range of the data length")
        self._indices = indices
        
    @property
    def data(self) -> Sequence[QAItem]:
        if self._indices is not None:
            return [self._data[i] for i in self._indices]
        return self._data
    
    @property
    def indices(self) -> Sequence[int]:
        return range(len(self._data)) if self._indices is None else self._indices

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return len(self._data)

    @overload
    def __getitem__(self, index: int) -> ResolvedItem: ...
    
    @overload
    def __getitem__(self, index: slice | list[int]) -> QADialogLoader: ...

    def __getitem__(self, index: int | slice | list[int]) -> ResolvedItem | QADialogLoader:
        if isinstance(index, (slice, list)):
            indices = self.indices
            if isinstance(index, slice):
                indices = indices[index]
            else:
                indices = [indices[i] for i in index]
            
            return self.__class__(
                self._data, self.resolver, self.question_type_to_model, self.is_open_ending, indices
            )
        if self._indices is not None:
            index = self._indices[index]
            
        item = self._data[index]

        question, answer = self._extract_qa(item)
        question = self._inject_images_into_question(question, item)

        if self.is_open_ending:
            q_part = user(question) + assistant_start
            q_resolved = list(self.resolver(q_part))
            resolved = MaskedContentBlocks(content_blocks=q_resolved, loss_mask=[False] * len(q_resolved))
            model_index = self.question_type_to_model.get(item["question_type"], 0)
            return resolved, model_index

        q_part = user(question)
        a_part = assistant(answer)

        q_resolved = list(self.resolver(q_part))
        a_resolved = list(self.resolver(a_part))

        resolved_blocks = q_resolved + a_resolved
        loss_mask = ([False] * len(q_resolved)) + ([True] * len(a_resolved))

        resolved = MaskedContentBlocks(content_blocks=resolved_blocks, loss_mask=loss_mask)
        model_index = self.question_type_to_model.get(item["question_type"], 0)
        return resolved, model_index

    @staticmethod
    def _extract_qa(item: QAItem) -> tuple[str, str]:
        q = item.get("question", None)
        a = item.get("answer", None)
        if isinstance(q, str) and isinstance(a, str):
            return q, a

        msgs = item.get("messages", [])
        q_text = ""
        a_text = ""
        if isinstance(msgs, list):
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role", "")).strip().lower()
                content = m.get("content", "")
                if not isinstance(content, str):
                    content = str(content)

                if role == "user" and not q_text:
                    q_text = content
                elif role == "assistant" and not a_text:
                    a_text = content

        return q_text, a_text

    @staticmethod
    def _inject_images_into_question(question: str, item: QAItem) -> str:
        """
        Your data uses '<image>' placeholder and actual paths are in item['images'].
        Convert into explicit tags that trainer.resolve_content_text can parse:
            <image_path>/abs/path.jpg</image_path>
        """
        images = item.get("images", [])
        if not isinstance(images, list):
            return question
        img_paths = [p for p in images if isinstance(p, str) and p]
        if not img_paths:
            return question

        out = str(question)

        for p in img_paths:
            if "<image>" in out:
                out = out.replace("<image>", f"<image_path>{p}</image_path>", 1)
            else:
                break
            
        if "<image_path>" not in out:
            prefix = "".join(f"<image_path>{p}</image_path>" for p in img_paths)
            out = prefix + out

        return out
    
    def __iter__(self) -> Iterator[ResolvedItem]:
        for i in range(len(self)):
            yield self[i]
            
    def with_open_ending(self, is_open_ending: bool=True) -> QADialogLoader:
        return self.__class__(
            self._data, self.resolver, self.question_type_to_model, is_open_ending=is_open_ending, indices=self._indices
        )
        
    def sample(self, k: int | None = None, ratio: float | None = None) -> QADialogLoader:
        if k is None:
            if ratio is None:
                raise ValueError("Either sample numbers `k` or sample ratio `ratio` must be specified.")
            k = math.ceil(len(self) * ratio)

        sampled_indices = random.sample(self.indices, k)
        
        return self.__class__(
            self._data, self.resolver, self.question_type_to_model, self.is_open_ending, sampled_indices
        )
        
    def choices(self, k: int | None = None, ratio: float | None = None) -> QADialogLoader:
        if k is None:
            if ratio is None:
                raise ValueError("Either sample numbers `k` or sample ratio `ratio` must be specified.")
            k = math.ceil(len(self) * ratio)
        
        chosen_indices = random.choices(self.indices, k=k)
        
        return self.__class__(
            self._data, self.resolver, self.question_type_to_model, self.is_open_ending, chosen_indices
        )