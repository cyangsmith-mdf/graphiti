"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Generator

from pydantic import BaseModel, Field

from ..tracer import NoOpTracer, Tracer, TracerSpan

EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', 1024))


class EmbedderConfig(BaseModel):
    embedding_dim: int = Field(default=EMBEDDING_DIM, frozen=True)


class EmbedderClient(ABC):
    def __init__(self):
        self.tracer: Tracer = NoOpTracer()

    def set_tracer(self, tracer: Tracer) -> None:
        self.tracer = tracer

    @abstractmethod
    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        pass

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        raise NotImplementedError()

    def _get_provider_type(self) -> str:
        class_name = self.__class__.__name__.lower()
        if 'openai' in class_name or 'azure' in class_name:
            return 'openai'
        if 'gemini' in class_name:
            return 'gemini'
        if 'voyage' in class_name:
            return 'voyage'
        return 'unknown'

    @contextmanager
    def _embedding_span(
        self,
        operation: str,
        *,
        input_count: int | None = None,
        model_name: str | None = None,
    ) -> Generator[TracerSpan, None, None]:
        with self.tracer.start_span(operation, skip_prefix=True) as span:
            attributes: dict[str, int | str | float] = {
                'openinference.span.kind': 'embedding',
                'embedding.operation': operation,
                'embedding.provider': self._get_provider_type(),
                'llm.token_count.prompt': 0,
            }
            if input_count is not None:
                attributes['embedding.input.count'] = input_count
            if model_name:
                attributes['embedding.model_name'] = model_name
            span.add_attributes(attributes)
            yield span
