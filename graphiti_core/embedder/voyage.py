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

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import voyageai
else:
    try:
        import voyageai
    except ImportError:
        raise ImportError(
            'voyageai is required for VoyageAIEmbedderClient. '
            'Install it with: pip install graphiti-core[voyageai]'
        ) from None

from pydantic import Field

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'voyage-3'


class VoyageAIEmbedderConfig(EmbedderConfig):
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    api_key: str | None = None


class VoyageAIEmbedder(EmbedderClient):
    """
    VoyageAI Embedder Client
    """

    def __init__(self, config: VoyageAIEmbedderConfig | None = None):
        super().__init__()
        if config is None:
            config = VoyageAIEmbedderConfig()
        self.config = config
        self.client = voyageai.AsyncClient(api_key=config.api_key)  # type: ignore[reportUnknownMemberType]

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        if isinstance(input_data, str):
            input_list = [input_data]
        elif isinstance(input_data, list):
            input_list = [str(i) for i in input_data if i]
        else:
            input_list = [str(i) for i in input_data if i is not None]

        input_list = [i for i in input_list if i]
        if len(input_list) == 0:
            return []

        with self._embedding_span(
            'embedding.create',
            input_count=len(input_list),
            model_name=self.config.embedding_model,
        ) as span:
            result = await self.client.embed(input_list, model=self.config.embedding_model)
            embedding = [float(x) for x in result.embeddings[0][: self.config.embedding_dim]]
            span.add_attributes({'embedding.vector_length': len(embedding)})
            return embedding

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        with self._embedding_span(
            'embedding.create_batch',
            input_count=len(input_data_list),
            model_name=self.config.embedding_model,
        ) as span:
            result = await self.client.embed(input_data_list, model=self.config.embedding_model)
            vectors = [
                [float(x) for x in embedding[: self.config.embedding_dim]]
                for embedding in result.embeddings
            ]
            if vectors:
                span.add_attributes({'embedding.vector_length': len(vectors[0])})
            return vectors
