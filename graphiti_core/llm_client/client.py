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

import hashlib
import json
import logging
import typing
from abc import ABC, abstractmethod
from contextvars import ContextVar

import httpx
from diskcache import Cache
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from ..prompts.models import Message
from ..tracer import NoOpTracer, Tracer, TracerSpan
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError

DEFAULT_TEMPERATURE = 0
DEFAULT_CACHE_DIR = './llm_cache'

# Context variable to track token usage across a trace
token_usage_context: ContextVar[dict[str, int] | None] = ContextVar('token_usage', default=None)


def get_extraction_language_instruction(group_id: str | None = None) -> str:
    """Returns instruction for language extraction behavior.

    Override this function to customize language extraction:
    - Return empty string to disable multilingual instructions
    - Return custom instructions for specific language requirements
    - Use group_id to provide different instructions per group/partition

    Args:
        group_id: Optional partition identifier for the graph

    Returns:
        str: Language instruction to append to system messages
    """
    return '\n\nAny extracted information should be returned in the same language as it was written in.'


logger = logging.getLogger(__name__)


def is_server_or_retry_error(exception):
    if isinstance(exception, RateLimitError | json.decoder.JSONDecodeError):
        return True

    return (
        isinstance(exception, httpx.HTTPStatusError) and 500 <= exception.response.status_code < 600
    )


class LLMClient(ABC):
    def __init__(self, config: LLMConfig | None, cache: bool = False):
        if config is None:
            config = LLMConfig()

        self.config = config
        self.model = config.model
        self.small_model = config.small_model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.cache_enabled = cache
        self.cache_dir = None
        self.tracer: Tracer = NoOpTracer()

        # Only create the cache directory if caching is enabled
        if self.cache_enabled:
            self.cache_dir = Cache(DEFAULT_CACHE_DIR)

    def set_tracer(self, tracer: Tracer) -> None:
        """Set the tracer for this LLM client."""
        self.tracer = tracer

    def _clean_input(self, input: str) -> str:
        """Clean input string of invalid unicode and control characters.

        Args:
            input: Raw input string to be cleaned

        Returns:
            Cleaned string safe for LLM processing
        """
        # Clean any invalid Unicode
        cleaned = input.encode('utf-8', errors='ignore').decode('utf-8')

        # Remove zero-width characters and other invisible unicode
        zero_width = '\u200b\u200c\u200d\ufeff\u2060'
        for char in zero_width:
            cleaned = cleaned.replace(char, '')

        # Remove control characters except newlines, returns, and tabs
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')

        return cleaned

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_random_exponential(multiplier=10, min=5, max=120),
        retry=retry_if_exception(is_server_or_retry_error),
        after=lambda retry_state: logger.warning(
            f'Retrying {retry_state.fn.__name__ if retry_state.fn else "function"} after {retry_state.attempt_number} attempts...'
        )
        if retry_state.attempt_number > 1
        else None,
        reraise=True,
    )
    async def _generate_response_with_retry(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        try:
            return await self._generate_response(messages, response_model, max_tokens, model_size)
        except (httpx.HTTPStatusError, RateLimitError) as e:
            raise e

    @abstractmethod
    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        pass

    def _get_cache_key(self, messages: list[Message]) -> str:
        # Create a unique cache key based on the messages and model
        message_str = json.dumps([m.model_dump() for m in messages], sort_keys=True)
        key_str = f'{self.model}:{message_str}'
        return hashlib.md5(key_str.encode()).hexdigest()

    def _resolve_traced_model_name(self, model_size: ModelSize) -> str | None:
        """Best-effort model name for telemetry."""
        if model_size == ModelSize.small and self.small_model:
            return self.small_model
        return self.model

    def _safe_json_dumps(self, payload: dict[str, typing.Any]) -> str:
        """Serialize dictionaries for span attributes."""
        return json.dumps(payload, default=str, sort_keys=True)

    def _build_input_message_attributes(self, messages: list[Message]) -> dict[str, str]:
        """Flatten prompt messages into OpenInference span attributes."""
        attributes: dict[str, str] = {}
        for idx, message in enumerate(messages):
            base_key = f'llm.input_messages.{idx}.message'
            attributes[f'{base_key}.role'] = message.role
            attributes[f'{base_key}.content'] = message.content
        return attributes

    def _add_llm_input_attributes(
        self,
        span: TracerSpan,
        *,
        messages: list[Message],
        model_name: str | None,
        model_size: ModelSize,
        max_tokens: int,
        prompt_name: str | None,
        invocation_parameters: dict[str, typing.Any] | None = None,
        extra_attributes: dict[str, typing.Any] | None = None,
    ) -> None:
        """Attach OpenInference-compatible attributes for an LLM invocation."""
        attributes: dict[str, typing.Any] = {
            'openinference.span.kind': 'LLM',
            'openinference.model.provider': self._get_provider_type(),
            'llm.provider': self._get_provider_type(),
            'model.size': model_size.value,
            'max_tokens': max_tokens,
            'cache.enabled': self.cache_enabled,
        }
        if model_name:
            attributes['llm.model_name'] = model_name
            attributes['openinference.model.name'] = model_name
        if prompt_name:
            attributes['prompt.name'] = prompt_name
        if invocation_parameters:
            # Remove None values for cleaner payloads
            filtered_params = {k: v for k, v in invocation_parameters.items() if v is not None}
            if filtered_params:
                attributes['llm.invocation_parameters'] = self._safe_json_dumps(filtered_params)
        attributes.update(self._build_input_message_attributes(messages))
        if extra_attributes:
            attributes.update(extra_attributes)
        span.add_attributes(attributes)

    def _record_llm_output(self, span: TracerSpan, output_payload: typing.Any) -> None:
        """Attach assistant output to the tracing span."""
        if output_payload is None:
            return
        if isinstance(output_payload, str):
            serialized_output = output_payload
        else:
            try:
                serialized_output = json.dumps(output_payload, default=str)
            except Exception:
                serialized_output = str(output_payload)
        span.add_attributes(
            {
                'llm.output_messages.0.message.role': 'assistant',
                'llm.output_messages.0.message.content': serialized_output,
                'output.value': serialized_output,
            }
        )

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, typing.Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        if response_model is not None:
            serialized_model = json.dumps(response_model.model_json_schema())
            messages[
                -1
            ].content += (
                f'\n\nRespond with a JSON object in the following format:\n\n{serialized_model}'
            )

        # Add multilingual extraction instructions
        messages[0].content += get_extraction_language_instruction(group_id)

        for message in messages:
            message.content = self._clean_input(message.content)

        model_name = self._resolve_traced_model_name(model_size)

        invocation_parameters: dict[str, typing.Any] = {
            'temperature': self.temperature,
            'max_tokens': max_tokens,
            'model_size': model_size.value,
            'model': model_name,
        }
        if group_id:
            invocation_parameters['group_id'] = group_id
        if prompt_name:
            invocation_parameters['prompt_name'] = prompt_name
        if response_model is not None:
            invocation_parameters['response_model'] = response_model.__name__

        # Wrap entire operation in tracing span
        with self.tracer.start_span('llm') as span:
            self._add_llm_input_attributes(
                span,
                messages=messages,
                model_name=model_name,
                model_size=model_size,
                max_tokens=max_tokens,
                prompt_name=prompt_name,
                invocation_parameters=invocation_parameters,
            )

            # Check cache first
            if self.cache_enabled and self.cache_dir is not None:
                cache_key = self._get_cache_key(messages)
                cached_response = self.cache_dir.get(cache_key)
                if cached_response is not None:
                    logger.debug(f'Cache hit for {cache_key}')
                    span.add_attributes({'cache.hit': True})
                    self._record_llm_output(span, cached_response)
                    return cached_response

            span.add_attributes({'cache.hit': False})

            # Execute LLM call
            try:
                response = await self._generate_response_with_retry(
                    messages, response_model, max_tokens, model_size
                )
            except Exception as e:
                span.set_status('error', str(e))
                span.record_exception(e)
                raise

            # Cache response if enabled
            if self.cache_enabled and self.cache_dir is not None:
                cache_key = self._get_cache_key(messages)
                self.cache_dir.set(cache_key, response)

            self._record_llm_output(span, response)
            return response

    def _get_provider_type(self) -> str:
        """Get provider type from class name."""
        class_name = self.__class__.__name__.lower()
        if 'openai' in class_name:
            return 'openai'
        elif 'anthropic' in class_name:
            return 'anthropic'
        elif 'gemini' in class_name:
            return 'gemini'
        elif 'groq' in class_name:
            return 'groq'
        else:
            return 'unknown'

    def _get_failed_generation_log(self, messages: list[Message], output: str | None) -> str:
        """
        Log the full input messages, the raw output (if any), and the exception for debugging failed generations.
        """
        log = ''
        log += f'Input messages: {json.dumps([m.model_dump() for m in messages], indent=2)}\n'
        if output is not None:
            if len(output) > 4000:
                log += f'Raw output: {output[:2000]}... (truncated) ...{output[-2000:]}\n'
            else:
                log += f'Raw output: {output}\n'
        else:
            log += 'No raw output available'
        return log
