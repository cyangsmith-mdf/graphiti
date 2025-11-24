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

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opentelemetry.trace import Span, StatusCode

try:
    from opentelemetry.trace import Span, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


class TracerSpan(ABC):
    """Abstract base class for tracer spans."""

    @abstractmethod
    def add_attributes(self, attributes: dict[str, Any]) -> None:
        """Add attributes to the span."""
        pass

    @abstractmethod
    def set_status(self, status: str, description: str | None = None) -> None:
        """Set the status of the span."""
        pass

    @abstractmethod
    def record_exception(self, exception: Exception) -> None:
        """Record an exception in the span."""
        pass


class Tracer(ABC):
    """Abstract base class for tracers."""

    @abstractmethod
    def start_span(self, name: str, skip_prefix: bool = False) -> AbstractContextManager[TracerSpan]:
        """Start a new span with the given name."""
        pass


class NoOpSpan(TracerSpan):
    """No-op span implementation that does nothing."""

    def add_attributes(self, attributes: dict[str, Any]) -> None:
        pass

    def set_status(self, status: str, description: str | None = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass


class NoOpTracer(Tracer):
    """No-op tracer implementation that does nothing."""

    @contextmanager
    def start_span(self, name: str, skip_prefix: bool = False) -> Generator[NoOpSpan, None, None]:
        """Return a no-op span."""
        yield NoOpSpan()


class OpenTelemetrySpan(TracerSpan):
    """Wrapper for OpenTelemetry span."""

    def __init__(self, span: 'Span'):
        self._span = span

    def add_attributes(self, attributes: dict[str, Any]) -> None:
        """Add attributes to the OpenTelemetry span."""
        try:
            # Filter out None values and convert all values to appropriate types
            filtered_attrs = {}
            for key, value in attributes.items():
                if value is not None:
                    # Convert to string if not a primitive type
                    if isinstance(value, str | int | float | bool):
                        filtered_attrs[key] = value
                    else:
                        filtered_attrs[key] = str(value)

            if filtered_attrs:
                self._span.set_attributes(filtered_attrs)
        except Exception:
            # Silently ignore tracing errors
            pass

    def set_status(self, status: str, description: str | None = None) -> None:
        """Set the status of the OpenTelemetry span."""
        try:
            if OTEL_AVAILABLE:
                normalized = status.strip().lower()
                if normalized == 'error':
                    self._span.set_status(StatusCode.ERROR, description)
                elif normalized == 'ok':
                    self._span.set_status(StatusCode.OK, description)
        except Exception:
            # Silently ignore tracing errors
            pass

    def record_exception(self, exception: Exception) -> None:
        """Record an exception in the OpenTelemetry span."""
        with suppress(Exception):
            self._span.record_exception(exception)


class OpenTelemetryTracer(Tracer):
    """Wrapper for OpenTelemetry tracer with configurable span name prefix."""

    def __init__(self, tracer: Any, span_prefix: str = 'graphiti'):
        """
        Initialize the OpenTelemetry tracer wrapper.

        Parameters
        ----------
        tracer : opentelemetry.trace.Tracer
            The OpenTelemetry tracer instance.
        span_prefix : str, optional
            Prefix to prepend to all span names. Defaults to 'graphiti'.
        """
        if not OTEL_AVAILABLE:
            raise ImportError(
                'OpenTelemetry is not installed. Install it with: pip install opentelemetry-api'
            )
        self._tracer = tracer
        self._span_prefix = span_prefix.rstrip('.')

    def start_span(self, name: str, skip_prefix: bool = False) -> AbstractContextManager[TracerSpan]:
        """Start a new OpenTelemetry span with the configured prefix."""
        if skip_prefix or not self._span_prefix:
            full_name = name
        else:
            full_name = f'{self._span_prefix}.{name}'

        return _OpenTelemetrySpanContext(self._tracer, full_name)


class _OpenTelemetrySpanContext(AbstractContextManager[TracerSpan]):
    """Context manager that wraps the OpenTelemetry span lifecycle safely."""

    def __init__(self, tracer: Any, span_name: str):
        self._tracer = tracer
        self._span_name = span_name
        self._cm: AbstractContextManager['Span'] | None = None
        self._span_wrapper: TracerSpan = NoOpSpan()

    def __enter__(self) -> TracerSpan:
        try:
            self._cm = self._tracer.start_as_current_span(self._span_name)
            span = self._cm.__enter__()
        except Exception:
            self._cm = None
            self._span_wrapper = NoOpSpan()
            return self._span_wrapper

        self._span_wrapper = OpenTelemetrySpan(span)
        return self._span_wrapper

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._cm is None:
            return False

        with suppress(Exception):
            return bool(self._cm.__exit__(exc_type, exc, tb))

        return False


def create_tracer(otel_tracer: Any | None = None, span_prefix: str = 'graphiti') -> Tracer:
    """
    Create a tracer instance.

    Parameters
    ----------
    otel_tracer : opentelemetry.trace.Tracer | None, optional
        An OpenTelemetry tracer instance. If None, a no-op tracer is returned.
    span_prefix : str, optional
        Prefix to prepend to all span names. Defaults to 'graphiti'.

    Returns
    -------
    Tracer
        A tracer instance (either OpenTelemetryTracer or NoOpTracer).

    Examples
    --------
    Using with OpenTelemetry:

    >>> from opentelemetry import trace
    >>> otel_tracer = trace.get_tracer(__name__)
    >>> tracer = create_tracer(otel_tracer, span_prefix='myapp.graphiti')

    Using no-op tracer:

    >>> tracer = create_tracer()  # Returns NoOpTracer
    """
    if otel_tracer is None:
        return NoOpTracer()

    if not OTEL_AVAILABLE:
        return NoOpTracer()

    return OpenTelemetryTracer(otel_tracer, span_prefix)
