"""Project-specific exceptions.

Expected envelope overflow is represented as data, not raised as an exception.
Exceptions in this module are reserved for invalid definitions and inputs.
"""


class IntentLeakageError(Exception):
    """Base class for package errors."""


class SchemaValidationError(IntentLeakageError, ValueError):
    """Raised when a paper-level data object violates its declared schema."""


class PartitionError(IntentLeakageError, ValueError):
    """Raised when shaping groups are not a complete, disjoint partition."""


class EnvelopeDefinitionError(IntentLeakageError, ValueError):
    """Raised when an envelope is incomplete or internally inconsistent."""


class UnsafeDecoyError(EnvelopeDefinitionError):
    """Raised when a decoy is not authenticated, authorized, and read-only."""
