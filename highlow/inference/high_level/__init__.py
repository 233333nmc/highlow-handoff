"""Public, framework-free high-level planner/verifier inference API."""

from .api import create_app, high_level_request_from_dict
from .backend import HighLevelBackend
from .bootstrap import create_sglang_app, create_sglang_app_from_env
from .contracts import (
    BackendInferenceRequest,
    BackendInferenceResponse,
    GenerationOptions,
    HealthState,
    HealthStatus,
    HighLevelCapabilities,
    HighLevelFrame,
    HighLevelOperation,
    HighLevelOutput,
    HighLevelRequest,
    HighLevelResponse,
    PlannerOutput,
    VerifierDecision,
    VerifierOutput,
    VerifierStatus,
)
from .errors import (
    BackendProtocolError,
    BackendTransportError,
    HighLevelError,
    HighLevelValidationError,
    UnsupportedVersionError,
)
from .prompts import (
    PLANNER_JSON_SCHEMA_V1,
    PLANNER_PROMPT_V1,
    PLANNER_SCHEMA_V1,
    VERIFIER_JSON_SCHEMA_V1,
    VERIFIER_PROMPT_V1,
    VERIFIER_SCHEMA_V1,
    PromptSchemaRegistry,
    PromptSchemaSpec,
    default_prompt_schema_registry,
)
from .service import HighLevelService
from .sglang import (
    HttpRequest,
    HttpResponse,
    HttpTransport,
    SGLangOpenAIBackend,
    TransportCallable,
    UrllibTransport,
)

SGLangBackend = SGLangOpenAIBackend

__all__ = [
    "BackendInferenceRequest",
    "BackendInferenceResponse",
    "BackendProtocolError",
    "BackendTransportError",
    "GenerationOptions",
    "HealthState",
    "HealthStatus",
    "HighLevelBackend",
    "HighLevelCapabilities",
    "HighLevelError",
    "HighLevelFrame",
    "HighLevelOperation",
    "HighLevelOutput",
    "HighLevelRequest",
    "HighLevelResponse",
    "HighLevelService",
    "HighLevelValidationError",
    "HttpRequest",
    "HttpResponse",
    "HttpTransport",
    "PLANNER_JSON_SCHEMA_V1",
    "PLANNER_PROMPT_V1",
    "PLANNER_SCHEMA_V1",
    "PlannerOutput",
    "PromptSchemaRegistry",
    "PromptSchemaSpec",
    "SGLangBackend",
    "SGLangOpenAIBackend",
    "TransportCallable",
    "UnsupportedVersionError",
    "UrllibTransport",
    "VERIFIER_JSON_SCHEMA_V1",
    "VERIFIER_PROMPT_V1",
    "VERIFIER_SCHEMA_V1",
    "VerifierDecision",
    "VerifierOutput",
    "VerifierStatus",
    "default_prompt_schema_registry",
    "create_app",
    "create_sglang_app",
    "create_sglang_app_from_env",
    "high_level_request_from_dict",
]
