"""
eval/deepeval_compat.py — Compatibility shim for DeepEval 2.5.3 on Python 3.13.

Problem: DeepEval 2.5.3 imports `langchain.schema.HumanMessage` which was
removed when LangChain was restructured into langchain-core 1.x.
This module patches sys.modules BEFORE importing deepeval so the import
chain doesn't blow up.

Also configures OPENAI_API_KEY / OPENAI_BASE_URL for DeepEval's internal LLM
from whichever credential is available (GITHUB_TOKEN, OPENROUTER_API_KEY,
or OPENAI_API_KEY), so DeepEval metrics actually make LLM calls rather than
falling back to stubs.

Must be imported BEFORE any deepeval import anywhere in the eval package.
Usage:
    from eval.deepeval_compat import (
        GEval, FaithfulnessMetric, AnswerRelevancyMetric,
        LLMTestCase, LLMTestCaseParams,
        DEEPEVAL_AVAILABLE,
    )
    if not DEEPEVAL_AVAILABLE:
        # fall back to heuristics
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

# Load .env so GITHUB_TOKEN / OPENROUTER_API_KEY / OPENAI_API_KEY are available
try:
    from dotenv import load_dotenv
    _repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(_repo_root / ".env", override=False)
except ImportError:
    pass

DEEPEVAL_AVAILABLE = False
GEval = None
FaithfulnessMetric = None
AnswerRelevancyMetric = None
LLMTestCase = None
LLMTestCaseParams = None


def _patch_langchain_schema() -> None:
    """
    Inject a stub `langchain.schema` module that supplies the three message
    classes DeepEval 2.5.3 imports but that were removed from LangChain 1.x.
    """
    if "langchain.schema" in sys.modules:
        return
    stub = types.ModuleType("langchain.schema")
    stub.HumanMessage = type("HumanMessage", (), {})
    stub.AIMessage = type("AIMessage", (), {})
    stub.SystemMessage = type("SystemMessage", (), {})
    stub_messages = types.ModuleType("langchain.schema.messages")
    stub_messages.HumanMessage = stub.HumanMessage
    stub_messages.AIMessage = stub.AIMessage
    stub_messages.SystemMessage = stub.SystemMessage
    sys.modules["langchain.schema"] = stub
    sys.modules["langchain.schema.messages"] = stub_messages


def _configure_deepeval_api() -> None:
    """
    Configure the OPENAI_API_KEY + OPENAI_BASE_URL that DeepEval's internal
    LLM client reads, using whichever key is available in this environment.

    Priority:
      1. OPENAI_API_KEY already set  → leave unchanged
      2. GITHUB_TOKEN set            → use it + GitHub Models base URL
      3. OPENROUTER_API_KEY set      → use it + OpenRouter base URL
    """
    if os.environ.get("OPENAI_API_KEY"):
        return  # already configured

    github_token = os.environ.get("GITHUB_TOKEN", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    if github_token:
        os.environ["OPENAI_API_KEY"] = github_token
        if not os.environ.get("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = "https://models.inference.ai.azure.com"
    elif openrouter_key:
        os.environ["OPENAI_API_KEY"] = openrouter_key
        if not os.environ.get("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"


try:
    _patch_langchain_schema()
    _configure_deepeval_api()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import deepeval  # noqa: F401 (side-effect import to trigger the patch)
        from deepeval.metrics import GEval as _GEval
        from deepeval.metrics import FaithfulnessMetric as _FaithfulnessMetric
        from deepeval.metrics import AnswerRelevancyMetric as _AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase as _LLMTestCase
        from deepeval.test_case import LLMTestCaseParams as _LLMTestCaseParams

    GEval = _GEval
    FaithfulnessMetric = _FaithfulnessMetric
    AnswerRelevancyMetric = _AnswerRelevancyMetric
    LLMTestCase = _LLMTestCase
    LLMTestCaseParams = _LLMTestCaseParams
    DEEPEVAL_AVAILABLE = True
    print("[deepeval_compat] DeepEval 2.5.3 loaded successfully (langchain.schema patched).", flush=True)

except Exception as _exc:
    print(f"[deepeval_compat] DeepEval unavailable: {_exc}. All metrics will use heuristic fallback.", flush=True)
    DEEPEVAL_AVAILABLE = False
