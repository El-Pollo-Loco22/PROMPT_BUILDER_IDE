import sys
import os

# Add project root to path so src imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_imports():
    """Verify all core dependencies are installed."""
    import langgraph
    import ollama
    import streamlit
    import pydantic
    import pydantic_settings


def test_config_loads():
    """Verify Settings loads with correct defaults."""
    from src.config import Settings

    s = Settings()
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.default_model == "llama3:8b"
    assert s.app_name == "Agentic Prompt IDE"


def test_prompt_schema_roundtrip():
    """Verify PromptDocument serializes and deserializes correctly."""
    from src.schemas.prompt import PromptBlock, PromptDocument

    doc = PromptDocument(
        blocks=[
            PromptBlock(role="system", content="You are a helpful assistant."),
            PromptBlock(role="user", content="Write a haiku.", metadata={"tag": "test"}),
        ]
    )

    # Serialize to dict and back
    data = doc.model_dump()
    restored = PromptDocument.model_validate(data)

    assert len(restored.blocks) == 2
    assert restored.blocks[0].role == "system"
    assert restored.blocks[1].metadata == {"tag": "test"}
    assert restored.version == "1.0.0"
