import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT


_rails: LLMRails | None = None
HANDLED_BOT_INTENTS = {
    "express greeting",
    "express farewell",
    "explain capabilities",
    "refuse off topic",
    "refuse jailbreak",
}


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses llama-3.1-8b-instant for fast intent classification at the gate —
    the heavier llama-3.3-70b-versatile is reserved for the RAG pipeline.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0
    ) 

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("🛡️ NeMo Guardrails initialised .")
    
    


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check") as span:
        result = _rails.generate(
            messages=[{"role": "user", "content": message}],
            options={"log": {"internal_events": True}},
        )

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        response = result.response
        content = response if isinstance(response, str) else "\n".join(
            item.get("content", "") for item in response if item.get("role") == "assistant"
        )
        events = result.log.internal_events if result.log else []
        bot_intents = [
            event.get("intent") for event in (events or [])
            if event.get("type") == "BotIntent"
        ]
        # Route using the executed dialog intent, independent of response wording.
        fired = any(intent in HANDLED_BOT_INTENTS for intent in bot_intents)
        if fired and not content.strip():
            raise RuntimeError("A guardrail handled the request but returned an empty response.")
        span.set_attribute("response", content)
        span.set_attribute("rail_fired", fired)
        span.set_attribute("bot_intents", bot_intents)
        logfire.info(
            "Guardrails result: matched={rail_fired}",
            rail_fired=fired,
            response=content,
        )

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
