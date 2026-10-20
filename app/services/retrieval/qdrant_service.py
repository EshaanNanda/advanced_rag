import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.services.retrieval.embedding import embed_query


# Initialize Qdrant Client
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)


class RetrievalError(RuntimeError):
    """A failed retrieval operation, distinct from a search with no matches."""


def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface.
    """
    try:
        with logfire.span("Generate query embedding"):
            query_vector = embed_query(query)
    except Exception as error:
        logfire.exception(
            "Retrieval failed during embedding generation",
            stage="embedding",
            error_type=type(error).__name__,
        )
        raise RetrievalError("Embedding generation failed; document search was not run.") from error

    try:
        # Using query_points - the modern standard for Qdrant
        with logfire.span("Search Qdrant collection", collection=settings.QDRANT_COLLECTION):
            response = client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )

        results = []
        for res in response.points:
            results.append({
                "content": res.payload.get("text", ""),
                "source": res.payload.get("source", "Unknown"),
                "score": res.score
            })
        
        return results
    except Exception as error:
        logfire.exception(
            "Retrieval failed during Qdrant search",
            stage="qdrant",
            error_type=type(error).__name__,
        )
        raise RetrievalError("Qdrant document search failed.") from error
