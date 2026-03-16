"""
Clinical Context Agent
-----------------------
Takes image findings and retrieves relevant medical context
using Qdrant vector search + fastembed.

Modes:
  - qdrant: real vector search over medical literature (default)
  - mock:   fallback if Qdrant unavailable
"""
import logging
import os
from dataclasses import dataclass, field

from agents.image_analysis import ImageFindings

logger = logging.getLogger(__name__)


@dataclass
class ClinicalContext:
    anonymized_id:        str
    relevant_conditions:  list[str]
    differential_diagnosis: list[str]
    recommended_followup: list[str]
    urgency_level:        str
    context_sources:      list[str]


# fallback knowledge base if Qdrant is down
FALLBACK_KNOWLEDGE = {
    "consolidation":   {"conditions": ["Pneumonia", "Pulmonary edema"],         "differential": ["Bacterial pneumonia", "Viral pneumonia"], "followup": ["Repeat CXR in 6-8 weeks", "Sputum culture"], "urgency": "urgent"},
    "pleural effusion":{"conditions": ["Heart failure", "Malignancy"],           "differential": ["Transudative", "Exudative"],             "followup": ["Echocardiogram", "Thoracentesis"],          "urgency": "urgent"},
    "pneumothorax":    {"conditions": ["Spontaneous pneumothorax"],              "differential": ["Primary", "Secondary"],                   "followup": ["Immediate assessment", "Chest tube"],       "urgency": "emergent"},
    "normal":          {"conditions": ["No acute disease"],                      "differential": ["Normal variant"],                         "followup": ["Routine follow-up"],                        "urgency": "routine"},
}


def _build_query(image_findings: ImageFindings) -> str:
    """Build a search query from image findings."""
    parts = []
    if image_findings.impression:
        parts.append(image_findings.impression)
    if image_findings.findings:
        parts.extend(image_findings.findings[:3])  # top 3 findings
    if image_findings.modality:
        parts.append(f"{image_findings.modality} imaging")
    return " ".join(parts)


def _qdrant_context(image_findings: ImageFindings) -> ClinicalContext:
    """Real RAG via Qdrant vector search."""
    from qdrant_client import QdrantClient

    qdrant_url        = os.environ.get("QDRANT_URL", "http://localhost:6333")
    collection_name   = "medical_literature"

    client = QdrantClient(url=qdrant_url)

    # check collection exists
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        logger.warning("Collection '%s' not found — run ingest_medical_knowledge.py first", collection_name)
        raise RuntimeError("Qdrant collection not found")

    query = _build_query(image_findings)
    logger.info(
        "Mode: QDRANT RAG | anon_id=%s | query='%s...'",
        image_findings.anonymized_id,
        query[:60],
    )

    # search — fastembed encodes the query automatically
    results = client.query(
        collection_name=collection_name,
        query_text=query,
        limit=3,
    )

    if not results:
        logger.warning("No Qdrant results — falling back to mock")
        raise RuntimeError("No results returned")

    # aggregate results
    all_conditions  = []
    all_differential = []
    all_followup    = []
    sources         = []
    urgency         = "routine"

    urgency_rank = {"routine": 0, "urgent": 1, "emergent": 2}

    for hit in results:
        payload = hit.metadata if hasattr(hit, "metadata") else {}

        conditions = payload.get("conditions", [])
        followup   = payload.get("followup", [])
        finding    = payload.get("finding", "unknown")
        hit_urgency = payload.get("urgency", "routine")
        source_id  = payload.get("id", "unknown")

        all_conditions.extend(conditions)
        all_followup.extend(followup)
        sources.append(f"qdrant:{source_id}:{finding}")

        # take highest urgency across all results
        if urgency_rank.get(hit_urgency, 0) > urgency_rank.get(urgency, 0):
            urgency = hit_urgency

        # use top result's finding as differential basis
        if not all_differential:
            all_differential = conditions[:2]

    # deduplicate
    all_conditions  = list(dict.fromkeys(all_conditions))
    all_followup    = list(dict.fromkeys(all_followup))
    all_differential = list(dict.fromkeys(all_differential))

    logger.info(
        "RAG retrieved %d results | urgency=%s | conditions=%s",
        len(results), urgency, all_conditions[:3],
    )

    return ClinicalContext(
        anonymized_id=image_findings.anonymized_id,
        relevant_conditions=all_conditions[:5],
        differential_diagnosis=all_differential[:3],
        recommended_followup=all_followup[:4],
        urgency_level=urgency,
        context_sources=sources,
    )


def _mock_context(image_findings: ImageFindings) -> ClinicalContext:
    """Fallback mock when Qdrant is unavailable."""
    logger.info("Mode: MOCK clinical context | anon_id=%s", image_findings.anonymized_id)

    combined = " ".join(image_findings.findings + [image_findings.impression]).lower()
    match = FALLBACK_KNOWLEDGE["normal"]

    for keyword, data in FALLBACK_KNOWLEDGE.items():
        if keyword in combined:
            match = data
            break

    return ClinicalContext(
        anonymized_id=image_findings.anonymized_id,
        relevant_conditions=match["conditions"],
        differential_diagnosis=match["differential"],
        recommended_followup=match["followup"],
        urgency_level=match["urgency"],
        context_sources=["fallback_knowledge_base"],
    )


def run(image_findings: ImageFindings) -> ClinicalContext:
    """
    Run clinical context retrieval.
    Tries Qdrant first, falls back to mock if unavailable.
    """
    try:
        return _qdrant_context(image_findings)
    except Exception as e:
        logger.warning("Qdrant RAG failed (%s) — using fallback", e)
        return _mock_context(image_findings)
