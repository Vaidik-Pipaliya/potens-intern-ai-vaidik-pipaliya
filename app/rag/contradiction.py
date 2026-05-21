import json
import re
import logging
from langchain_core.prompts import PromptTemplate
from app.rag.vectorstore import get_vectorstore
from app.rag.llm import get_llm, extract_text_from_response

logger = logging.getLogger(__name__)

CONTRADICTION_TEMPLATE = """You are a meticulous document auditor and comparison engine.
Your task is to analyze content retrieved from two different documents (Document 1 and Document 2) regarding a specific topic, and determine if they contain any factual contradictions or inconsistencies.

Topic: {topic}

--- Document 1 ({doc1_id}) Context ---
{doc1_context}

--- Document 2 ({doc2_id}) Context ---
{doc2_context}

Instructions:
1. Examine the provided contexts for Document 1 and Document 2.
2. Determine if there is a factual contradiction, conflicting information, or inconsistency between the two documents regarding the topic: "{topic}".
3. If a contradiction is found, set "contradiction_found" to true, write a clear "reasoning" explaining the exact conflict, and extract the exact "evidence_doc1" and "evidence_doc2" from the contexts.
4. If no contradiction is found (e.g., they agree, or one/both do not mention the topic), set "contradiction_found" to false, explain in "reasoning" that no contradiction was found, and extract whatever relevant text exists (if any) as evidence.
5. You MUST return ONLY a valid JSON object matching the following structure. Do NOT wrap the JSON in markdown formatting (no ```json).

JSON Schema:
{{
  "contradiction_found": boolean,
  "reasoning": "string",
  "evidence_doc1": "string",
  "evidence_doc2": "string"
}}
"""

def analyze_contradiction(doc1_id: str, doc2_id: str, topic: str) -> dict:
    """
    Compares two documents on a topic to find factual contradictions.
    """
    db = get_vectorstore()
    
    # Isolated retrieval per file — avoids mixing v1/v2 policy text in one context window.
    try:
        docs1 = db.similarity_search(topic, k=4, filter={"source": doc1_id})
    except Exception as e:
        logger.error(f"Error searching doc1 {doc1_id}: {str(e)}")
        docs1 = []
        
    # Retrieve relevant chunks for Document 2
    try:
        docs2 = db.similarity_search(topic, k=4, filter={"source": doc2_id})
    except Exception as e:
        logger.error(f"Error searching doc2 {doc2_id}: {str(e)}")
        docs2 = []

    # Format context strings
    doc1_context = "\n\n".join([f"[Chunk {idx+1}]: {doc.page_content}" for idx, doc in enumerate(docs1)]) if docs1 else "No information found in this document."
    doc2_context = "\n\n".join([f"[Chunk {idx+1}]: {doc.page_content}" for idx, doc in enumerate(docs2)]) if docs2 else "No information found in this document."
    
    # Instantiate LLM and Prompt
    llm = get_llm()
    prompt = PromptTemplate.from_template(CONTRADICTION_TEMPLATE)
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            "topic": topic,
            "doc1_id": doc1_id,
            "doc2_id": doc2_id,
            "doc1_context": doc1_context,
            "doc2_context": doc2_context
        })
        
        content = extract_text_from_response(response.content).strip()
        
        # Groq/Llama often wraps JSON in markdown fences despite the prompt — strip before json.loads.
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n", "", content)
            content = re.sub(r"\n```$", "", content)
            content = content.strip()
            
        data = json.loads(content)
        
        return {
            "contradiction_found": bool(data.get("contradiction_found", False)),
            "reasoning": str(data.get("reasoning", "No analysis returned.")),
            "evidence_doc1": str(data.get("evidence_doc1", "")),
            "evidence_doc2": str(data.get("evidence_doc2", ""))
        }
    except Exception as e:
        logger.error(f"Error analyzing contradiction: {str(e)}")
        return {
            "contradiction_found": False,
            "reasoning": f"Failed to perform contradiction analysis: {str(e)}",
            "evidence_doc1": "",
            "evidence_doc2": ""
        }
