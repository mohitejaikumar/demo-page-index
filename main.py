# simple_pageindex_rag_refactored.py
# Python 3.10+
# pip install pageindex openai requests

import os
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any

import requests
from pageindex import PageIndexClient
import pageindex.utils as utils
from openai import AsyncOpenAI



PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PAGEINDEX_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing API keys. Set PAGEINDEX_API_KEY and OPENAI_API_KEY")

PDF_URL = "https://arxiv.org/pdf/1706.03762"
DOWNLOAD_DIR = Path("./data")
DOWNLOAD_DIR.mkdir(exist_ok=True)

MODEL_NAME = "gpt-4.1"
POLL_INTERVAL = 5
MAX_POLL_ATTEMPTS = 60



piClient = PageIndexClient(api_key=PAGEINDEX_API_KEY)
llmClient = AsyncOpenAI(api_key=OPENAI_API_KEY)



def downloadPdf(url: str, downloadDir: Path) -> Path:
    """Download PDF if not already present."""
    pdfPath = downloadDir / url.split("/")[-1]

    if pdfPath.exists():
        print(f"PDF already exists: {pdfPath}")
        return pdfPath

    print(f"Downloading {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    pdfPath.write_bytes(response.content)
    print(f"Saved to {pdfPath}")
    return pdfPath


def submitDocument(pdfPath: Path) -> str:
    """Submit document to PageIndex."""
    print("Submitting document to PageIndex...")
    response = piClient.submit_document(str(pdfPath))
    docId = response.get("doc_id") or response.get("id") or response
    print(f"Submitted. doc_id: {docId}")
    return docId


def waitForTreeGeneration(docId: str) -> None:
    """Poll PageIndex until retrieval tree is ready."""
    print("Waiting for tree generation...")

    for attempt in range(MAX_POLL_ATTEMPTS):
        if piClient.is_retrieval_ready(docId):
            print("Tree is ready.")
            return

        print(f"Not ready... ({attempt + 1}/{MAX_POLL_ATTEMPTS})")
        time.sleep(POLL_INTERVAL)

    raise TimeoutError("Timed out waiting for PageIndex tree generation")


def fetchTree(docId: str) -> List[Dict[str, Any]]:
    """Fetch tree with node summaries."""
    response = piClient.get_tree(docId, node_summary=True)
    tree = response.get("result") if isinstance(response, dict) else response
    print(f"Fetched tree. Top-level nodes: {len(tree)}")
    return tree


async def callLlm(prompt: str, temperature: float = 0.0) -> str:
    """Call OpenAI LLM asynchronously."""
    response = await llmClient.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()



async def searchRelevantNodes(tree: List[Dict], query: str) -> Dict:
    """Use LLM to select relevant nodes from tree."""
    treeWithoutText = utils.remove_fields(tree.copy(), fields=["text"])

    prompt = f"""
You are given a question and a tree structure of a document.
Each node contains node id, title, and summary.

Find nodes likely to contain the answer.

Question: {query}

Document tree:
{json.dumps(treeWithoutText, indent=2)}

Return JSON:
{{
  "thinking": "...",
  "node_list": ["node_id_1"]
}}
Only return JSON.
"""

    resultText = await callLlm(prompt)

    try:
        return json.loads(resultText)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON from LLM:\n{resultText}")


def extractNodeText(nodeMap: Dict, nodeIds: List[str]) -> str:
    """Extract and combine text from selected nodes."""
    retrievedTexts = []

    for nodeId in nodeIds:
        node = nodeMap.get(nodeId)
        if not node:
            continue

        nodeText = node.get("text") or ""
        if isinstance(nodeText, list):
            nodeText = "\n\n".join(nodeText)

        retrievedTexts.append(
            f"--- Node {nodeId}: {node.get('title')} ---\n{nodeText}"
        )

    return "\n\n".join(retrievedTexts) or "No context retrieved."


async def generateFinalAnswer(query: str, context: str) -> str:
    """Generate grounded answer from retrieved context."""
    prompt = f"""
Answer the question using ONLY the context below.

Question:
{query}

Context:
{context}

Provide a concise grounded answer and include node IDs used.
"""

    return await callLlm(prompt)


async def runPipeline(query: str) -> None:
    pdfPath = downloadPdf(PDF_URL, DOWNLOAD_DIR)

    docId = submitDocument(pdfPath)
    waitForTreeGeneration(docId)

    tree = fetchTree(docId)
    nodeMap = utils.create_node_mapping(tree)

    print("\nSearching relevant nodes...")
    searchResult = await searchRelevantNodes(tree, query)

    print("\n=== LLM Reasoning (truncated) ===")
    print(searchResult.get("thinking", "")[:500])

    nodeIds = searchResult.get("node_list", [])
    print("Selected node IDs:", nodeIds)

    context = extractNodeText(nodeMap, nodeIds)

    print("\nGenerating final answer...")
    finalAnswer = await generateFinalAnswer(query, context)

    print("\n=== Final Answer ===\n")
    print(finalAnswer)
    

if __name__ == "__main__":
    asyncio.run(
        runPipeline("What is Self-Attention?")
    )