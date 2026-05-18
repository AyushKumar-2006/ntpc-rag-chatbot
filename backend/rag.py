# backend/rag.py
import os
import asyncio
import chromadb
import ollama
from embedder import embed_text

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHROMA_PATH     = "/Users/ayushkumar/ntpc-rag-chatbot/backend/chroma"
COLLECTION_NAME = "ntpc_content"
MODEL           = "llama3.2"
TOP_K           = 15
DISTANCE_CUTOFF = 1.2
MAX_CONTEXT_CHARS = 12000

YEAR_TO_FILE = {
    "2017-18": "2017-18.pdf",
    "2018-19": "annual-report-2018-19.pdf",
    "2019-20": "44-final-NTPC-AR-30082020.pdf",
    "2020-21": "NTPC_Annual Report_20-21.pdf",
    "2021-22": "Annual-Report-2021-22.pdf",
    "2022-23": "47th Annual Report of NTPC 2022-2023 New.pdf",
    "2023-24": "Annual Report 2023-24.pdf",
    "2024-25": "Annual Report 2024-25_1.pdf",
}
LATEST_YEAR = "2024-25"

QUERY_EXPANSIONS: dict[str, str] = {
    "brighter plan":    "Brighter Plan 2032 renewable energy strategy decarbonization net zero",
    "vision":           "NTPC vision mission strategy long term plan",
    "strategy":         "NTPC strategy plan target goals objectives",
    "green hydrogen":   "green hydrogen projects pilot NTPC electrolyzer",
    "renewable":        "renewable energy capacity target solar wind 2032 GW",
    "solar":            "solar power capacity MW GW projects commissioned",
    "wind":             "wind energy capacity projects commissioned MW",
    "hydro":            "hydro power hydroelectric capacity stations",
    "nuclear":          "nuclear energy power NTPC atomic",
    "coal":             "coal based thermal power stations capacity generation",
    "dividend":         "dividend per share interim final equity shareholder payout",
    "total income":     "total income revenue from operations turnover crore",
    "profit":           "profit after tax PAT net profit crore",
    "revenue":          "revenue total income operations crore financial",
    "debt":             "debt borrowings long term short term crore ratio",
    "credit rating":    "credit rating CRISIL ICRA AAA debt rating",
    "capex":            "capital expenditure capex investment crore",
    "pat":              "profit after tax PAT net profit crore earnings",
    "ebitda":           "EBITDA operating profit earnings before interest tax depreciation",
    "plf":              "Plant Load Factor PLF coal stations national average generation",
    "capacity":         "installed capacity commercial MW GW stations total",
    "generation":       "gross generation units BU billion units sent out",
    "employees":        "number of employees workforce manpower human resource",
    "stations":         "power stations plants NTPC locations units",
    "csr":              "CSR corporate social responsibility community development education health",
    "esg":              "ESG environment social governance sustainability reporting",
    "carbon":           "carbon emission CO2 greenhouse gas climate change",
    "water":            "water consumption stewardship management conservation",
    "environment":      "environment sustainability green emission pollution",
    "subsidiaries":     "subsidiaries joint ventures NTPC group companies list",
    "board":            "board of directors chairman CMD management executive",
    "awards":           "awards recognition achievements honours certifications",
    "related party":    "related party transactions subsidiaries associates",
    "auditor":          "statutory auditor CA chartered accountant audit firm",
    "director":         "board of directors independent executive non-executive members",
    "chairman":         "chairman CMD MD CEO management board NTPC",
    "shareholder":      "shareholders equity shares AGM annual general meeting",
}

SYSTEM_PROMPT = """You are a strict document retrieval assistant for NTPC Limited.

ABSOLUTE RULES:
1. Use ONLY the context provided — ZERO outside knowledge
2. If not in context → say EXACTLY: "This information is not available in the provided NTPC reports."
3. NEVER say 'However', 'based on general knowledge', 'I am unable to verify' — FORBIDDEN
4. NEVER guess — only state what is written in context
5. Always cite exact report name and page number
6. State numbers directly — never say 'not explicitly stated'
7. Minimum 3-4 sentences with all numbers"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHROMADB INIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def init_collection():
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col    = client.get_collection(name=COLLECTION_NAME)
        print(f"✅ ChromaDB loaded. Documents: {col.count()}")
        return col
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return None

collection = init_collection()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_years(question: str) -> list[str]:
    return [y for y in YEAR_TO_FILE if y in question]

def is_current_question(question: str) -> bool:
    return any(kw in question.lower() for kw in
               ["current", "latest", "now", "today", "recent", "abhi", "present"])

def expand_query(question: str) -> str:
    q_lower    = question.lower()
    expansions = [exp for key, exp in QUERY_EXPANSIONS.items() if key in q_lower]
    if expansions:
        return f"{question} {' '.join(expansions)}"
    return question

def generate_sub_queries(question: str) -> list[str]:
    q     = question.strip()
    q_exp = expand_query(q)
    stopwords = {"what","is","are","the","of","in","for","a","an","and","or",
                 "was","were","has","have","how","many","much","ntpc","about",
                 "tell","me","give","from","to","its","their","which","who"}
    keywords = " ".join(
        w for w in q.lower().split()
        if w not in stopwords and len(w) > 3
    )
    return list(dict.fromkeys([q, q_exp, keywords]))

def query_chroma_single(embedding, year_filter=None, top_k=TOP_K):
    kwargs = {
        "query_embeddings": [embedding],
        "n_results":        min(top_k, collection.count()),
        "include":          ["documents", "metadatas", "distances"],
    }
    if year_filter and year_filter in YEAR_TO_FILE:
        kwargs["where"] = {"source": YEAR_TO_FILE[year_filter]}
    return collection.query(**kwargs)

def multi_query_chroma(question: str, year_filter=None) -> tuple[list, list, list]:
    sub_queries = generate_sub_queries(question)
    seen_chunks: dict[str, tuple[str, dict, float]] = {}

    for sq in sub_queries:
        try:
            emb   = embed_text(sq)
            res   = query_chroma_single(emb, year_filter=year_filter)
            docs  = res["documents"][0]
            metas = res["metadatas"][0]
            dists = res["distances"][0]
            for chunk, meta, dist in zip(docs, metas, dists):
                key = chunk[:120]
                if key not in seen_chunks or dist < seen_chunks[key][2]:
                    seen_chunks[key] = (chunk, meta, dist)
        except Exception:
            pass

    ranked = sorted(seen_chunks.values(), key=lambda x: x[2])
    ranked = [r for r in ranked if r[2] < DISTANCE_CUTOFF]

    if not ranked:
        ranked = sorted(seen_chunks.values(), key=lambda x: x[2])[:TOP_K]

    return (
        [r[0] for r in ranked],
        [r[1] for r in ranked],
        [r[2] for r in ranked],
    )

async def fetch_multi_year_parallel(question: str, years: list[str]):
    loop = asyncio.get_event_loop()

    async def fetch_one(yr):
        return await loop.run_in_executor(
            None,
            lambda: multi_query_chroma(question, year_filter=yr)
        )

    results = await asyncio.gather(*[fetch_one(yr) for yr in years])

    all_chunks, all_metas, all_dists = [], [], []
    for chunks, metas, dists in results:
        all_chunks += chunks
        all_metas  += metas
        all_dists  += dists

    combined = sorted(zip(all_chunks, all_metas, all_dists), key=lambda x: x[2])
    seen, final = set(), []
    for chunk, meta, dist in combined:
        key = chunk[:120]
        if key not in seen:
            seen.add(key)
            final.append((chunk, meta, dist))

    return (
        [x[0] for x in final],
        [x[1] for x in final],
        [x[2] for x in final],
    )

def build_context(chunks: list[str], metadatas: list[dict]) -> str:
    parts, total = [], 0
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        source = meta.get("source", meta.get("title", "Unknown"))
        page   = meta.get("page", "")
        label  = f"{source} | Page {page}" if page else source
        entry  = f"[Source {i+1}: {label}]\n{chunk}"
        if total + len(entry) > MAX_CONTEXT_CHARS:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n---\n\n".join(parts)

def build_sources(metadatas: list[dict], distances: list[float]) -> list[dict]:
    sources, seen = [], set()
    for meta, dist in sorted(zip(metadatas, distances), key=lambda x: x[1]):
        if dist >= DISTANCE_CUTOFF:
            continue
        source = meta.get("source", "Unknown")
        if source not in seen:
            seen.add(source)
            sources.append({
                "title":   source.replace(".pdf","").replace("-"," ").replace("_"," "),
                "url":     "",
                "section": f"Page {meta.get('page','N/A')}"
            })
        if len(sources) == 4:
            break
    return sources

def parse_llm_response(raw: str) -> tuple[str, list[str]]:
    followups, answer_lines = [], []
    for line in raw.split("\n"):
        if line.strip().startswith("FOLLOWUP:"):
            followups.append(line.replace("FOLLOWUP:", "").strip())
        else:
            answer_lines.append(line)
    return "\n".join(answer_lines).strip(), followups

DEFAULT_FOLLOWUPS = [
    "What is NTPC's renewable energy capacity?",
    "What are NTPC's financial highlights for the latest year?",
    "How has NTPC's installed capacity grown over the years?",
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN RAG FUNCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def get_rag_answer(question: str) -> dict:

    if collection is None or collection.count() == 0:
        return {
            "answer":              "Knowledge base not ready. Please run indexer.py first.",
            "sources":             [],
            "follow_up_questions": DEFAULT_FOLLOWUPS,
        }

    found_years   = detect_years(question)
    is_comparison = len(found_years) > 1
    is_current    = not found_years and is_current_question(question)
    year_hint     = LATEST_YEAR if is_current else (found_years[0] if len(found_years) == 1 else None)

    if is_comparison:
        chunks, metadatas, distances = await fetch_multi_year_parallel(question, found_years)
    else:
        chunks, metadatas, distances = multi_query_chroma(question, year_filter=year_hint)

    if not chunks:
        return {
            "answer":              "This information is not available in the provided NTPC reports.",
            "sources":             [],
            "follow_up_questions": DEFAULT_FOLLOWUPS,
        }

    context = build_context(chunks, metadatas)

    if is_comparison:
        year_instruction = f"Compare data across years: {', '.join(found_years)} — show EACH year separately with numbers."
    elif year_hint:
        year_instruction = f"Use data from year: {year_hint} only."
    else:
        year_instruction = "Use the most recent year available in context."

    user_message = f"""Context from NTPC Annual Reports:

{context}

Question: {question}
Instruction: {year_instruction}

Rules:
- Detailed and complete answer with exact numbers
- For comparisons: show ALL years side by side
- ONLY use information from the context above
- Always cite exact report name and page number
- Minimum 3-4 sentences

End your response with exactly 3 follow-up questions, each on a new line prefixed with 'FOLLOWUP:'"""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            options={
                "temperature": 0.1,
                "num_predict": 600,
                "num_ctx":     4096,
            }
        )
        raw = response["message"]["content"]
    except Exception as e:
        return {
            "answer":              f"LLM error: {str(e)}. Check if Ollama is running.",
            "sources":             [],
            "follow_up_questions": DEFAULT_FOLLOWUPS,
        }

    answer, followups = parse_llm_response(raw)
    sources           = build_sources(metadatas, distances)

    return {
        "answer":              answer,
        "sources":             sources,
        "follow_up_questions": followups[:3] if followups else DEFAULT_FOLLOWUPS,
    }