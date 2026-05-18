# backend/rag.py
import os
import chromadb
import ollama
from embedder import embed_text

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHROMA_PATH = "/Users/ayushkumar/ntpc-rag-chatbot/backend/chroma"
COLLECTION_NAME = "ntpc_content"
MODEL = "llama3.2"
TOP_K = 30

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

# Query expansion — fuzzy topics ko better search karo
QUERY_EXPANSIONS = {
    # Strategy
    "brighter plan":        "Brighter Plan 2032 renewable energy strategy decarbonization net zero",
    "vision":               "NTPC vision mission strategy long term plan",
    "strategy":             "NTPC strategy plan target goals objectives",

    # Energy
    "green hydrogen":       "green hydrogen projects pilot NTPC electrolyzer",
    "renewable target":     "renewable energy capacity target solar wind 2032 GW",
    "solar":                "solar power capacity MW GW projects commissioned",
    "wind":                 "wind energy capacity projects commissioned MW",
    "hydro":                "hydro power hydroelectric capacity stations",
    "nuclear":              "nuclear energy power NTPC atomic",
    "coal":                 "coal based thermal power stations capacity generation",

    # Financial
    "dividend history":     "dividend per share interim final equity shareholder payout",
    "total income":         "total income revenue from operations turnover crore",
    "profit":               "profit after tax PAT net profit crore",
    "revenue":              "revenue total income operations crore financial",
    "debt":                 "debt borrowings long term short term crore ratio",
    "credit rating":        "credit rating CRISIL ICRA AAA debt rating",
    "capex":                "capital expenditure capex investment crore",
    "pat":                  "profit after tax PAT net profit crore earnings",
    "ebitda":               "EBITDA operating profit earnings before interest tax",

    # Operational
    "plf":                  "Plant Load Factor PLF coal stations national average generation",
    "capacity":             "installed capacity commercial MW GW stations total",
    "generation":           "gross generation units BU billion units sent out",
    "employees":            "number of employees workforce manpower human resource",
    "stations":             "power stations plants NTPC locations units",

    # Sustainability
    "csr":                  "CSR corporate social responsibility community development education health",
    "esg":                  "ESG environment social governance sustainability reporting",
    "carbon":               "carbon emission CO2 greenhouse gas climate change",
    "water":                "water consumption stewardship management conservation",
    "environment":          "environment sustainability green emission pollution",

    # Corporate
    "subsidiaries":         "subsidiaries joint ventures NTPC group companies list",
    "board":                "board of directors chairman CMD management",
    "awards":               "awards recognition achievements honours certifications",
    "csr":                  "CSR corporate social responsibility community",
    "related party":        "related party transactions subsidiaries associates",

   #  some more 
   "auditor":    "statutory auditor CA chartered accountant audit firm",
   "pankaj":     "auditor statutory CA chartered accountant firm",
   "director":   "board of directors independent executive non-executive",
   "chairman":   "chairman CMD MD CEO management board",
   "director":   "Shri Mahabir Prasad has done M.Sc. (Statistics) from University of Delhi and is a law graduate."
}

SYSTEM_PROMPT = """You are a strict document retrieval assistant for NTPC Limited.

ABSOLUTE RULES — NO EXCEPTIONS:
1. Use ONLY the context provided — ZERO outside knowledge
2. If not in context → respond EXACTLY: "This information is not available in the provided NTPC reports."
3. NEVER say 'However', 'I can tell you', 'based on general knowledge', 'I am unable to verify but' — FORBIDDEN
4. NEVER suggest or guess — only state what is written in context
5. Always mention exact report name and page number
6. Numbers in context → state directly, never say 'not explicitly stated'
7. Detailed answers — minimum 3-4 sentences with all numbers"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHROMADB INIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def init_collection():
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col = client.get_collection(name=COLLECTION_NAME)
        print(f"✅ ChromaDB loaded. Documents: {col.count()}")
        return col
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return None

collection = init_collection()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_years(question: str):
    return [y for y in YEAR_TO_FILE if y in question]

def is_current_question(question: str) -> bool:
    keywords = ["current", "latest", "now", "today", "recent", "abhi", "present"]
    return any(kw in question.lower() for kw in keywords)

def expand_query(question: str, base_q: str) -> str:
    q_lower = question.lower()
    for key, expansion in QUERY_EXPANSIONS.items():
        if key in q_lower:
            return f"{base_q} {expansion}"
    return base_q

def query_chroma(embedding, year_filter=None, top_k=TOP_K):
    try:
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": min(top_k, collection.count()),
            "include": ["documents", "metadatas", "distances"]
        }
        if year_filter:
            kwargs["where"] = {"source": YEAR_TO_FILE[year_filter]}
        return collection.query(**kwargs)
    except Exception:
        return collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

def fetch_multi_year(embedding, years):
    all_chunks, all_metas, all_dists = [], [], []
    for yr in years:
        try:
            res = query_chroma(embedding, year_filter=yr, top_k=10)
            all_chunks += res["documents"][0]
            all_metas  += res["metadatas"][0]
            all_dists  += res["distances"][0]
        except Exception:
            pass
    return all_chunks, all_metas, all_dists

def build_context(chunks, metadatas):
    parts = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        source = meta.get("source", meta.get("title", "Unknown"))
        page   = meta.get("page", "")
        label  = f"{source} | Page {page}" if page else source
        parts.append(f"[Source {i+1}: {label}]\n{chunk}")
    return "\n\n---\n\n".join(parts)

def build_sources(metadatas, distances):
    sources = []
    seen = set()
    for meta, dist in sorted(zip(metadatas, distances), key=lambda x: x[1]):
        source = meta.get("source", "Unknown")
        if source not in seen and dist < 1.5:
            seen.add(source)
            sources.append({
                "title": source.replace(".pdf", "").replace("-", " ").replace("_", " "),
                "url": "",
                "section": f"Page {meta.get('page', 'N/A')}"
            })
    return sources[:4]

def parse_llm_response(raw: str):
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
    "How has NTPC's installed capacity grown over the years?"
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN RAG FUNCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def get_rag_answer(question: str) -> dict:

    if collection is None or collection.count() == 0:
        return {
            "answer": "Knowledge base not ready. Please run indexer.py first.",
            "sources": [],
            "follow_up_questions": DEFAULT_FOLLOWUPS
        }

    # Step 1: Detect years + intent
    found_years   = detect_years(question)
    is_comparison = len(found_years) > 1
    is_current    = not found_years and is_current_question(question)

    if is_current:
        year_hint  = LATEST_YEAR
        enhanced_q = f"{question} (latest year: {LATEST_YEAR})"
    elif len(found_years) == 1:
        year_hint  = found_years[0]
        enhanced_q = f"{question} (year: {found_years[0]})"
    else:
        year_hint  = None
        enhanced_q = question

    # Step 1b: Query expansion for fuzzy topics
    enhanced_q = expand_query(question, enhanced_q)

    # Step 2: Embed
    embedding = embed_text(enhanced_q)

    # Step 3: Fetch
    if is_comparison:
        chunks, metadatas, distances = fetch_multi_year(embedding, found_years)
    else:
        results   = query_chroma(embedding, year_filter=year_hint)
        chunks    = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

    if not chunks:
        return {
            "answer": "This information is not available in the provided NTPC reports.",
            "sources": [],
            "follow_up_questions": DEFAULT_FOLLOWUPS
        }

    # Step 4: Build context + prompt
    context = build_context(chunks, metadatas)

    if is_comparison:
        year_instruction = f"(Compare data across years: {', '.join(found_years)} — show EACH year separately)"
    elif year_hint:
        year_instruction = f"(Use data from year: {year_hint} only)"
    else:
        year_instruction = "(Use the most recent year available in context)"

    user_message = f"""Context from NTPC Annual Reports:

{context}

Question: {question}
{year_instruction}

IMPORTANT:
- Give a DETAILED and COMPLETE answer with exact numbers
- For comparison: show ALL years side by side with numbers
- For strategy/plan questions: include all targets, timelines, and key details
- ONLY use information from the context above
- Always mention exact report name and page number
- Minimum 3-4 sentences

End with exactly 3 follow-up questions on separate lines prefixed with 'FOLLOWUP:'"""

    # Step 5: LLM call
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message}
            ]
        )
        raw = response["message"]["content"]
    except Exception as e:
        return {
            "answer": f"LLM error: {str(e)}. Please check if Ollama is running.",
            "sources": [],
            "follow_up_questions": DEFAULT_FOLLOWUPS
        }

    # Step 6: Parse + return
    answer, followups = parse_llm_response(raw)
    sources = build_sources(metadatas, distances)

    return {
        "answer": answer,
        "sources": sources,
        "follow_up_questions": followups[:3] if followups else DEFAULT_FOLLOWUPS
    }
