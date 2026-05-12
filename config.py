"""
config.py
=========
Central configuration for the Egyptian Road & Mobility Assistant.
Covers: traffic laws, vehicle registration, accident/insurance law,
commercial vehicles, driver fitness, road infrastructure, and international driving.

SOURCE POLICY:
  All URLs in SOURCES were manually verified from live search results (May 2026).
  No URL is fabricated or guessed. If a source returns 404 during scraping,
  remove it — do NOT replace with a guessed URL. Instead, search for a real article.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
DATA_DIR    = ROOT_DIR / "data"
SCRAPED_DIR = DATA_DIR / "scraped"
INDEX_DIR   = DATA_DIR / "index"
LOG_DIR     = ROOT_DIR / "logs"
CACHE_DIR   = DATA_DIR / "cache"

for _d in [SCRAPED_DIR, INDEX_DIR, LOG_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL       = "http://localhost:11434"
GENERATOR_MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"
EMBED_MODEL_NAME      = "qwen3-embedding:0.6b"
EMBED_BATCH_SIZE      = 16    # chunks per embedding batch

# ── Generation ────────────────────────────────────────────────────────────────
# ── Generation — update these values in your config.py ───────────────────────

GENERATOR_MAX_NEW_TOKENS = 800   # was 512 — more room for complete answers
GENERATOR_TEMPERATURE    = 0.1
GENERATOR_THINKING_MODE  = False  # /no_think by default; set True per-request if needed

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400   # words
CHUNK_OVERLAP = 50
MIN_CHUNK_LEN = 80    # characters; discard shorter fragments

# ── Scraper ───────────────────────────────────────────────────────────────────
SCRAPE_DELAY_SEC   = 1.5
SCRAPE_TIMEOUT_SEC = 15
SCRAPE_MAX_RETRIES = 2
SCRAPE_MAX_DEPTH   = 3
USER_AGENT         = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
SCRAPE_HEADERS = {"User-Agent": USER_AGENT}

# ── FAISS ─────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH    = INDEX_DIR / "faiss.index"
FAISS_METADATA_PATH = INDEX_DIR / "metadata.jsonl"
FAISS_DIM           = 1024   # qwen3-embedding:0.6b output dim

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K     = 5
MIN_SCORE = 0.20    # cosine similarity floor; lower = recall-biased

# ── Reranker ──────────────────────────────────────────────────────────────────
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_ENABLED    = True
RERANKER_FETCH_K    = 20   # FAISS retrieves this many; reranker re-scores → TOP_K
RERANKER_TOP_K      = 5

# ── Query enhancement (HyDE) ──────────────────────────────────────────────────
HYDE_ENABLED           = True
HYDE_FALLBACK_ON_ERROR = True   # if HyDE fails, fall back to raw query

# ── Conversation memory ───────────────────────────────────────────────────────
SESSION_TTL_MINUTES = 30
SESSION_MAX_TURNS   = 6        # how many past (Q,A) pairs to include in prompt

# ── Semantic cache ────────────────────────────────────────────────────────────
CACHE_ENABLED              = True
CACHE_SIMILARITY_THRESHOLD = 0.92
CACHE_PATH                 = CACHE_DIR / "query_cache.jsonl"

# ── Prompt template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are a legal assistant specializing in Egyptian traffic law, vehicle registration, and road regulations.

STRICT RULES:
1. Answer ONLY from the context provided below. Do not use outside knowledge.
2. If the context contains a partial answer, give what you found and note it may be incomplete.
3. If the context has NO relevant information, say exactly: "لا تتوفر معلومات كافية في المصادر المتاحة." (for Arabic questions) or "Insufficient information in the available sources." (for English questions). Do not guess.
4. LANGUAGE: If the question is in Arabic → answer in Arabic only. If in English → answer in English only. No mixing.
5. FORMAT: If the answer involves steps or procedures → use a numbered list. If it involves a list of documents or fees → use bullet points. Otherwise → use clear paragraphs.
6. Be concise and direct. Do not repeat the question. Do not add preamble like "Great question!" or "Based on the context...".

Context:
{context}

Question: {question}

Answer:"""

PROMPT_TEMPLATE_WITH_HISTORY = """
You are a legal assistant specializing in Egyptian traffic law, vehicle registration, and road regulations.

STRICT RULES:
1. Answer ONLY from the context provided below. Do not use outside knowledge.
2. If the context contains a partial answer, give what you found and note it may be incomplete.
3. If the context has NO relevant information, say exactly: "لا تتوفر معلومات كافية في المصادر المتاحة." (for Arabic questions) or "Insufficient information in the available sources." (for English questions). Do not guess.
4. LANGUAGE: If the question is in Arabic → answer in Arabic only. If in English → answer in English only. No mixing.
5. FORMAT: If the answer involves steps or procedures → use a numbered list. If it involves a list of documents or fees → use bullet points. Otherwise → use clear paragraphs.
6. Be concise and direct. Do not repeat the question. Do not add preamble like "Great question!" or "Based on the context...".

Previous conversation:
{history}

Context:
{context}

Question: {question}

Answer:"""

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST        = "0.0.0.0"
API_PORT        = 8000
API_TITLE       = "Egyptian Road & Mobility Assistant"
API_VERSION     = "2.0.0"
API_DESCRIPTION = """
## Egyptian Road & Mobility Assistant

A bilingual Arabic/English RAG system covering all aspects of Egyptian road law:

- 🚗 **Traffic laws** — Law No. 66/1973 + all amendments through 2025
- 📋 **Driving licenses** — new, renewal, replacement, international permits
- 💰 **Traffic fines** — updated 2024–2025 penalty schedule
- 🚘 **Vehicle registration** — new registration, renewal, ownership transfer, import
- ⚖️ **Accident & liability** — insurance law No.155/2024, compensation, procedures
- 🚛 **Commercial vehicles** — trucks, taxis, ride-hail regulations
- 🏥 **Driver fitness** — medical requirements, age rules, disability access
- 🛣️ **Road infrastructure** — highways, speed cameras, toll roads
- 🌍 **International driving** — foreign licenses, IDP, tourists in Egypt
- 🚶 **Pedestrians** — rights, obligations, crossing rules, violations

**Languages:** Arabic 🇪🇬 and English 🇬🇧
"""

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = LOG_DIR / "pipeline.log"

# ── Data sources ──────────────────────────────────────────────────────────────
# POLICY: Every URL below was confirmed live via web search (May 2026).
# group values: traffic_law | driving_license | vehicle_registration |
#               accident_liability | commercial_vehicles | driver_fitness |
#               international_driving | road_infrastructure | pedestrians
#
# NOTE ON RERANKER: ms-marco cross-encoder is English-trained. Arabic queries
# will get low cross-encoder scores. RERANKER_ENABLED is left True but the
# retriever should fall back to FAISS order when max reranker score < -5.0.
# See rag/reranker.py for the fallback logic.

SOURCES = [

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN 1 — Traffic Law (قانون المرور)
    # Covers: Law 66/1973, amendments, penalties, violations, enforcement
    # ══════════════════════════════════════════════════════════════════════════

    # Core law text — mohamah.net (verified working)
    {
        "name": "mohamah_traffic_law",
        "urls": ["https://www.mohamah.net/law/%D9%82%D8%A7%D9%86%D9%88%D9%86-%D8%A7%D9%84%D9%85%D8%B1%D9%88%D8%B1/"],
        "group": "traffic_law",
        "language": "ar",
        "follow_links": False,
    },

    # 2024 traffic law amendments — cabinet approval, new fines up to 15k EGP
    # Confirmed URL from youm7.com search results
    {
        "name": "youm7_traffic_law_amendments_2025",
        "urls": ["https://www.youm7.com/story/2025/12/24/%D9%85%D8%AC%D9%84%D8%B3-%D8%A7%D9%84%D9%88%D8%B2%D8%B1%D8%A7%D8%A1-%D9%8A%D9%88%D8%A7%D9%81%D9%82-%D8%B9%D9%84%D9%89-%D8%AA%D8%BA%D9%84%D9%8A%D8%B8-%D8%B9%D9%82%D9%88%D8%A7%D8%AA-%D9%85%D8%AE%D8%A7%D9%84%D9%81%D8%A7%D8%AA-%D8%A7%D9%84%D9%85%D8%B1%D9%88%D8%B1-%D8%A7%D8%B9%D8%B1%D9%81-%D8%A7%D9%84%D8%BA%D8%B1%D8%A7%D9%85%D8%A7%D8%AA/7245476"],
        "group": "traffic_law",
        "language": "ar",
        "follow_links": False,
    },

    # Speed limit violations, new penalties — youm7 Jan 2026
    {
        "name": "youm7_speed_penalty_new_law",
        "urls": ["https://www.youm7.com/story/2026/1/9/%D9%82%D8%A7%D9%86%D9%88%D9%86-%D8%A7%D9%84%D9%85%D8%B1%D9%88%D8%B1-%D8%A8%D8%AB%D9%88%D8%A8%D9%87-%D8%A7%D9%84%D8%AC%D8%AF%D9%8A%D8%AF-%D9%87%D9%84-%D8%AA%D9%86%D8%AC%D8%AD-%D8%AA%D8%BA%D9%84%D9%8A%D8%B8-%D8%B9%D9%82%D9%88%D8%A8%D8%A7%D8%AA-%D8%A7%D9%84%D8%B3%D8%B1%D8%B9%D8%A9-%D8%A7%D9%84%D8%B2%D8%A7%D8%A6%D8%AF%D8%A9/7251334"],
        "group": "traffic_law",
        "language": "ar",
        "follow_links": False,
    },

    # Traffic fines table — masrawy (seatbelt, phone, speed amounts)
    {
        "name": "masrawy_fines_table_2024",
        "urls": ["https://www.masrawy.com/news/news_cases/details/2024/1/13/2523162/%D8%A7%D9%84%D8%AD%D8%B2%D8%A7%D9%85-%D9%88%D8%AA%D8%B7%D8%A8%D9%8A%D9%82-%D8%A7%D9%84%D8%B1%D8%A7%D8%AF%D8%A7%D8%B1-%D9%88%D8%A7%D9%84%D8%B3%D8%B1%D8%B9%D8%A9-%D8%A7%D9%84%D8%B2%D8%A7%D8%A6%D8%AF%D8%A9-%D8%BA%D8%B1%D8%A7%D9%85%D8%A7%D8%AA-%D8%A7%D9%84%D9%85%D8%AE%D8%A7%D9%84%D9%81%D8%A7%D8%AA-%D8%A7%D9%84%D9%85%D8%B1%D9%88%D8%B1%D9%8A%D8%A9-"],
        "group": "traffic_law",
        "language": "ar",
        "follow_links": False,
    },

    # 5 major violations and their exact fines — masrawy Apr 2024
    {
        "name": "masrawy_top5_violations_2024",
        "urls": ["https://www.masrawy.com/news/news_cases/details/2024/4/25/2572610/%D8%A8%D8%A7%D9%84%D8%A3%D8%B1%D9%82%D8%A7%D9%85-%D8%BA%D8%B1%D8%A7%D9%85%D8%A9-%D8%A3%D8%A8%D8%B2-5-%D9%85%D8%AE%D8%A7%D9%84%D9%81%D8%A7%D8%AA-%D9%85%D8%B1%D9%88%D8%B1%D9%8A%D8%A9-%D8%B9%D9%84%D9%89-%D8%A7%D9%84%D8%B7%D8%B1%D9%82"],
        "group": "traffic_law",
        "language": "ar",
        "follow_links": False,
    },

    # 10 violations that result in prison + fines — masrawy
    {
        "name": "masrawy_prison_violations",
        "urls": ["https://www.masrawy.com/news/news_egypt/details/2023/9/3/2462723/%D9%85%D9%86%D9%87%D8%A7-%D8%A7%D9%84%D8%B3%D9%8A%D8%B1-%D8%A8%D8%AF%D9%88%D9%86-%D9%81%D8%B1%D8%A7%D9%85%D9%84-%D9%A1%D9%A0-%D9%85%D8%AE%D8%A7%D9%84%D9%81%D8%A7%D8%AA-%D9%81%D9%8A-%D9%82%D8%A7%D9%86%D9%88%D9%86-%D8%A7%D9%84%D9%85%D8%B1%D9%88%D8%B1-%D8%AA%D8%B9%D8%B1%D8%B6%D9%83-%D9%84%D9%84%D8%AD%D8%A8%D8%B3-%D9%88%D8%A7%D9%84%D8%BA%D8%B1%D8%A7%D9%85%D8%A9-%D8%AA%D8%B9%D8%B1%D9%81-%D8%B9%D9%84%D9%8A%D9%87%D8%A7"],
        "group": "traffic_law",
        "language": "ar",
        "follow_links": False,
    },

    # Violations not eligible for settlement — masrawy Feb 2024
    {
        "name": "masrawy_non_settleable_violations",
        "urls": ["https://www.masrawy.com/news/news_cases/details/2024/2/19/2540319/%D9%85%D9%86-%D8%A8%D9%8A%D9%86%D9%87%D8%A7-%D8%A7%D9%84%D8%B3%D8%B1%D9%8A%D9%86%D8%A9-%D9%88%D8%B7%D9%85%D8%B3-%D8%A7%D9%84%D9%84%D9%88%D8%AD%D8%A7%D8%AA-%D9%85%D8%AE%D8%A7%D9%84%D9%81%D8%A7%D8%AA-%D9%84%D8%A7-%D9%8A%D8%AC%D9%88%D8%B2-%D8%A7%D9%84%D8%AA%D8%B5%D8%A7%D9%84%D8%AD-%D9%81%D9%8A%D9%87%D8%A7-%D9%88%D9%81%D9%82-%D8%A7-%D9%84%D9%82%D8%A7%D9%86%D9%88%D9%86-%D8%A7%D9%84%D9%85%D8%B1%D9%88%D8%B1-%D8%A7%D9%84%D8%AC%D8%AF%D9%8A%D8%AF"],
        "group": "traffic_law",
        "language": "ar",
        "follow_links": False,
    },

    # Interior Ministry gazette — executive regulation amendment 2024
    {
        "name": "masrawy_gazette_regulation_2024",
        "urls": ["https://www.masrawy.com/news/news_cases/details/2024/4/3/2562802/%D8%A7%D9%84%D8%AC%D8%B1%D9%8A%D8%AF%D8%A9-%D8%A7%D9%84%D8%B1%D8%B3%D9%85%D9%8A%D8%A9-%D8%AA%D9%86%D8%B4%D8%B1-%D8%AA%D8%B9%D8%AF%D9%8A%D9%84-%D8%A8%D8%B9%D8%B6-%D8%A3%D8%AD%D9%83%D8%A7%D9%85-%D8%A7%D9%84%D9%84%D8%A7%D8%A6%D8%AD%D8%A9-%D8%A7%D9%84%D8%AA%D9%86%D9%81%D9%8A%D8%B0%D9%8A%D8%A9-%D9%84%D9%82%D8%A7%D9%86%D9%88%D9%86-%D8%A7%D9%84%D9%85%D8%B1%D9%88%D8%B1"],
        "group": "traffic_law",
        "language": "ar",
        "follow_links": False,
    },

    # Vehicle modification penalties — youm7 Aug 2024
    {
        "name": "youm7_vehicle_modification_penalty",
        "urls": ["https://www.youm7.com/story/2024/8/6/%D8%AA%D8%B9%D8%B1%D9%81-%D8%B9%D9%84%D9%89-%D8%B9%D9%82%D9%88%D8%A8%D8%A9-%D8%AA%D8%B9%D8%AF%D9%8A%D9%84-%D8%A7%D9%84%D8%B3%D9%8A%D8%A7%D8%B1%D8%A7%D8%AA-%D9%81%D9%89-%D9%82%D8%A7%D9%86%D9%88%D9%86-%D8%A7%D9%84%D9%85%D8%B1%D9%88%D8%B1-%D8%AA%D8%B9%D8%B1%D8%B6%D9%83-%D9%84%D9%84%D8%AD%D8%A8%D8%B3/6664105"],
        "group": "traffic_law",
        "language": "ar",
        "follow_links": False,
    },

    # New speed/lane violation fines — elbalad Dec 2025
    {
        "name": "elbalad_speed_lane_amendments",
        "urls": ["https://www.elbalad.news/6812089"],
        "group": "traffic_law",
        "language": "ar",
        "follow_links": False,
    },

    # Traffic law amendments — cabinet approved, masrawy Dec 2025
    {
        "name": "masrawy_traffic_law_amendments_2025",
        "urls": ["https://www.masrawy.com/news/news_egypt/details/2025/12/24/2913350/%D8%A7%D9%84%D8%AD%D9%83%D9%88%D9%85%D8%A9-%D8%AA%D8%B9%D8%AF%D9%84-%D9%82%D8%A7%D9%86%D9%88%D9%86-%D8%A7%D9%84%D9%85%D8%B1%D9%88%D8%B1-%D9%88%D8%A7%D9%84%D8%BA%D8%B1%D8%A7%D9%85%D8%A7%D8%AA-%D8%AA%D8%B5%D9%84-%D9%84%D9%8015-%D8%A3%D9%84%D9%81-%D8%AC%D9%86%D9%8A%D9%87"],
        "group": "traffic_law",
        "language": "ar",
        "follow_links": False,
    },

    # Dostor — traffic law new fines 2024 (confirmed URL from search)
    {
        "name": "dostor_traffic_law_2024",
        "urls": ["https://www.dostor.org/4689469"],
        "group": "traffic_law",
        "language": "ar",
        "follow_links": False,
    },

    # elwatan — 11 offenses that carry prison + fine
    {
        "name": "elwatan_prison_offenses",
        "urls": ["https://www.elwatannews.com/news/details/7640457"],
        "group": "traffic_law",
        "language": "ar",
        "follow_links": False,
    },

    # Heavy vehicle overloading fines — masrawy Mar 2024
    {
        "name": "masrawy_overload_fines",
        "urls": ["https://www.masrawy.com/news/news_egypt/details/2024/3/6/2548270/%D8%BA%D8%B1%D8%A7%D9%85%D8%A7%D8%AA-%D9%85%D8%BA%D9%84%D8%B8%D8%A9-%D8%A7%D9%84%D9%86%D9%82%D9%84-%D8%AA%D9%86%D8%A7%D8%B4%D8%AF-%D8%B4%D8%B1%D9%83%D8%A7%D8%AA-%D8%A7%D9%84%D9%86%D9%82%D9%84-%D9%88%D8%B3%D8%A7%D8%A6%D9%82%D9%8A-%D8%A7%D9%84%D8%B4%D8%A7%D8%AD%D9%86%D8%A7%D8%AA-%D8%A7%D9%84%D8%A7%D9%84%D8%AA%D8%B2%D8%A7%D9%85-%D8%A8%D8%A7%D9%84%D8%AD%D9%85%D9%88%D9%84%D8%A7%D8%AA-%D8%A7%D9%84%D9%85%D8%B3%D9%85%D9%88%D8%AD-%D8%A7%D9%84%D8%B3%D9%8A%D8%B1-%D8%A8%D9%87%D8%A7-%D8%AA%D9%81%D8%A7%D8%B5%D9%8A%D9%84"],
        "group": "traffic_law",
        "language": "ar",
        "follow_links": False,
    },

    # English — driving rules and regulations Egypt (adcidl, verified working)
    {
        "name": "adcidl_egypt_driving_rules",
        "urls": ["https://www.adcidl.com/Driving-in-Egypt.html"],
        "group": "traffic_law",
        "language": "en",
        "follow_links": False,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN 2 — Driving Licenses (رخصة القيادة)
    # Covers: new license, renewal, replacement, conditions, medical, age
    # ══════════════════════════════════════════════════════════════════════════

    # Driving license procedures — youm7 (confirmed Mar 2022)
    {
        "name": "youm7_driving_license_steps",
        "urls": ["https://www.youm7.com/story/2022/3/21/%D8%AE%D8%B7%D9%88%D8%A7%D8%AA-%D8%AA%D8%B3%D8%A7%D8%B9%D8%AF%D9%83-%D9%81%D9%89-%D8%A7%D8%B3%D8%AA%D8%AE%D8%B1%D8%A7%D8%AC-%D8%B1%D8%AE%D8%B5%D8%A9-%D9%82%D9%8A%D8%A7%D8%AF%D8%A9-%D8%AA%D8%B9%D8%B1%D9%81-%D8%B9%D9%84%D9%8A%D9%87%D8%A7/5698305"],
        "group": "driving_license",
        "language": "ar",
        "follow_links": False,
    },

    # License conditions — youm7 (confirmed Apr 2022)
    {
        "name": "youm7_license_conditions",
        "urls": ["https://www.youm7.com/story/2022/4/29/%D8%A7%D9%84%D9%85%D8%B1%D9%88%D8%B1-%D8%AA%D8%AD%D8%AF%D8%AF-%D8%AE%D8%B7%D9%88%D8%A7%D8%AA-%D8%A7%D8%B3%D8%AA%D8%AE%D8%B1%D8%A7%D8%AC-%D8%B1%D8%AE%D8%B5%D8%A9-%D8%A7%D9%84%D9%82%D9%8A%D8%A7%D8%AF%D8%A9-%D8%A7%D8%B9%D8%B1%D9%81-%D8%A7%D9%84%D8%AA%D9%81%D8%A7%D8%B5%D9%8A%D9%84/5742806"],
        "group": "driving_license",
        "language": "ar",
        "follow_links": False,
    },

    # License requirements — youm7 Dec 2024
    {
        "name": "youm7_license_requirements_2024",
        "urls": ["https://www.youm7.com/story/2024/12/17/%D8%AA%D8%B9%D8%B1%D9%81-%D8%B9%D9%84%D9%89-%D8%B4%D8%B1%D9%88%D8%B7-%D8%A7%D8%B3%D8%AA%D8%AE%D8%B1%D8%A7%D8%AC-%D8%B1%D8%AE%D8%B5%D8%A9-%D9%82%D9%8A%D8%A7%D8%AF%D8%A9-%D8%AE%D8%A7%D8%B5%D8%A9-%D9%84%D9%84%D8%B3%D8%A7%D8%A6%D9%82%D9%8A%D9%86/6813996"],
        "group": "driving_license",
        "language": "ar",
        "follow_links": False,
    },

    # License renewal — youm7 Sep 2025
    {
        "name": "youm7_license_renewal_2025",
        "urls": ["https://www.youm7.com/7125822"],
        "group": "driving_license",
        "language": "ar",
        "follow_links": False,
    },

    # Lost license replacement online — youm7 (confirmed Sep 2022)
    {
        "name": "youm7_lost_license_replacement",
        "urls": ["https://www.youm7.com/story/2022/9/20/%D8%A7%D8%B9%D8%B1%D9%81-%D8%AE%D8%B7%D9%88%D8%A7%D8%AA-%D8%A7%D8%B3%D8%AA%D8%AE%D8%B1%D8%A7%D8%AC-%D8%A8%D8%AF%D9%84-%D9%81%D8%A7%D9%82%D8%AF-%D8%B1%D8%AE%D8%B5%D8%A9-%D8%A7%D9%84%D9%82%D9%8A%D8%A7%D8%AF%D8%A9-%D8%A5%D9%84%D9%83%D8%AA%D8%B1%D9%88%D9%86%D9%8A%D8%A7/5911245"],
        "group": "driving_license",
        "language": "ar",
        "follow_links": False,
    },

    # Mobile traffic units — renew license from home (masrawy Apr 2024)
    {
        "name": "masrawy_license_mobile_units",
        "urls": ["https://www.masrawy.com/news/news_cases/details/2024/4/20/2570131/-%D9%85%D8%B4-%D9%85%D8%AD%D8%AA%D8%A7%D8%AC-%D8%AA%D8%B1%D9%88%D8%AD-%D8%A7%D9%84%D9%85%D8%B1%D9%88%D8%B1-%D9%81%D8%AD%D8%B5-%D9%88%D8%AA%D8%AC%D8%AF%D9%8A%D8%AF-%D8%B3%D9%8A%D8%A7%D8%B1%D8%AA%D9%83-%D9%85%D9%86-%D9%85%D9%83%D8%A7%D9%86%D9%83-%D8%A7%D9%84%D8%AE%D8%B7%D9%88%D8%A7%D8%AA-"],
        "group": "driving_license",
        "language": "ar",
        "follow_links": False,
    },

    # Vehicle license renewal — fees and steps 2024 (elbalad)
    {
        "name": "elbalad_vehicle_license_renewal_fees",
        "urls": ["https://www.elbalad.news/6009765"],
        "group": "driving_license",
        "language": "ar",
        "follow_links": False,
    },

    # Vehicle license renewal with inspection — elwatan detailed steps
    {
        "name": "elwatan_license_renewal_inspection",
        "urls": ["https://www.elwatannews.com/news/details/6987568"],
        "group": "driving_license",
        "language": "ar",
        "follow_links": False,
    },

    # Vehicle license renewal fees by CC (elwatan 2024)
    {
        "name": "elwatan_license_fees_2024",
        "urls": ["https://www.elwatannews.com/news/details/7142497"],
        "group": "driving_license",
        "language": "ar",
        "follow_links": False,
    },

    # English — driving license guide (expatfocus, verified working)
    {
        "name": "expatfocus_driving_license",
        "urls": ["https://www.expatfocus.com/egypt/guide/egypt-driving-licenses"],
        "group": "driving_license",
        "language": "en",
        "follow_links": False,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN 3 — Vehicle Registration (تسجيل وترخيص السيارات)
    # Covers: new car registration, renewal, fees, technical inspection,
    #         number plates, ownership transfer, import procedures
    # ══════════════════════════════════════════════════════════════════════════

    # New car registration procedures 2024 (zyadda.com)
    {
        "name": "zyadda_new_car_registration",
        "urls": ["https://www.zyadda.com/procedures-licensing-new-car-egypt/"],
        "group": "vehicle_registration",
        "language": "ar",
        "follow_links": False,
    },

    # New car license — 10 steps, fees by cc (elwatan 2024)
    {
        "name": "elwatan_new_car_license_steps",
        "urls": ["https://www.elwatannews.com/news/details/7104592"],
        "group": "vehicle_registration",
        "language": "ar",
        "follow_links": False,
    },

    # New car license fees by engine size (elwatan Dec 2023)
    {
        "name": "elwatan_car_license_fees_cc",
        "urls": ["https://www.elwatannews.com/news/details/6997861"],
        "group": "vehicle_registration",
        "language": "ar",
        "follow_links": False,
    },

    # Ownership transfer — steps and documents (almasryalyoum)
    {
        "name": "almasry_ownership_transfer_steps",
        "urls": ["https://www.almasryalyoum.com/news/details/3070743"],
        "group": "vehicle_registration",
        "language": "ar",
        "follow_links": False,
    },

    # Ownership transfer — masrawy (شهادة بيانات, Feb 2024)
    {
        "name": "masrawy_ownership_transfer_data_cert",
        "urls": ["https://www.masrawy.com/news/news_cases/details/2024/2/20/2541071"],
        "group": "vehicle_registration",
        "language": "ar",
        "follow_links": False,
    },

    # Ownership transfer — naqal malakiya forms + steps (tahiamasr)
    {
        "name": "tahiamasr_ownership_transfer",
        "urls": ["https://www.tahiamasr.com/831711"],
        "group": "vehicle_registration",
        "language": "ar",
        "follow_links": False,
    },

    # Registration fees at notary (almasryalyoum 2024 fee table)
    {
        "name": "almasry_registration_fees_notary",
        "urls": ["https://www.almasryalyoum.com/news/details/3106210"],
        "group": "vehicle_registration",
        "language": "ar",
        "follow_links": False,
    },

    # Vehicle data certificate / شهادة بيانات (dostor 2026)
    {
        "name": "dostor_vehicle_data_certificate",
        "urls": ["https://www.dostor.org/5545575"],
        "group": "vehicle_registration",
        "language": "ar",
        "follow_links": False,
    },

    # Import a car from abroad — 7 steps (youm7 Oct 2023)
    {
        "name": "youm7_car_import_7steps",
        "urls": ["https://www.youm7.com/story/2023/10/17/%D8%AA%D8%B9%D8%B1%D9%81-%D8%B9%D9%84%D9%89-%D8%B7%D8%B1%D9%8A%D9%82%D8%A9-%D8%A7%D8%B3%D8%AA%D9%8A%D8%B1%D8%A7%D8%AF-%D8%B3%D9%8A%D8%A7%D8%B1%D8%A9-%D9%85%D9%86-%D8%A7%D9%84%D8%AE%D8%A7%D8%B1%D8%AC-%C3%97-7-%D8%AE%D8%B7%D9%88%D8%A7%D8%AA/6339441"],
        "group": "vehicle_registration",
        "language": "ar",
        "follow_links": False,
    },

    # Import car from Europe — conditions and customs (youm7 Jan 2020)
    {
        "name": "youm7_car_import_europe",
        "urls": ["https://www.youm7.com/story/2020/1/19/%D8%A5%D8%B2%D8%A7%D9%89-%D8%AA%D8%B4%D8%AA%D8%B1%D9%89-%D8%B9%D8%B1%D8%A8%D9%8A%D8%A9-%D8%AC%D8%AF%D9%8A%D8%AF%D8%A9-%D8%A3%D9%88-%D9%85%D8%B3%D8%AA%D8%B9%D9%85%D9%84%D8%A9-%D9%85%D9%86-%D8%A3%D9%88%D8%B1%D9%88%D8%A8%D8%A7-%D9%88%D8%AA%D8%AF%D8%AE%D9%84%D9%87%D8%A7-%D9%85%D8%B5%D8%B1/4594382"],
        "group": "vehicle_registration",
        "language": "ar",
        "follow_links": False,
    },

    # English — vehicle registration plates of Egypt (wikiwand)
    {
        "name": "wikiwand_egypt_plates",
        "urls": ["https://www.wikiwand.com/en/Vehicle_registration_plates_of_Egypt"],
        "group": "vehicle_registration",
        "language": "en",
        "follow_links": False,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN 4 — Accident Liability & Insurance (حوادث وتأمين)
    # Covers: what to do after accident, insurance law 155/2024,
    #         compensation claims, government fund for unknown vehicles
    # ══════════════════════════════════════════════════════════════════════════

    # Insurance compensation steps — 7 steps youm7 (Jun 2020)
    {
        "name": "youm7_insurance_compensation_steps",
        "urls": ["https://www.youm7.com/story/2020/6/4/7-%D8%AE%D8%B7%D9%88%D8%A7%D8%AA-%D9%84%D9%84%D8%AD%D8%B5%D9%88%D9%84-%D8%B9%D9%84%D9%89-%D8%AA%D8%B9%D9%88%D9%8A%D8%B6-%D8%AD%D8%A7%D8%AF%D8%AB-%D8%A7%D9%84%D8%B3%D9%8A%D8%A7%D8%B6/4805470"],
        "group": "accident_liability",
        "language": "ar",
        "follow_links": False,
    },

    # Road accident rights, 40k EGP compensation, government fund
    # for unknown vehicles — youm7 (Dec 2019, confirmed URL)
    {
        "name": "youm7_accident_rights_fund",
        "urls": ["https://www.youm7.com/story/2019/12/16/%D8%AD%D9%82%D9%88%D9%82-%D8%B9%D9%84%D9%89-%D8%A7%D9%84%D8%B7%D8%B1%D9%8A%D9%82-%D9%8A%D8%AC%D9%87%D9%84%D9%87%D8%A7-%D8%A7%D9%84%D9%85%D9%84%D8%A7%D9%8A%D9%8A%D9%86-%D8%A7%D9%84%D9%85%D9%8F%D8%B4%D8%B1%D8%B9-%D9%83%D9%84%D9%81-%D8%B4%D8%B1%D9%83%D8%A7%D8%AA-%D8%A7%D9%84%D8%AA%D8%A3%D9%85%D9%8A%D9%86-%D8%A8%D8%AA%D8%B9%D9%88%D9%8A%D8%B6/4547645"],
        "group": "accident_liability",
        "language": "ar",
        "follow_links": False,
    },

    # Rights + documents for road accident compensation — youm7 Feb 2022
    {
        "name": "youm7_accident_compensation_docs",
        "urls": ["https://www.youm7.com/story/2022/2/3/%D8%AD%D9%82%D9%88%D9%82-%D8%B9%D9%84%D9%89-%D8%A7%D9%84%D8%B7%D8%B1%D9%8A%D9%82-%D9%8A%D8%AC%D9%87%D9%84%D9%87%D8%A7-%D8%A7%D9%84%D9%85%D8%AA%D8%B6%D8%B1%D8%B1%D9%8A%D9%86-%D8%A7%D9%84%D9%85%D9%8F%D8%B4%D8%B1%D8%B9-%D9%83%D9%84%D9%81-%D8%B4%D8%B1%D9%83%D8%A7%D8%AA-%D8%A7%D9%84%D8%AA%D8%A3%D9%85%D9%8A%D9%86-%D8%A8%D8%AA%D8%B9%D9%88%D9%8A%D8%B6/5288612"],
        "group": "accident_liability",
        "language": "ar",
        "follow_links": False,
    },

    # Role of insurance in road accident compensation, law 155/2024 — youm7 Jul 2025
    {
        "name": "youm7_insurance_law_155_2024",
        "urls": ["https://www.youm7.com/story/2025/7/13/%D8%AA%D8%B9%D8%B1%D9%81-%D8%B9%D9%84%D9%89-%D8%AF%D9%88%D8%B1-%D8%A7%D9%84%D8%AA%D8%A3%D9%85%D9%8A%D9%86-%D9%81%D9%8A-%D8%AA%D8%B9%D9%88%D9%8A%D8%B6-%D8%A7%D9%84%D8%A3%D8%B6%D8%B1%D8%A7%D8%B1-%D8%A7%D9%84%D9%86%D8%A7%D8%AC%D9%85%D8%A9-%D8%B9%D9%86-%D8%AD%D9%88%D8%A7%D8%AF%D8%AB/7050688"],
        "group": "accident_liability",
        "language": "ar",
        "follow_links": False,
    },

    # Compensation steps — youm7 Apr 2025
    {
        "name": "youm7_compensation_steps_2025",
        "urls": ["https://www.youm7.com/story/2025/4/1/%D8%A7%D8%B9%D8%B1%D9%81-%D8%A3%D9%87%D9%85-%D8%AE%D8%B7%D9%88%D8%A7%D8%AA-%D8%A7%D9%84%D8%AD%D8%B5%D9%88%D9%84-%D8%B9%D9%84%D9%89-%D8%AA%D8%B9%D9%88%D9%8A%D8%B6-%D8%AD%D8%A7%D8%AF%D8%AB-%D8%A7%D9%84%D8%B3%D9%8A%D8%A7%D8%B1%D8%A9-%D9%85%D9%86-%D8%B4%D8%B1%D9%83%D8%A7%D8%AA/6939302"],
        "group": "accident_liability",
        "language": "ar",
        "follow_links": False,
    },

    # Insurance depreciation rates after accident — masrawy Apr 2026
    {
        "name": "masrawy_insurance_depreciation_2026",
        "urls": ["https://www.masrawy.com/autos/autos_news/details/2026/4/7/2968931/%D9%85%D8%A7-%D8%AA%D8%A3%D8%AB%D9%8A%D8%B1-%D8%AA%D8%B9%D8%AF%D9%8A%D9%84%D8%A7%D8%AA-%D9%86%D8%B3%D8%A8-%D8%A7%D8%B3%D8%AA%D9%87%D9%84%D8%A7%D9%83-%D8%AA%D8%A3%D9%85%D9%8A%D9%86-%D8%A7%D9%84%D8%B3%D9%8A%D8%A7%D8%B1%D8%A7%D8%AA-%D8%B9%D9%84%D9%89-%D8%A7%D9%84%D8%B9%D9%85%D9%84%D8%A7%D8%A1-%D8%A8%D8%B9%D8%AF-%D8%A7%D9%84%D8%AD%D9%88%D8%A7%D8%AF%D8%AB-"],
        "group": "accident_liability",
        "language": "ar",
        "follow_links": False,
    },

    # mohamah.net insurance law text (verified working)
    {
        "name": "mohamah_insurance_law",
        "urls": ["https://www.mohamah.net/law/%D9%82%D8%A7%D9%86%D9%88%D9%86-%D8%A7%D9%84%D8%AA%D8%A3%D9%85%D9%8A%D9%86/"],
        "group": "accident_liability",
        "language": "ar",
        "follow_links": False,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN 5 — Commercial Vehicles (المركبات التجارية)
    # Covers: trucks, taxis, Uber/Careem, overloading, heavy transport rules
    # ══════════════════════════════════════════════════════════════════════════

    # Heavy vehicles / truck penalty amendments — youm7 Sep 2021
    {
        "name": "youm7_heavy_vehicle_penalties",
        "urls": ["https://www.youm7.com/story/2021/9/24/%D8%A7%D9%84%D8%AD%D8%A8%D8%B3-%D9%88%D8%A7%D9%84%D8%BA%D8%B1%D8%A7%D9%85%D8%A9-%D9%82%D8%A7%D9%86%D9%88%D9%86-%D8%A7%D9%84%D9%85%D8%B1%D9%88%D8%B1-%D8%A7%D9%84%D8%AC%D8%AF%D9%8A%D8%AF-%D9%8A%D8%B4%D8%AF%D8%AF-%D8%B9%D9%82%D9%88%D8%A8%D8%A7%D8%AA-%D9%85%D8%B1%D8%AA%D9%83%D8%A8%D9%89-%D8%AD%D9%88%D8%A7%D8%AF%D8%AB-%D8%B3%D9%8A%D8%A7%D8%B1%D8%A7%D8%AA/5469965"],
        "group": "commercial_vehicles",
        "language": "ar",
        "follow_links": False,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN 6 — Driver Fitness (اللياقة الطبية للسائق)
    # Covers: medical requirements, age rules, disability driving
    # ══════════════════════════════════════════════════════════════════════════

    # Disabled persons vehicle — conditions 2024 (youm7 Oct 2024)
    {
        "name": "youm7_disabled_vehicle_2024",
        "urls": ["https://www.youm7.com/story/2024/10/13/%D8%B4%D8%B1%D9%88%D8%B7-%D8%AC%D8%AF%D9%8A%D8%AF%D8%A9-%D9%84%D9%84%D8%AD%D8%B5%D9%88%D9%84-%D8%B9%D9%84%D9%89-%D8%B3%D9%8A%D8%A7%D8%B1%D8%A9-%D9%85%D8%B9%D8%A7%D9%82%D9%8A%D9%86-%D8%AE%D8%B7%D9%88%D8%A7%D8%AA-%D8%AD%D8%AC%D8%B2-%D8%A7%D9%84%D9%83%D8%B4%D9%81-%D9%88%D8%A7%D9%84%D8%A3%D9%88%D8%B1%D8%A7%D9%82/6738808"],
        "group": "driver_fitness",
        "language": "ar",
        "follow_links": False,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN 7 — International Driving (القيادة الدولية)
    # Covers: IDP procedures, conditions, foreign licenses in Egypt
    # ══════════════════════════════════════════════════════════════════════════

    # International license — steps and documents youm7 (Mar 2022)
    {
        "name": "youm7_international_license_steps",
        "urls": ["https://www.youm7.com/story/2022/3/2/%D8%AA%D8%B9%D8%B1%D9%81-%D8%B9%D9%84%D9%89-%D8%A5%D8%AC%D8%B1%D8%A7%D8%A1%D8%A7%D8%AA-%D8%A7%D8%B3%D8%AA%D8%AE%D8%B1%D8%A7%D8%AC-%D8%B1%D8%AE%D8%B5%D8%A9-%D9%82%D9%8A%D8%A7%D8%AF%D8%A9-%D8%AF%D9%88%D9%84%D9%8A%D8%A9-%C3%97-8-%D9%85%D8%B9%D9%84%D9%88%D9%85%D8%A7%D8%AA/5674078"],
        "group": "international_driving",
        "language": "ar",
        "follow_links": False,
    },

    # International license — for travelers abroad youm7 Sep 2025
    {
        "name": "youm7_international_license_2025",
        "urls": ["https://www.youm7.com/story/2025/9/19/%D9%84%D9%84%D9%85%D8%B3%D8%A7%D9%81%D8%B1%D9%8A%D9%86-%D8%AE%D8%A7%D8%B1%D8%AC-%D9%85%D8%B5%D8%B1-%D8%AE%D8%B7%D9%88%D8%A7%D8%AA-%D8%A7%D8%B3%D8%AA%D8%AE%D8%B1%D8%A7%D8%AC-%D8%B1%D8%AE%D8%B5%D8%A9-%D8%A7%D9%84%D9%82%D9%8A%D8%A7%D8%AF%D8%A9-%D8%A7%D9%84%D8%AF%D9%88%D9%84%D9%8A%D8%A9/7125839"],
        "group": "international_driving",
        "language": "ar",
        "follow_links": False,
    },

    # English — IDP Egypt (internationaldrivingpermit.org, verified working)
    {
        "name": "idp_egypt_rules",
        "urls": ["https://internationaldrivingpermit.org/country/egypt/"],
        "group": "international_driving",
        "language": "en",
        "follow_links": False,
    },

    # English — expatfocus IDP/driving in Egypt (verified working)
    {
        "name": "expatfocus_driving_egypt",
        "urls": ["https://www.expatfocus.com/egypt/guide/egypt-driving-licenses"],
        "group": "international_driving",
        "language": "en",
        "follow_links": False,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN 8 — Road Infrastructure (البنية التحتية للطرق)
    # Covers: speed cameras, highway rules, speed limits, tolls, smart transport
    # ══════════════════════════════════════════════════════════════════════════

    # Speed cameras — radar detector ban, fines youm7
    # (pulled from confirmed search result on radar violations)
    {
        "name": "masrawy_radar_detector_ban",
        "urls": ["https://www.masrawy.com/news/news_cases/details/2024/1/13/2523162/%D8%A7%D9%84%D8%AD%D8%B2%D8%A7%D9%85-%D9%88%D8%AA%D8%B7%D8%A8%D9%8A%D9%82-%D8%A7%D9%84%D8%B1%D8%A7%D8%AF%D8%A7%D8%B1-%D9%88%D8%A7%D9%84%D8%B3%D8%B1%D8%B9%D8%A9-%D8%A7%D9%84%D8%B2%D8%A7%D8%A6%D8%AF%D8%A9-%D8%BA%D8%B1%D8%A7%D9%85%D8%A7%D8%AA-%D8%A7%D9%84%D9%85%D8%AE%D8%A7%D9%84%D9%81%D8%A7%D8%AA-%D8%A7%D9%84%D9%85%D8%B1%D9%88%D8%B1%D9%8A%D8%A9-"],
        "group": "road_infrastructure",
        "language": "ar",
        "follow_links": False,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN 9 — Pedestrians (المشاة)
    # Covers: pedestrian rules, crossing violations, rights on the road,
    #         pedestrian accidents and liability
    # ══════════════════════════════════════════════════════════════════════════

    # Pedestrian violations — settlement allowed (law text from traffic.moi.gov.eg)
    # Note: this is a static page on the MOI site, not JS-rendered — confirmed accessible
    {
        "name": "moi_pedestrian_violation_settlement",
        "urls": ["https://traffic.moi.gov.eg/Arabic/OurServices/InfoServices/TrafficGuide/Pages/traffic-violations-and-penalties.aspx"],
        "group": "pedestrians",
        "language": "ar",
        "follow_links": False,
    },

    # Pedestrian safety — traffic police directive (youm7 tag page)
    {
        "name": "youm7_pedestrian_safety",
        "urls": ["https://www.youm7.com/Tags/Index?id=82663&tag=%D8%AD%D9%88%D8%A7%D8%AF%D8%AB-%D8%A7%D9%84%D8%B3%D9%8A%D8%A7%D8%B1%D8%A7%D8%AA"],
        "group": "pedestrians",
        "language": "ar",
        "follow_links": False,
    },

    # dostor — new traffic law 2024 covers pedestrians and all road users
    {
        "name": "dostor_traffic_all_users_2024",
        "urls": ["https://www.dostor.org/4689469"],
        "group": "pedestrians",
        "language": "ar",
        "follow_links": False,
    },

]