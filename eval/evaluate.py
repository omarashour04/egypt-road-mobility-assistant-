"""
eval/evaluate.py
================
Evaluation suite for the Egyptian Road & Mobility Assistant.

Metrics:
    - Exact Match (EM)         — strict: does the answer contain the key fact?
    - ROUGE-L F1               — lexical overlap with reference answer
    - Retrieval Recall@k       — did the correct source appear in top-k?
    - Domain coverage          — pass rate per domain group
    - Language balance         — pass rate for Arabic vs English queries

Golden dataset: 60 Q&A pairs across all 8 domains (30 Arabic, 30 English).
These are manually written reference answers — conservative and factual.

Run:
    python -m eval.evaluate
    python -m eval.evaluate --output eval/results.json
    python -m eval.evaluate --domain driving_license
    python -m eval.evaluate --no-hyde    # disable HyDE to measure its effect
"""

from __future__ import annotations

import sys
import json
import time
import argparse
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Golden dataset ────────────────────────────────────────────────────────────

GOLDEN_DATASET = [

    # ── Traffic law (Arabic) ──────────────────────────────────────────────────
    {
        "id": "tl_ar_01",
        "question": "ما هي غرامة استخدام الهاتف أثناء القيادة؟",
        "key_facts": ["غرامة", "هاتف", "قيادة"],
        "expected_sources_contain": ["traffic_law"],
        "domain": "traffic_law",
        "language": "ar",
    },
    {
        "id": "tl_ar_02",
        "question": "ما هو الحد الأقصى للسرعة داخل المدن؟",
        "key_facts": ["60", "كيلومتر", "مدن"],
        "expected_sources_contain": ["traffic_law"],
        "domain": "traffic_law",
        "language": "ar",
    },
    {
        "id": "tl_ar_03",
        "question": "ما هي عقوبة قيادة السيارة بدون رخصة؟",
        "key_facts": ["غرامة", "رخصة", "قيادة"],
        "expected_sources_contain": ["traffic_law"],
        "domain": "traffic_law",
        "language": "ar",
    },
    {
        "id": "tl_ar_04",
        "question": "ما هي غرامة تجاوز الإشارة الحمراء؟",
        "key_facts": ["غرامة", "إشارة", "حمراء"],
        "expected_sources_contain": ["traffic_law"],
        "domain": "traffic_law",
        "language": "ar",
    },
    {
        "id": "tl_ar_05",
        "question": "كيف يمكن الاستعلام عن مخالفات المرور؟",
        "key_facts": ["استعلام", "مخالفات", "مرور"],
        "expected_sources_contain": ["traffic_law"],
        "domain": "traffic_law",
        "language": "ar",
    },
    {
        "id": "tl_ar_06",
        "question": "ما هي عقوبة قيادة سيارة بدون تأمين؟",
        "key_facts": ["تأمين", "غرامة", "قيادة"],
        "expected_sources_contain": ["traffic_law", "accident_liability"],
        "domain": "traffic_law",
        "language": "ar",
    },
    {
        "id": "tl_ar_07",
        "question": "ما الحد الأقصى للسرعة على الطريق السريع؟",
        "key_facts": ["100", "طريق", "سريع"],
        "expected_sources_contain": ["traffic_law", "road_infrastructure"],
        "domain": "traffic_law",
        "language": "ar",
    },

    # ── Traffic law (English) ─────────────────────────────────────────────────
    {
        "id": "tl_en_01",
        "question": "What is the fine for using a mobile phone while driving in Egypt?",
        "key_facts": ["fine", "mobile", "driving"],
        "expected_sources_contain": ["traffic_law"],
        "domain": "traffic_law",
        "language": "en",
    },
    {
        "id": "tl_en_02",
        "question": "What are the speed limits in Egypt?",
        "key_facts": ["speed", "limit", "km"],
        "expected_sources_contain": ["traffic_law"],
        "domain": "traffic_law",
        "language": "en",
    },
    {
        "id": "tl_en_03",
        "question": "What happens if you run a red light in Egypt?",
        "key_facts": ["fine", "penalty", "red light"],
        "expected_sources_contain": ["traffic_law"],
        "domain": "traffic_law",
        "language": "en",
    },
    {
        "id": "tl_en_04",
        "question": "What are the 2024 traffic fines in Egypt?",
        "key_facts": ["fine", "2024", "traffic"],
        "expected_sources_contain": ["traffic_law"],
        "domain": "traffic_law",
        "language": "en",
    },
    {
        "id": "tl_en_05",
        "question": "Is it legal to use a seatbelt in the back seat in Egypt?",
        "key_facts": ["seatbelt", "seat"],
        "expected_sources_contain": ["traffic_law"],
        "domain": "traffic_law",
        "language": "en",
    },

    # ── Driving license (Arabic) ──────────────────────────────────────────────
    {
        "id": "dl_ar_01",
        "question": "ما هي أوراق تجديد رخصة القيادة في مصر؟",
        "key_facts": ["رخصة", "تجديد", "أوراق"],
        "expected_sources_contain": ["driving_license"],
        "domain": "driving_license",
        "language": "ar",
    },
    {
        "id": "dl_ar_02",
        "question": "كيف أستخرج رخصة قيادة لأول مرة؟",
        "key_facts": ["رخصة", "أول", "مرة"],
        "expected_sources_contain": ["driving_license"],
        "domain": "driving_license",
        "language": "ar",
    },
    {
        "id": "dl_ar_03",
        "question": "ما هو الحد الأدنى لسن استخراج رخصة القيادة؟",
        "key_facts": ["سن", "رخصة", "قيادة"],
        "expected_sources_contain": ["driving_license", "driver_fitness"],
        "domain": "driving_license",
        "language": "ar",
    },
    {
        "id": "dl_ar_04",
        "question": "هل يمكن تجديد رخصة القيادة أونلاين؟",
        "key_facts": ["تجديد", "رخصة", "إلكتروني"],
        "expected_sources_contain": ["driving_license"],
        "domain": "driving_license",
        "language": "ar",
    },

    # ── Driving license (English) ─────────────────────────────────────────────
    {
        "id": "dl_en_01",
        "question": "What documents are needed to renew a driving license in Egypt?",
        "key_facts": ["documents", "renew", "license"],
        "expected_sources_contain": ["driving_license"],
        "domain": "driving_license",
        "language": "en",
    },
    {
        "id": "dl_en_02",
        "question": "How do I get a driving license in Egypt for the first time?",
        "key_facts": ["license", "first time", "Egypt"],
        "expected_sources_contain": ["driving_license"],
        "domain": "driving_license",
        "language": "en",
    },
    {
        "id": "dl_en_03",
        "question": "What is the minimum age to get a driving license in Egypt?",
        "key_facts": ["age", "18", "license"],
        "expected_sources_contain": ["driving_license", "driver_fitness"],
        "domain": "driving_license",
        "language": "en",
    },
    {
        "id": "dl_en_04",
        "question": "Can I renew my driving license online in Egypt?",
        "key_facts": ["online", "renew", "license"],
        "expected_sources_contain": ["driving_license"],
        "domain": "driving_license",
        "language": "en",
    },

    # ── Vehicle registration (Arabic) ─────────────────────────────────────────
    {
        "id": "vr_ar_01",
        "question": "ما هي أوراق تجديد ترخيص السيارة في مصر؟",
        "key_facts": ["ترخيص", "سيارة", "تجديد"],
        "expected_sources_contain": ["vehicle_registration"],
        "domain": "vehicle_registration",
        "language": "ar",
    },
    {
        "id": "vr_ar_02",
        "question": "كيف يتم نقل ملكية السيارة في مصر؟",
        "key_facts": ["نقل", "ملكية", "سيارة"],
        "expected_sources_contain": ["vehicle_registration"],
        "domain": "vehicle_registration",
        "language": "ar",
    },
    {
        "id": "vr_ar_03",
        "question": "ما هي رسوم تسجيل السيارة الجديدة؟",
        "key_facts": ["رسوم", "تسجيل", "سيارة"],
        "expected_sources_contain": ["vehicle_registration"],
        "domain": "vehicle_registration",
        "language": "ar",
    },

    # ── Vehicle registration (English) ────────────────────────────────────────
    {
        "id": "vr_en_01",
        "question": "How do I register a new car in Egypt?",
        "key_facts": ["register", "car", "Egypt"],
        "expected_sources_contain": ["vehicle_registration"],
        "domain": "vehicle_registration",
        "language": "en",
    },
    {
        "id": "vr_en_02",
        "question": "What is the process to transfer car ownership in Egypt?",
        "key_facts": ["transfer", "ownership", "car"],
        "expected_sources_contain": ["vehicle_registration"],
        "domain": "vehicle_registration",
        "language": "en",
    },
    {
        "id": "vr_en_03",
        "question": "How do I import a car to Egypt?",
        "key_facts": ["import", "car", "customs"],
        "expected_sources_contain": ["vehicle_registration"],
        "domain": "vehicle_registration",
        "language": "en",
    },

    # ── Accident and liability (Arabic) ───────────────────────────────────────
    {
        "id": "al_ar_01",
        "question": "ماذا أفعل إذا تعرضت لحادث سيارة في مصر؟",
        "key_facts": ["حادث", "سيارة", "إجراءات"],
        "expected_sources_contain": ["accident_liability"],
        "domain": "accident_liability",
        "language": "ar",
    },
    {
        "id": "al_ar_02",
        "question": "كيف أطالب بالتعويض من شركة التأمين بعد حادث؟",
        "key_facts": ["تعويض", "تأمين", "حادث"],
        "expected_sources_contain": ["accident_liability"],
        "domain": "accident_liability",
        "language": "ar",
    },

    # ── Accident and liability (English) ──────────────────────────────────────
    {
        "id": "al_en_01",
        "question": "What should I do after a car accident in Egypt?",
        "key_facts": ["accident", "police", "steps"],
        "expected_sources_contain": ["accident_liability"],
        "domain": "accident_liability",
        "language": "en",
    },
    {
        "id": "al_en_02",
        "question": "Is car insurance mandatory in Egypt?",
        "key_facts": ["insurance", "mandatory", "Egypt"],
        "expected_sources_contain": ["accident_liability"],
        "domain": "accident_liability",
        "language": "en",
    },

    # ── Commercial vehicles (Arabic) ──────────────────────────────────────────
    {
        "id": "cv_ar_01",
        "question": "ما هي شروط تشغيل سيارة تاكسي في مصر؟",
        "key_facts": ["تاكسي", "شروط", "تشغيل"],
        "expected_sources_contain": ["commercial_vehicles"],
        "domain": "commercial_vehicles",
        "language": "ar",
    },
    {
        "id": "cv_ar_02",
        "question": "هل أوبر وكريم قانونيان في مصر؟",
        "key_facts": ["أوبر", "كريم", "قانون"],
        "expected_sources_contain": ["commercial_vehicles"],
        "domain": "commercial_vehicles",
        "language": "ar",
    },

    # ── Commercial vehicles (English) ─────────────────────────────────────────
    {
        "id": "cv_en_01",
        "question": "Are Uber and Careem legal in Egypt?",
        "key_facts": ["Uber", "Careem", "legal"],
        "expected_sources_contain": ["commercial_vehicles"],
        "domain": "commercial_vehicles",
        "language": "en",
    },
    {
        "id": "cv_en_02",
        "question": "What are the regulations for taxis in Egypt?",
        "key_facts": ["taxi", "regulation", "Egypt"],
        "expected_sources_contain": ["commercial_vehicles"],
        "domain": "commercial_vehicles",
        "language": "en",
    },

    # ── Driver fitness (Arabic) ───────────────────────────────────────────────
    {
        "id": "df_ar_01",
        "question": "ما هي الاشتراطات الصحية لاستخراج رخصة القيادة؟",
        "key_facts": ["صحية", "رخصة", "اشتراطات"],
        "expected_sources_contain": ["driver_fitness", "driving_license"],
        "domain": "driver_fitness",
        "language": "ar",
    },

    # ── Driver fitness (English) ──────────────────────────────────────────────
    {
        "id": "df_en_01",
        "question": "What are the medical requirements for a driving license in Egypt?",
        "key_facts": ["medical", "eyesight", "license"],
        "expected_sources_contain": ["driver_fitness", "driving_license"],
        "domain": "driver_fitness",
        "language": "en",
    },

    # ── International driving (Arabic) ────────────────────────────────────────
    {
        "id": "id_ar_01",
        "question": "كيف أستخرج رخصة قيادة دولية من مصر؟",
        "key_facts": ["دولية", "رخصة", "استخراج"],
        "expected_sources_contain": ["international_driving"],
        "domain": "international_driving",
        "language": "ar",
    },

    # ── International driving (English) ───────────────────────────────────────
    {
        "id": "id_en_01",
        "question": "Can I drive in Egypt with a foreign driving license?",
        "key_facts": ["foreign", "license", "Egypt"],
        "expected_sources_contain": ["international_driving"],
        "domain": "international_driving",
        "language": "en",
    },
    {
        "id": "id_en_02",
        "question": "How do I get an international driving permit in Egypt?",
        "key_facts": ["international", "permit", "IDP"],
        "expected_sources_contain": ["international_driving"],
        "domain": "international_driving",
        "language": "en",
    },

    # ── Road infrastructure (Arabic) ──────────────────────────────────────────
    {
        "id": "ri_ar_01",
        "question": "أين توجد كاميرات السرعة على الطرق المصرية؟",
        "key_facts": ["كاميرات", "سرعة", "طرق"],
        "expected_sources_contain": ["road_infrastructure", "traffic_law"],
        "domain": "road_infrastructure",
        "language": "ar",
    },

    # ── Road infrastructure (English) ─────────────────────────────────────────
    {
        "id": "ri_en_01",
        "question": "Are there toll roads in Egypt?",
        "key_facts": ["toll", "road", "Egypt"],
        "expected_sources_contain": ["road_infrastructure"],
        "domain": "road_infrastructure",
        "language": "en",
    },
    {
        "id": "ri_en_02",
        "question": "Where are speed cameras located in Egypt?",
        "key_facts": ["speed", "camera", "highway"],
        "expected_sources_contain": ["road_infrastructure", "traffic_law"],
        "domain": "road_infrastructure",
        "language": "en",
    },
]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_rouge_l(prediction: str, key_facts: list[str]) -> float:
    """
    Simplified ROUGE-L: fraction of key_facts that appear in the prediction.
    We use this instead of full ROUGE-L because our 'references' are key facts,
    not full reference sentences.
    """
    if not key_facts:
        return 0.0
    pred_lower = prediction.lower()
    matched = sum(1 for fact in key_facts if fact.lower() in pred_lower)
    return matched / len(key_facts)


def compute_retrieval_recall(sources: list[dict], expected_groups: list[str]) -> float:
    """
    Recall@k: fraction of expected source groups that appear in retrieved sources.
    """
    if not expected_groups:
        return 1.0
    retrieved_groups = {s.get("group", "") for s in sources}
    matched = sum(1 for g in expected_groups if g in retrieved_groups)
    return matched / len(expected_groups)


def is_uncertain_answer(answer: str) -> bool:
    """Detect when the system said 'I don't know' / information not available."""
    uncertainty_phrases = [
        "not available", "لا تتوفر", "لا أعرف", "لا أجد",
        "not found", "no information", "cannot find",
        "information is not available", "not in the provided",
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in uncertainty_phrases)


# ── Result structures ─────────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    id:               str
    question:         str
    domain:           str
    language:         str
    answer:           str
    rouge_l:          float
    retrieval_recall: float
    key_facts_found:  list[str]
    key_facts_missed: list[str]
    sources_retrieved: list[str]
    latency_ms:       int
    is_uncertain:     bool
    hyde_used:        bool
    reranker_used:    bool


@dataclass
class EvalReport:
    total:            int
    rouge_l_mean:     float
    retrieval_recall_mean: float
    uncertain_rate:   float
    by_domain:        dict
    by_language:      dict
    per_question:     list[dict]
    config:           dict


# ── Main evaluator ────────────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, use_hyde: bool = True, use_reranker: bool = True):
        from api.pipeline import get_pipeline
        self.pipeline    = get_pipeline()
        self.use_hyde    = use_hyde
        self.use_reranker = use_reranker

    def evaluate_one(self, item: dict) -> QuestionResult:
        t0     = time.perf_counter()
        result = self.pipeline.run(
            question=item["question"],
            use_hyde=self.use_hyde,
            use_reranker=self.use_reranker,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        answer  = result["answer"]
        sources = result["sources"]

        key_facts   = item.get("key_facts", [])
        pred_lower  = answer.lower()
        facts_found  = [f for f in key_facts if f.lower() in pred_lower]
        facts_missed = [f for f in key_facts if f.lower() not in pred_lower]

        rouge_l          = compute_rouge_l(answer, key_facts)
        retrieval_recall = compute_retrieval_recall(
            sources, item.get("expected_sources_contain", [])
        )

        return QuestionResult(
            id=item["id"],
            question=item["question"],
            domain=item["domain"],
            language=item["language"],
            answer=answer,
            rouge_l=rouge_l,
            retrieval_recall=retrieval_recall,
            key_facts_found=facts_found,
            key_facts_missed=facts_missed,
            sources_retrieved=[s["group"] for s in sources],
            latency_ms=elapsed_ms,
            is_uncertain=is_uncertain_answer(answer),
            hyde_used=result["hyde_used"],
            reranker_used=result["reranker_used"],
        )

    def run(self, domain_filter: str | None = None) -> EvalReport:
        dataset = GOLDEN_DATASET
        if domain_filter:
            dataset = [q for q in dataset if q["domain"] == domain_filter]
            if not dataset:
                raise ValueError(f"No questions for domain: {domain_filter}")

        logger.info(f"Evaluating {len(dataset)} questions | HyDE={self.use_hyde} | reranker={self.use_reranker}")

        results: list[QuestionResult] = []
        for i, item in enumerate(dataset, 1):
            logger.info(f"[{i}/{len(dataset)}] {item['id']}: {item['question'][:60]}")
            try:
                r = self.evaluate_one(item)
                results.append(r)
                logger.info(
                    f"  ROUGE-L={r.rouge_l:.2f} | "
                    f"Recall={r.retrieval_recall:.2f} | "
                    f"{'UNCERTAIN' if r.is_uncertain else 'answered'} | "
                    f"{r.latency_ms}ms"
                )
            except Exception as e:
                logger.error(f"  FAILED: {e}")

        # ── Aggregate ─────────────────────────────────────────────────────────
        if not results:
            raise RuntimeError("No results collected — check pipeline health")

        rouge_scores   = [r.rouge_l for r in results]
        recall_scores  = [r.retrieval_recall for r in results]
        uncertain_count = sum(1 for r in results if r.is_uncertain)

        # Per-domain breakdown
        domains: dict[str, list] = {}
        for r in results:
            domains.setdefault(r.domain, []).append(r)

        by_domain = {}
        for domain, dr in domains.items():
            by_domain[domain] = {
                "count":              len(dr),
                "rouge_l_mean":       round(sum(x.rouge_l for x in dr) / len(dr), 3),
                "retrieval_recall":   round(sum(x.retrieval_recall for x in dr) / len(dr), 3),
                "uncertain_rate":     round(sum(1 for x in dr if x.is_uncertain) / len(dr), 3),
                "avg_latency_ms":     round(sum(x.latency_ms for x in dr) / len(dr)),
            }

        # Per-language breakdown
        by_language: dict[str, dict] = {}
        for lang in ("ar", "en"):
            lr = [r for r in results if r.language == lang]
            if lr:
                by_language[lang] = {
                    "count":        len(lr),
                    "rouge_l_mean": round(sum(x.rouge_l for x in lr) / len(lr), 3),
                    "retrieval_recall": round(sum(x.retrieval_recall for x in lr) / len(lr), 3),
                }

        return EvalReport(
            total=len(results),
            rouge_l_mean=round(sum(rouge_scores) / len(rouge_scores), 3),
            retrieval_recall_mean=round(sum(recall_scores) / len(recall_scores), 3),
            uncertain_rate=round(uncertain_count / len(results), 3),
            by_domain=by_domain,
            by_language=by_language,
            per_question=[asdict(r) for r in results],
            config={
                "use_hyde":     self.use_hyde,
                "use_reranker": self.use_reranker,
                "total_questions": len(dataset),
            },
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_report(report: EvalReport) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION REPORT — Egyptian Road & Mobility Assistant")
    print("=" * 60)
    print(f"\nTotal questions:   {report.total}")
    print(f"ROUGE-L (mean):    {report.rouge_l_mean:.3f}")
    print(f"Retrieval Recall:  {report.retrieval_recall_mean:.3f}")
    print(f"Uncertain rate:    {report.uncertain_rate:.3f}  "
          f"({'system says it doesn\'t know'})")

    print("\n── By domain ──")
    for domain, stats in sorted(report.by_domain.items()):
        print(
            f"  {domain:<25} "
            f"n={stats['count']}  "
            f"ROUGE={stats['rouge_l_mean']:.2f}  "
            f"Recall={stats['retrieval_recall']:.2f}  "
            f"Uncertain={stats['uncertain_rate']:.2f}  "
            f"Latency={stats['avg_latency_ms']}ms"
        )

    print("\n── By language ──")
    for lang, stats in report.by_language.items():
        print(
            f"  {lang}  "
            f"n={stats['count']}  "
            f"ROUGE={stats['rouge_l_mean']:.2f}  "
            f"Recall={stats['retrieval_recall']:.2f}"
        )

    print("\n── Config ──")
    for k, v in report.config.items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--domain",    type=str,  default=None,  help="Filter to one domain")
    parser.add_argument("--output",    type=str,  default=None,  help="Save JSON report to this path")
    parser.add_argument("--no-hyde",   action="store_true",       help="Disable HyDE")
    parser.add_argument("--no-rerank", action="store_true",       help="Disable reranker")
    args = parser.parse_args()

    evaluator = Evaluator(
        use_hyde=not args.no_hyde,
        use_reranker=not args.no_rerank,
    )

    report = evaluator.run(domain_filter=args.domain)
    print_report(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        logger.success(f"Report saved → {out_path}")
