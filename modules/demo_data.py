"""
Curated example for the 5 bundled RAG sample papers.

IMPORTANT: this data was written BY HAND. It is NOT computed by the PaperPilot pipeline.
It is only shown when the user explicitly ticks "Show hand-written curated example" in the sidebar,
and the app labels it clearly. The old offline "heuristic engine" that lived in this file invented
contradictions, gaps and fixed scores, so it was removed. The honest offline engine now lives in
extraction.py, rules.py and hypothesis.py.
"""

import re
from typing import Dict, List, Any


PRECOMPUTED_PAPERS_ANALYSIS = {
    "claims": [
        # Paper 1 Claims
        {
            "id": "CLM_P1_1",
            "paper_id": "P1",
            "paper_title": "Dense Retrieval vs. Graph RAG in Medical QA",
            "topic": "Hallucination Reduction",
            "statement": "Knowledge-graph-augmented retrieval reduces medical factual hallucinations by 34.2% compared to standard dense vector retrieval.",
            "quote": "In this empirical study across 10,000 PubMed QA pairs, we demonstrate that Graph-augmented retrieval reduces medical factual hallucination by 34.2% compared to standard dense vector retrieval.",
            "section": "Key Empirical Findings & Claims",
            "confidence": 0.96,
            "methodology": "Empirical benchmark on 10,000 PubMed QA pairs using Neo4j over 2.4M articles with GPT-4 and Med-PaLM-2."
        },
        {
            "id": "CLM_P1_2",
            "paper_id": "P1",
            "paper_title": "Dense Retrieval vs. Graph RAG in Medical QA",
            "topic": "Multi-hop Reasoning",
            "statement": "Dense vector embeddings collapse in multi-hop diagnostic reasoning across disparate medical records, exhibiting a 42% accuracy drop on 3-hop queries.",
            "quote": "Vector embeddings alone struggle with multi-hop diagnostic reasoning across disparate patient history nodes, showing a 42% accuracy collapse on 3-hop queries.",
            "section": "Key Empirical Findings & Claims",
            "confidence": 0.91,
            "methodology": "Comparative multi-hop diagnostic path traversal against Dense Passage Retrieval (DPR)."
        },
        {
            "id": "CLM_P1_3",
            "paper_id": "P1",
            "paper_title": "Dense Retrieval vs. Graph RAG in Medical QA",
            "topic": "System Latency & Cost",
            "statement": "Structured knowledge graph retrieval introduces a 4.1x latency penalty (mean 420ms vs 102ms), challenging real-time bedside deployment.",
            "quote": "Knowledge graph retrieval introduces a 4.1x latency penalty, making real-time clinical deployment cost-prohibitive on commodity hardware.",
            "section": "Key Empirical Findings & Claims",
            "confidence": 0.94,
            "methodology": "Wall-clock latency benchmarking on commodity server hardware."
        },

        # Paper 2 Claims
        {
            "id": "CLM_P2_1",
            "paper_id": "P2",
            "paper_title": "Can Scaling Vector Chunk Size Replace Graph Structures?",
            "topic": "Retrieval Accuracy & F1",
            "statement": "Hierarchical chunking (1024 tokens) with hybrid BM25-dense re-ranking matches or exceeds Graph RAG factuality scores without structured graph overhead.",
            "quote": "Hierarchical chunking with 1024-token context windows and hybrid BM25-dense re-ranking matches or exceeds Graph RAG factuality scores without structured graph overhead (F1: 88.4% vs 87.9%).",
            "section": "Experimental Results & Claims",
            "confidence": 0.93,
            "methodology": "Ablation over 12 retrieval configurations on MIMIC-IV clinical discharge summaries and BioASQ."
        },
        {
            "id": "CLM_P2_2",
            "paper_id": "P2",
            "paper_title": "Can Scaling Vector Chunk Size Replace Graph Structures?",
            "topic": "Inference Latency",
            "statement": "Dense vector search achieves sub-85ms latency while retaining 96% of key clinical facts, bypassing Graph RAG's latency penalty.",
            "quote": "Graph RAG's reported latency penalty is avoidable; vector search achieves < 85ms inference while retaining 96% of key clinical facts.",
            "section": "Experimental Results & Claims",
            "confidence": 0.95,
            "methodology": "Comparative latency benchmarking with BGE-Large embeddings and Cohere Rerank-v3."
        },
        {
            "id": "CLM_P2_3",
            "paper_id": "P2",
            "paper_title": "Can Scaling Vector Chunk Size Replace Graph Structures?",
            "topic": "Graph Entity Disambiguation Errors",
            "statement": "Automated knowledge graph entity disambiguation errors introduce false-positive hallucination risks that outweigh structural gains.",
            "quote": "Entity disambiguation errors in automated knowledge graphs introduce false-positive hallucination risks that outweigh structural gains.",
            "section": "Experimental Results & Claims",
            "confidence": 0.88,
            "methodology": "Error analysis of false-positive relational edges generated by automated entity-linking pipelines."
        },

        # Paper 3 Claims
        {
            "id": "CLM_P3_1",
            "paper_id": "P3",
            "paper_title": "Self-Consistency and Chain-of-Thought as Intrinsic Filters",
            "topic": "Test-Time Compute vs Retrieval",
            "statement": "Sampling multiple reasoning paths (Self-Consistency k=10) reduces reasoning hallucinations by 28.5% purely at generation time without any external retrieval index.",
            "quote": "Sampling multiple reasoning paths (Self-Consistency k=10) reduces reasoning hallucinations by 28.5% purely at generation time without any external retrieval index.",
            "section": "Core Claims & Observations",
            "confidence": 0.95,
            "methodology": "Tested LLaMA-2-70B and Claude-2 on MedQA and USMLE Step 1 with k=10 majority voting."
        },
        {
            "id": "CLM_P3_2",
            "paper_id": "P3",
            "paper_title": "Self-Consistency and Chain-of-Thought as Intrinsic Filters",
            "topic": "Context Poisoning",
            "statement": "External retrieval indices pollute model reasoning when retrieved documents contain conflicting medical consensus, dropping coherence by 17.4%.",
            "quote": "External retrieval indices pollute model reasoning when retrieved documents contain conflicting medical consensus, dropping reasoning coherence by 17.4%.",
            "section": "Core Claims & Observations",
            "confidence": 0.90,
            "methodology": "Inoculation experiments introducing conflicting PubMed abstracts into the prompt context."
        },
        {
            "id": "CLM_P3_3",
            "paper_id": "P3",
            "paper_title": "Self-Consistency and Chain-of-Thought as Intrinsic Filters",
            "topic": "Parametric Memory Reliability",
            "statement": "Parametric memory in models > 70B parameters is more consistent than uncurated web retrieval on established foundational science.",
            "quote": "Parametric memory in models > 70B parameters is more consistent than uncurated web retrieval on established foundational science.",
            "section": "Core Claims & Observations",
            "confidence": 0.87,
            "methodology": "Zero-shot parametric vs web-retrieved RAG evaluation across standard biomedical curricula."
        },

        # Paper 4 Claims
        {
            "id": "CLM_P4_1",
            "paper_id": "P4",
            "paper_title": "Adaptive Hybrid Retrieval under Uncertainty",
            "topic": "Entropy Gating & Routing",
            "statement": "Dynamic entropy gating routes queries to internal parametric memory when confidence is > 0.92, invoking hybrid dense-graph retrieval only on high-uncertainty prompts.",
            "quote": "Dynamic entropy gating routes queries to internal parametric memory when confidence is > 0.92, invoking hybrid dense-graph retrieval only on high-uncertainty prompts.",
            "section": "Experimental Evaluation & Claims",
            "confidence": 0.94,
            "methodology": "Token entropy and attention dispersion monitoring with dynamic threshold routing."
        },
        {
            "id": "CLM_P4_2",
            "paper_id": "P4",
            "paper_title": "Adaptive Hybrid Retrieval under Uncertainty",
            "topic": "Latency vs Accuracy Pareto Frontier",
            "statement": "Hybrid uncertainty gating cuts overall query latency by 58% while maintaining 91.2% factual correctness across diverse benchmark suites.",
            "quote": "Hybrid gating cuts overall query latency by 58% while maintaining 91.2% factual correctness across diverse benchmark suites.",
            "section": "Experimental Evaluation & Claims",
            "confidence": 0.92,
            "methodology": "Multi-dataset benchmark on HotpotQA, MedQA, and StrategyQA."
        },
        {
            "id": "CLM_P4_3",
            "paper_id": "P4",
            "paper_title": "Adaptive Hybrid Retrieval under Uncertainty",
            "topic": "Context Poisoning Mitigation",
            "statement": "Selective retrieval prevents context poisoning observed in unconstrained RAG pipelines by suppressing retrieval when internal consensus is robust.",
            "quote": "Selective retrieval prevents context poisoning observed in unconstrained RAG pipelines by suppressing retrieval when internal consensus is robust.",
            "section": "Experimental Evaluation & Claims",
            "confidence": 0.89,
            "methodology": "Controlled benchmark with adversarial and noisy distractors."
        },

        # Paper 5 Claims
        {
            "id": "CLM_P5_1",
            "paper_id": "P5",
            "paper_title": "Long-Context LLMs vs. RAG: Empirical Audit",
            "topic": "Lost-in-the-Middle Degradation",
            "statement": "Native long-context models (1M+ tokens) exhibit severe factual degradation when evidence is buried beyond the 60th percentile of context depth, dropping accuracy from 84% to 51%.",
            "quote": "Native long-context models (1M+ tokens) exhibit 'lost-in-the-middle' factual degradation when evidence is buried beyond the 60th percentile of context depth (accuracy drops from 84% to 51%).",
            "section": "Claims & Findings",
            "confidence": 0.97,
            "methodology": "'Needle-in-Multi-Study' benchmark consisting of 50 full-length clinical trials (450k tokens total) on Gemini 1.5 Pro and Claude 3.5 Sonnet."
        },
        {
            "id": "CLM_P5_2",
            "paper_id": "P5",
            "paper_title": "Long-Context LLMs vs. RAG: Empirical Audit",
            "topic": "RAG Necessity in Long Context",
            "statement": "Targeted retrieval-augmented generation remains strictly necessary for needle-in-haystack factual precision regardless of context window expansion.",
            "quote": "Targeted retrieval-augmented generation remains strictly necessary for needle-in-haystack factual precision regardless of context window expansion.",
            "section": "Claims & Findings",
            "confidence": 0.95,
            "methodology": "Comparative evaluation of brute-force context stuffing vs targeted chunked hybrid RAG."
        },
        {
            "id": "CLM_P5_3",
            "paper_id": "P5",
            "paper_title": "Long-Context LLMs vs. RAG: Empirical Audit",
            "topic": "Attention Dispersion & Hallucination",
            "statement": "Attention dispersion across 100k+ tokens increases hallucination rate by 19% compared to focused 4k-token chunk retrieval.",
            "quote": "Attention dispersion across 100k+ tokens increases hallucination rate by 19% compared to focused 4k-token chunk retrieval.",
            "section": "Claims & Findings",
            "confidence": 0.91,
            "methodology": "Attention map visualization and hallucination attribution on long multi-document reasoning."
        }
    ],

    "contradictions": [
        {
            "id": "CONT_1",
            "topic": "Graph RAG Necessity vs. Hierarchical Vector Chunking",
            "severity": "Direct Contradiction",
            "claim_a": {
                "paper": "Paper 1 (Vance et al.)",
                "statement": "Knowledge-graph RAG is required to eliminate vector collapse on multi-hop reasoning, reducing hallucinations by 34.2%.",
                "quote": "Graph-augmented retrieval reduces medical factual hallucination by 34.2% compared to standard dense vector retrieval... showing a 42% accuracy collapse on 3-hop queries."
            },
            "claim_b": {
                "paper": "Paper 2 (Mehta et al.)",
                "statement": "Hierarchical chunking (1024 tokens) with hybrid BM25-dense re-ranking matches or beats Graph RAG (F1: 88.4% vs 87.9%) without graph overhead or entity disambiguation errors.",
                "quote": "Hierarchical chunking with 1024-token context windows and hybrid BM25-dense re-ranking matches or exceeds Graph RAG factuality scores without structured graph overhead (F1: 88.4% vs 87.9%)."
            },
            "root_cause": "Different benchmark datasets (PubMed QA multi-hop relations vs MIMIC-IV clinical summaries) and differing treatment of entity disambiguation noise in automated graph construction.",
            "resolution_path": "Benchmark both systems on a unified multi-hop clinical benchmark while evaluating varying entity relational density."
        },
        {
            "id": "CONT_2",
            "topic": "External RAG Ingestion vs. Pure Intrinsic Parametric Reasoning",
            "severity": "Direct Contradiction",
            "claim_a": {
                "paper": "Papers 1 & 2 (Vance et al. & Mehta et al.)",
                "statement": "External retrieval is essential to ground clinical models and prevent fatal medical hallucinations.",
                "quote": "Retrieval-Augmented Generation (RAG) is increasingly deployed in clinical decision support... targeted retrieval remains strictly necessary."
            },
            "claim_b": {
                "paper": "Paper 3 (Reynolds & Zhang)",
                "statement": "External retrieval introduces context poisoning from conflicting consensus, whereas test-time Self-Consistency Chain-of-Thought (k=10) reduces hallucinations by 28.5% purely intrinsically.",
                "quote": "External retrieval indices pollute model reasoning when retrieved documents contain conflicting medical consensus, dropping reasoning coherence by 17.4%."
            },
            "root_cause": "Assumption of corpus quality: Vance & Mehta assume curated factual retrieval indices, whereas Reynolds & Zhang test uncurated environments with conflicting consensus.",
            "resolution_path": "Implement uncertainty gating that only accesses external RAG when parametric model entropy indicates low confidence."
        },
        {
            "id": "CONT_3",
            "topic": "1M+ Long-Context Feasibility vs. Targeted Chunked RAG",
            "severity": "Methodological Tension",
            "claim_a": {
                "paper": "Industry Assumption / Long-Context Proponents",
                "statement": "Context window scaling to 1M+ tokens eliminates the need for complex chunking and retrieval pipelines.",
                "quote": "With context windows expanding to millions of tokens, industry claims suggest that RAG architectures are obsolete."
            },
            "claim_b": {
                "paper": "Paper 5 (Tanaka & O'Connor)",
                "statement": "Long context causes severe 'lost-in-the-middle' accuracy collapse (84% down to 51%) beyond the 60th percentile, making targeted RAG indispensable.",
                "quote": "Native long-context models (1M+ tokens) exhibit 'lost-in-the-middle' factual degradation when evidence is buried beyond the 60th percentile of context depth."
            },
            "root_cause": "Attention dispersion across 100k+ tokens dilutes query-relevant attention heads, causing positional degradation that chunked RAG avoids.",
            "resolution_path": "Integrate hierarchical chunk-level indexing within long-context caching windows."
        }
    ],

    "research_gaps": [
        {
            "id": "GAP_1",
            "title": "Continuous Real-Time Streaming Subgraph Ingestion without Latency Spikes",
            "category": "Deployment & Operational Constraint",
            "source_papers": ["Paper 1 (Vance et al.)", "Paper 4 (Al-Mansoor et al.)"],
            "description": "Current Graph RAG requires static offline graph indexing (e.g., Neo4j batches over PubMed). In live clinical or emergency care, medical updates occur continuously. There is no existing framework to dynamically update knowledge subgraphs in real-time without re-indexing or causing latency spikes above 400ms.",
            "evidence_quote": "Existing graph RAG frameworks cannot ingest real-time streaming clinical updates without full index recalculation.",
            "impact_score": "Critical",
            "suggested_direction": "Investigate incremental streaming graph neural networks (GNNs) with dynamic edge insertion and caching."
        },
        {
            "id": "GAP_2",
            "title": "Automated Arbitration under Conflicting Scientific Consensus",
            "category": "Theoretical & Architectural Boundary",
            "source_papers": ["Paper 3 (Reynolds & Zhang)", "Paper 4 (Al-Mansoor et al.)"],
            "description": "When retrieved external documents report conflicting clinical trial conclusions (e.g. drug efficacy discrepancies), current RAG systems suffer from context poisoning and hallucinatory compromise. There is no automated framework to detect conflicting consensus and apply epistemic arbitration before feeding context to the generator.",
            "evidence_quote": "External retrieval indices pollute model reasoning when retrieved documents contain conflicting medical consensus, dropping reasoning coherence by 17.4%.",
            "impact_score": "High",
            "suggested_direction": "Design a dialectical multi-agent verification layer that extracts contradictory claims and scores their methodological evidence hierarchy before generation."
        },
        {
            "id": "GAP_3",
            "title": "Out-of-Distribution Calibration for Entropy-Gated Uncertainty Routing",
            "category": "Methodological Limitation",
            "source_papers": ["Paper 4 (Al-Mansoor et al.)"],
            "description": "Dynamic entropy gating effectively cuts latency by 58% when routing known queries, but uncertainty calibration is tuned on static validation splits. In out-of-distribution (OOD) medical specialties or novel pathologies, model confidence is poorly calibrated, risking unretrieved hallucinations.",
            "evidence_quote": "How to adapt uncertainty thresholds in out-of-distribution clinical scenarios remains an open research challenge.",
            "impact_score": "High",
            "suggested_direction": "Develop conformal prediction bounds and Bayesian test-time uncertainty estimators that dynamically adapt routing thresholds."
        },
        {
            "id": "GAP_4",
            "title": "Positional Attention Decay in Extreme-Scale Multi-Document Reasoning",
            "category": "Empirical & Evaluation Deficit",
            "source_papers": ["Paper 5 (Tanaka et al.)", "Paper 2 (Mehta et al.)"],
            "description": "While long context fails beyond the 60th percentile ('lost-in-the-middle'), the exact micro-architectural mechanism of attention dispersion during multi-hop reasoning over 100k+ tokens remains uncharacterized.",
            "evidence_quote": "The interaction between long-context attention heads and cached key-value states during iterative reasoning is poorly understood.",
            "impact_score": "Moderate",
            "suggested_direction": "Conduct fine-grained attention-head probe audits and test rotary position embedding (RoPE) recalibration for deep context."
        }
    ],

    "hypotheses": [
        {
            "id": "HYP_1",
            "title": "Entropy-Gated Streaming Subgraph Expansion (EG-SSE)",
            "formulation": "If dynamic token entropy routing (tau > 0.08) is coupled with a micro-cached streaming edge-insertion algorithm, then clinical QA hallucination will drop by >30% while retaining sub-120ms latency, resolving the trade-off between Graph RAG accuracy and vector speed.",
            "addresses_gap": "GAP_1",
            "resolves_contradiction": "CONT_1",
            "rationale": "Combines Al-Mansoor's entropy routing with Vance's multi-hop graph traversal, only expanding subgraphs for high-entropy tokens to circumvent the 4.1x latency bottleneck.",
            "testability_score": "High",
            "proposed_methodology": "Deploy on MIMIC-IV and PubMed QA using streaming Redis-Graph caching with LLaMA-3-70B; measure F1, Latency (p95), and Hallucination Rate."
        },
        {
            "id": "HYP_2",
            "title": "Dialectical Multi-Agent Consensus Filter for Context Poisoning Defense",
            "formulation": "If retrieved candidate passages undergo an autonomous pairwise contradiction cross-audit before prompt injection, then context poisoning from conflicting trial consensus will decrease by >40% without sacrificing retrieval recall.",
            "addresses_gap": "GAP_2",
            "resolves_contradiction": "CONT_2",
            "rationale": "Directly tackles Reynolds & Zhang's finding that conflicting retrieved consensus causes a 17.4% drop in reasoning coherence by filtering or explicitly flagging contradictory claims prior to generation.",
            "testability_score": "High",
            "proposed_methodology": "Synthetic inoculation benchmark using contradictory medical guidelines; evaluate reasoning coherence and factual precision."
        },
        {
            "id": "HYP_3",
            "title": "Conformal Uncertainty Calibration for Robust Clinical Routing",
            "formulation": "If conformal prediction sets are computed over intermediate transformer layer representations, then out-of-distribution retrieval gating will maintain <5% unretrieved hallucination error across unseen medical domains.",
            "addresses_gap": "GAP_3",
            "resolves_contradiction": "CONT_2",
            "rationale": "Addresses Al-Mansoor's stated limitation regarding OOD uncertainty miscalibration by providing mathematically provable confidence coverage bounds.",
            "testability_score": "Medium",
            "proposed_methodology": "Train conformal calibration on cardiology and test zero-shot on oncology and rare genetic disorders."
        },
        {
            "id": "HYP_4",
            "title": "Hierarchical Anchor RAG for Lost-in-the-Middle Mitigation",
            "formulation": "If 1024-token hierarchical chunk vectors are inserted as soft anchor prompt tokens in 1M+ token contexts, then lost-in-the-middle degradation beyond the 60th percentile will be reduced from 33% to under 6%.",
            "addresses_gap": "GAP_4",
            "resolves_contradiction": "CONT_3",
            "rationale": "Unifies Mehta's 1024-token hierarchical chunking with Tanaka's long-context audit, using vector representations as attention beacons.",
            "testability_score": "High",
            "proposed_methodology": "Test on Tanaka's Needle-in-Multi-Study benchmark across context lengths from 100k to 1M tokens."
        }
    ],

    "verifications": {
        "HYP_1": {
            "hypothesis_id": "HYP_1",
            "overall_grounding_score": 92,
            "feasibility_rating": "Highly Grounded",
            "supporting_evidence": [
                {
                    "paper_title": "Paper 4 (Al-Mansoor et al.)",
                    "quote": "Dynamic entropy gating routes queries to internal parametric memory when confidence is > 0.92, invoking hybrid dense-graph retrieval only on high-uncertainty prompts.",
                    "relevance": "Validates that entropy routing cuts latency by 58% while maintaining 91.2% accuracy."
                },
                {
                    "paper_title": "Paper 1 (Vance et al.)",
                    "quote": "Graph-augmented retrieval reduces medical factual hallucination by 34.2% compared to standard dense vector retrieval.",
                    "relevance": "Confirms the substantial factual advantage of graph traversal for multi-hop clinical queries."
                }
            ],
            "contradicting_or_limiting_evidence": [
                {
                    "paper_title": "Paper 1 (Vance et al.)",
                    "quote": "Existing graph RAG frameworks cannot ingest real-time streaming clinical updates without full index recalculation.",
                    "limitation": "Warns that streaming edge insertion requires specialized dynamic graph infrastructure."
                },
                {
                    "paper_title": "Paper 2 (Mehta et al.)",
                    "quote": "Entity disambiguation errors in automated knowledge graphs introduce false-positive hallucination risks.",
                    "limitation": "Streaming graph updates risk compounding entity disambiguation noise."
                }
            ],
            "falsification_criteria": "If latency exceeds 150ms or if streaming graph noise reduces F1 below Mehta's 88.4% hierarchical chunking baseline.",
            "synthesis_verdict": "Strong theoretical and empirical grounding across Papers 1, 2, and 4. Highly promising architecture for next-generation clinical AI."
        },
        "HYP_2": {
            "hypothesis_id": "HYP_2",
            "overall_grounding_score": 89,
            "feasibility_rating": "Highly Grounded",
            "supporting_evidence": [
                {
                    "paper_title": "Paper 3 (Reynolds & Zhang)",
                    "quote": "External retrieval indices pollute model reasoning when retrieved documents contain conflicting medical consensus, dropping reasoning coherence by 17.4%.",
                    "relevance": "Directly demonstrates the acute vulnerability that the dialectical consensus filter is engineered to solve."
                },
                {
                    "paper_title": "Paper 4 (Al-Mansoor et al.)",
                    "quote": "Selective retrieval prevents context poisoning observed in unconstrained RAG pipelines by suppressing retrieval when internal consensus is robust.",
                    "relevance": "Corroborates that filtering low-quality or contradictory retrieved context preserves model integrity."
                }
            ],
            "contradicting_or_limiting_evidence": [
                {
                    "paper_title": "Paper 3 (Reynolds & Zhang)",
                    "quote": "Parametric memory in models > 70B parameters is more consistent than uncurated web retrieval on established foundational science.",
                    "limitation": "For foundational science, skipping retrieval entirely via self-consistency may be cheaper than multi-agent pairwise audits."
                }
            ],
            "falsification_criteria": "If dialectical multi-agent filtering introduces more than 80ms overhead without reducing conflicting consensus hallucinations by at least 25%.",
            "synthesis_verdict": "Rigorous grounding in observed context-poisoning failure modes. Solves an acknowledged blind spot in clinical RAG."
        },
        "HYP_3": {
            "hypothesis_id": "HYP_3",
            "overall_grounding_score": 84,
            "feasibility_rating": "Plausible / Strongly Grounded",
            "supporting_evidence": [
                {
                    "paper_title": "Paper 4 (Al-Mansoor et al.)",
                    "quote": "How to adapt uncertainty thresholds in out-of-distribution clinical scenarios remains an open research challenge.",
                    "relevance": "Directly targets the explicit open problem formulated by Al-Mansoor et al."
                }
            ],
            "contradicting_or_limiting_evidence": [
                {
                    "paper_title": "Paper 3 (Reynolds & Zhang)",
                    "quote": "Self-consistency fails on zero-shot niche knowledge outside the training cut-off.",
                    "limitation": "If internal representations lack domain knowledge, conformal prediction sets may become trivially uninformative."
                }
            ],
            "falsification_criteria": "If conformal calibration produces prediction set sizes that exceed actionable clinical utility (>3 candidate paths for >50% of cases).",
            "synthesis_verdict": "Statistically sound formulation bridging uncertainty theory with retrieval decision-making."
        },
        "HYP_4": {
            "hypothesis_id": "HYP_4",
            "overall_grounding_score": 88,
            "feasibility_rating": "Highly Grounded",
            "supporting_evidence": [
                {
                    "paper_title": "Paper 5 (Tanaka & O'Connor)",
                    "quote": "Native long-context models (1M+ tokens) exhibit 'lost-in-the-middle' factual degradation when evidence is buried beyond the 60th percentile of context depth (accuracy drops from 84% to 51%).",
                    "relevance": "Defines the exact failure phenomenon and quantitative baseline that anchor tokens seek to repair."
                },
                {
                    "paper_title": "Paper 2 (Mehta et al.)",
                    "quote": "Hierarchical chunking with 1024-token context windows and hybrid BM25-dense re-ranking matches or exceeds Graph RAG factuality scores.",
                    "relevance": "Proves that 1024-token chunks preserve structural coherence and factual precision."
                }
            ],
            "contradicting_or_limiting_evidence": [
                {
                    "paper_title": "Paper 5 (Tanaka & O'Connor)",
                    "quote": "Attention dispersion across 100k+ tokens increases hallucination rate by 19% compared to focused 4k-token chunk retrieval.",
                    "limitation": "Anchor prompt tokens may themselves contribute to attention dispersion if not masked properly."
                }
            ],
            "falsification_criteria": "If anchor tokens fail to improve deep-context needle retrieval beyond 65% accuracy on 1M token benchmarks.",
            "synthesis_verdict": "Direct synthesis of Paper 2 and Paper 5 findings, offering an elegant test-time context architecture."
        }
    }
}
