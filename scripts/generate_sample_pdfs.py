"""
Script to generate 5 high-caliber realistic research paper PDFs for PaperPilot demo and testing.
Uses ReportLab to format title, authors, abstract, methodology, results, claims, and discussion.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle

PAPERS_SPEC = [
    {
        "filename": "Paper_1_Dense_vs_Graph_RAG.pdf",
        "title": "Dense Retrieval vs. Graph RAG: Evaluating Factuality and Hallucination in Complex Medical Question Answering",
        "authors": "Dr. Elena Vance, Dr. Marcus Thorne, Dr. Julian Rivera",
        "affiliation": "Institute for Clinical Intelligence & Neural Systems, Boston, MA",
        "year": "2024",
        "abstract": "Retrieval-Augmented Generation (RAG) is increasingly deployed in clinical decision support. While dense vector retrieval models (such as Contriever and DPR) are standard, their capacity for multi-hop relational inference remains contested. In this empirical study across 10,000 PubMed QA pairs, we demonstrate that Graph-augmented retrieval reduces medical factual hallucination by 34.2% compared to standard dense vector retrieval. However, structured knowledge graph traversal incurs a substantial 4.1x latency penalty (mean 420ms vs 102ms), challenging real-time bedside deployment. We provide statistical bounds and analyze failure modes where vector embeddings lose relational context across disparate medical records.",
        "sections": [
            ("1. Introduction", 
             "Large language models (LLMs) continue to suffer from catastrophic factual hallucinations in clinical medicine. Standard dense retrieval encodes documents into continuous latent vectors, measuring semantic proximity. However, biomedical relationships often require explicit symbolic traversal (e.g., drug-interaction pathways)."),
            ("2. Methodology & Experimental Setup", 
             "We constructed an enterprise biomedical knowledge graph using Neo4j over 2.4 million PubMed articles. We compared Dense Passage Retrieval (DPR) with our hybrid GraphRAG pipeline using GPT-4 and Med-PaLM-2 as generation backbones. Evaluation metrics include Factual Precision, Hallucination Rate (HR), and Inference Latency across 10,000 queries."),
            ("3. Key Empirical Findings & Claims", 
             "Finding 1: Graph-augmented retrieval reduces medical factual hallucination by 34.2% compared to standard dense vector retrieval.\n"
             "Finding 2: Vector embeddings alone struggle with multi-hop diagnostic reasoning across disparate patient history nodes, showing a 42% accuracy collapse on 3-hop queries.\n"
             "Finding 3: Knowledge graph retrieval introduces a 4.1x latency penalty, making real-time clinical deployment cost-prohibitive on commodity hardware."),
            ("4. Limitations & Research Gaps", 
             "Our study is strictly restricted to static English medical literature. We identify a critical gap: existing graph RAG frameworks cannot ingest real-time streaming clinical updates without full index recalculation. Furthermore, multi-modal medical imaging was excluded from graph nodes."),
            ("5. Conclusion", 
             "While Graph RAG establishes superior factuality bounds for complex medical queries, the engineering overhead and latency overhead necessitate hybrid routing mechanisms.")
        ]
    },
    {
        "filename": "Paper_2_Scaling_Vector_Chunk_Size.pdf",
        "title": "Can Scaling Vector Chunk Size Replace Graph Structures in Retrieval Systems?",
        "authors": "Dr. Arjun Mehta, Dr. Sophia Lin, Dr. Kevin Patel",
        "affiliation": "Department of Computer Science, Stanford AI Laboratory",
        "year": "2024",
        "abstract": "Recent literature advocates for structured knowledge graphs to resolve multi-hop reasoning deficits in RAG. We challenge this premise. Through systematic ablations across MIMIC-IV clinical notes and BioASQ datasets, we demonstrate that hierarchical chunking with 1024-token context windows and hybrid BM25-dense re-ranking matches or exceeds Graph RAG factuality scores without structured graph overhead (F1: 88.4% vs 87.9%). Furthermore, dense vector search achieves under 85ms inference while retaining 96% of key clinical facts. We also discover that automated graph entity disambiguation errors introduce false-positive hallucination risks that outweigh structural gains.",
        "sections": [
            ("1. Introduction", 
             "The push toward complex Knowledge Graph RAG systems introduces immense maintenance friction and compute latency. We investigate whether sophisticated chunking strategies and hybrid lexical-dense ranking can match structured graphs without graph engineering."),
            ("2. Methodology", 
             "We evaluate 12 retrieval configurations on MIMIC-IV discharge summaries. We implement Hierarchical Chunking (256, 512, 1024 tokens) paired with BGE-Large embeddings and Cohere Rerank-v3, benchmarking against Neo4j-based GraphRAG architectures."),
            ("3. Experimental Results & Claims", 
             "Finding 1: Hierarchical chunking with 1024-token context windows and hybrid BM25-dense re-ranking matches or exceeds Graph RAG factuality scores without structured graph overhead (F1: 88.4% vs 87.9%).\n"
             "Finding 2: Graph RAG's reported latency penalty is avoidable; vector search achieves < 85ms inference while retaining 96% of key clinical facts.\n"
             "Finding 3: Entity disambiguation errors in automated knowledge graphs introduce false-positive hallucination risks that outweigh structural gains."),
            ("4. Limitations & Research Gaps", 
             "Our benchmarks focused on clinical summaries with moderate entity density. In domains with extreme relational sparseness (e.g., genomics), vector search may degrade. We note a lack of standardized synthetic benchmarks that isolate topological vs semantic retrieval."),
            ("5. Conclusion", 
             "Before adopting computationally heavy graph databases, practitioners should exhaust hierarchical chunking with cross-encoder re-ranking, which delivers competitive factuality with 5x lower operational latency.")
        ]
    },
    {
        "filename": "Paper_3_Self_Consistency_Intrinsic_Reasoning.pdf",
        "title": "Self-Consistency and Chain-of-Thought as Intrinsic Hallucination Filters",
        "authors": "Dr. David K. Reynolds, Dr. Claire Zhang",
        "affiliation": "Cognitive AI Systems, Oxford Deep Research",
        "year": "2023",
        "abstract": "External retrieval-augmented pipelines are widely regarded as the primary defense against language model hallucinations. In this work, we demonstrate that test-time compute via Self-Consistency Chain-of-Thought (CoT) provides an orthogonal and frequently superior hallucination reduction mechanism. Sampling multiple reasoning paths (k=10) reduces reasoning hallucinations by 28.5% purely at generation time without any external retrieval index. Crucially, we find that external retrieval indices pollute model reasoning when retrieved documents contain conflicting medical consensus or contradictory trial data. Parametric memory in models exceeding 70B parameters is more consistent than uncurated web retrieval.",
        "sections": [
            ("1. Introduction", 
             "RAG assumes that external corpora are ground truth. However, in emerging scientific domains, retrieved passages often present contradictory evidence, actively misleading the generator. We analyze whether intrinsic reasoning paths can filter hallucinations without retrieval reliance."),
            ("2. Methodology", 
             "We test LLaMA-2-70B and Claude-2 across MedQA and USMLE Step 1 examinations under two regimes: (a) External RAG with top-5 retrieved passages, and (b) Parametric Self-Consistency with k=10 sampled reasoning paths with majority voting."),
            ("3. Core Claims & Observations", 
             "Finding 1: Sampling multiple reasoning paths (Self-Consistency k=10) reduces reasoning hallucinations by 28.5% purely at generation time without any external retrieval index.\n"
             "Finding 2: External retrieval indices pollute model reasoning when retrieved documents contain conflicting medical consensus, dropping reasoning coherence by 17.4%.\n"
             "Finding 3: Parametric memory in models > 70B parameters is more consistent than uncurated web retrieval on established foundational science."),
            ("4. Identified Gaps", 
             "Self-consistency fails on zero-shot niche knowledge outside the training cut-off. There is no automated framework to determine dynamically whether a query is solvable via parametric memory versus requiring retrieval."),
            ("5. Summary", 
             "Intrinsic reasoning compute should not be abandoned in favor of naive retrieval. A resilient architecture must arbitrate between intrinsic verification and retrieval ingestion.")
        ]
    },
    {
        "filename": "Paper_4_Adaptive_Hybrid_Retrieval.pdf",
        "title": "Adaptive Hybrid Retrieval: Balancing Parametric and Non-Parametric Memory under Uncertainty",
        "authors": "Dr. Tariq Al-Mansoor, Dr. Rachel Green, Dr. Liam Chen",
        "affiliation": "Zurich AI Research & ETH Zurich",
        "year": "2025",
        "abstract": "The dichotomy between purely parametric generation and unconstrained RAG forces practitioners to trade off latency, cost, and hallucination rates. We introduce Adaptive Hybrid Retrieval (AHR), a dynamic entropy-gated framework that routes queries between internal parametric reasoning, dense vector search, and structured graph traversal based on model prediction uncertainty. Dynamic entropy gating routes queries to internal parametric memory when confidence is > 0.92, invoking hybrid dense-graph retrieval only on high-uncertainty prompts. This hybrid gating cuts overall query latency by 58% while maintaining 91.2% factual correctness, preventing context poisoning observed in unconstrained RAG pipelines.",
        "sections": [
            ("1. Introduction", 
             "Neither pure RAG (Vance et al., Mehta et al.) nor pure parametric reasoning (Reynolds & Zhang) offers an optimal Pareto frontier across latency and factuality. A system must intelligently assess its own epistemic uncertainty before retrieving external context."),
            ("2. Architecture & Implementation", 
             "Our system monitors token entropy and attention dispersion across prompt representations. When epistemic uncertainty exceeds tau = 0.08, the router triggers a two-tier retrieval: fast vector retrieval for entity lookup and selective subgraph expansion for multi-hop queries."),
            ("3. Experimental Evaluation & Claims", 
             "Finding 1: Dynamic entropy gating routes queries to internal parametric memory when confidence is > 0.92, invoking hybrid dense-graph retrieval only on high-uncertainty prompts.\n"
             "Finding 2: Hybrid gating cuts overall query latency by 58% while maintaining 91.2% factual correctness across diverse benchmark suites.\n"
             "Finding 3: Selective retrieval prevents context poisoning observed in unconstrained RAG pipelines by suppressing retrieval when internal consensus is robust."),
            ("4. Limitations & Research Gaps", 
             "Uncertainty calibration requires fine-tuning on domain-specific validation splits. How to adapt uncertainty thresholds in out-of-distribution clinical scenarios remains an open research challenge. Cross-institution privacy constraints were not evaluated."),
            ("5. Conclusion", 
             "Adaptive routing reconciles the conflict between retrieval latency and parametric hallucination, offering a unified path for resilient clinical LLM deployments.")
        ]
    },
    {
        "filename": "Paper_5_Long_Context_vs_RAG.pdf",
        "title": "Long-Context LLMs vs. RAG: An Empirical Audit of 1M+ Token Reasoning",
        "authors": "Dr. Yuki Tanaka, Dr. Samuel O'Connor, Dr. Maya Lin",
        "affiliation": "Tokyo Institute of AI Technologies & DeepScale Lab",
        "year": "2025",
        "abstract": "With context windows expanding to millions of tokens, industry claims suggest that RAG architectures are obsolete. We present an empirical audit of frontier 1M+ token LLMs on multi-document scientific synthesis and clinical meta-analyses. We show that native long-context models exhibit severe 'lost-in-the-middle' factual degradation when evidence is buried beyond the 60th percentile of context depth (accuracy drops from 84% to 51%). Consequently, targeted retrieval-augmented generation remains strictly necessary for needle-in-haystack factual precision regardless of context window expansion. Attention dispersion across 100k+ tokens increases hallucination rate by 19% compared to focused 4k-token chunk retrieval.",
        "sections": [
            ("1. Introduction", 
             "The emergence of 1M+ token context windows has fueled speculation that external indexing and retrieval pipelines can be replaced by naive context stuffing. We conduct stress-testing on complex legal and medical corpora to audit this hypothesis."),
            ("2. Evaluation Protocol", 
             "We construct the 'Needle-in-Multi-Study' benchmark consisting of 50 full-length clinical trials (averaging 450,000 tokens total). We benchmark Gemini 1.5 Pro, Claude 3.5 Sonnet, and Llama-3-70B with context stuffing versus chunked hybrid RAG."),
            ("3. Claims & Findings", 
             "Finding 1: Native long-context models (1M+ tokens) exhibit 'lost-in-the-middle' factual degradation when evidence is buried beyond the 60th percentile of context depth (accuracy drops from 84% to 51%).\n"
             "Finding 2: Targeted retrieval-augmented generation remains strictly necessary for needle-in-haystack factual precision regardless of context window expansion.\n"
             "Finding 3: Attention dispersion across 100k+ tokens increases hallucination rate by 19% compared to focused 4k-token chunk retrieval."),
            ("4. Unresolved Research Gaps", 
             "The interaction between long-context attention heads and cached key-value states during iterative reasoning is poorly understood. Further research is required to construct hybrid chunk-and-stream architectures that maintain global reasoning with local precision."),
            ("5. Conclusion", 
             "Context stuffing does not obviate retrieval. Precision-targeted retrieval remains an irreplaceable architectural component for verifiable, hallucination-free scientific reasoning.")
        ]
    }
]

def generate_pdf(spec, output_dir):
    filepath = os.path.join(output_dir, spec["filename"])
    doc = SimpleDocTemplate(
        filepath,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
        title=spec["title"],
        author=spec["authors"]
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=10
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=12
    )
    
    abstract_heading = ParagraphStyle(
        'AbstractHeading',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=8,
        spaceAfter=4
    )
    
    abstract_body = ParagraphStyle(
        'AbstractBody',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=14
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0F766E'),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=8
    )

    story = []
    
    # Title
    story.append(Paragraph(spec["title"], title_style))
    
    # Authors and metadata
    meta_text = f"<b>Authors:</b> {spec['authors']}<br/><b>Institution:</b> {spec['affiliation']} &nbsp;|&nbsp; <b>Year:</b> {spec['year']}"
    story.append(Paragraph(meta_text, meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0D9488'), spaceBefore=2, spaceAfter=12))
    
    # Abstract
    story.append(Paragraph("ABSTRACT", abstract_heading))
    story.append(Paragraph(spec["abstract"], abstract_body))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceBefore=4, spaceAfter=10))
    
    # Sections
    for sec_title, sec_body in spec["sections"]:
        story.append(Paragraph(sec_title, section_heading))
        # Format bullet points or lines
        paragraphs = sec_body.split("\n")
        for p in paragraphs:
            if p.strip().startswith("Finding"):
                bullet_style = ParagraphStyle(
                    'BulletStyle',
                    parent=body_style,
                    leftIndent=15,
                    fontName='Helvetica',
                    textColor=colors.HexColor('#0F172A')
                )
                story.append(Paragraph(f"• <b>{p.strip()}</b>", bullet_style))
            else:
                story.append(Paragraph(p.strip(), body_style))
        story.append(Spacer(1, 4))
        
    doc.build(story)
    print(f"Generated: {filepath}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "data", "sample_papers")
    os.makedirs(target_dir, exist_ok=True)
    
    for spec in PAPERS_SPEC:
        generate_pdf(spec, target_dir)
        
    print(f"All 5 research paper PDFs generated successfully in {target_dir}")

if __name__ == "__main__":
    main()
