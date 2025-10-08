# Plan 2: Related Work Section Improvement for Defactify Paper

**Target File:** `overleaf/Defactify_Text_Shared_Task_Dataset_paper/related_work.tex`

**Date:** October 8, 2025

**Goal:** Restructure and complete the Related Work section following the 2-section format (AI-Generated Content + Fake News Detection) with dataset-centric focus and 20-30 paper citations.

---

## Current Issues with related_work.tex

1. **All substantive content is commented out** (lines 3-29) - Comprehensive related work exists but is inactive
2. **Incomplete structure** (lines 30-54) - Only TODOs and planning notes remain active
3. **Missing requested 2-section organization** - Doesn't follow instructions for AI-generated content + Fake news sections
4. **Wrong emphasis** - Current draft emphasizes methods over datasets (this is a dataset paper!)
5. **Inconsistent hierarchy** - Needs clearer subsection breakdown
6. **No positioning statement** - Doesn't explain what's novel about the 58K NYT + 6 LLMs dataset

---

## Recommended Structure (Based on Instructions)

```latex
\section{Related Work}

[Brief 3-4 sentence intro paragraph positioning this work within AI-generated text
detection and fake news detection research, emphasizing the need for large-scale,
diverse datasets]

\subsection{AI-Generated Text Detection}

\subsubsection{Detection Datasets}
[PRIMARY FOCUS - Detailed coverage of existing datasets with statistics]

Key datasets to cover:
- M4 Dataset (multi-domain, 7 languages, GPT-4 included)
- RAID Benchmark (6M generations, 11 models, 8 domains)
- TuringBench (10K news articles, 19 LLMs, politics-focused)
- MGTBench (benchmarking framework)
- LLM-DetectAIve (236K examples with humanized/polished labels)
- HART (21.5K examples, 4 categories: human, AI-refined, AI-generated, humanized)
- FAIDSet (84K multilingual texts, collaborative generations)
- DetectRL (real-world benchmark, NeurIPS 2024)
- ArguGPT (argumentative essays domain)
- HC3 (Human-ChatGPT comparison corpus)
- CHEAT (academic writing dataset)
- Ghostbuster dataset (creative writing focus)

[Target: 10-12 dataset citations with 1-2 sentences each describing scale,
domain, and unique features]

\subsubsection{Detection Methods}
[BRIEF OVERVIEW - Methods for context only]

Group by approach:
- **Zero-shot methods:** DetectGPT (probability curvature), Binoculars
  (cross-perplexity, 90%+ accuracy)
- **Rewriting-based:** RAIDAR (edit distance after rewriting - OUR BASELINE),
  Learning2Rewrite (fine-tuned rewrite model)
- **Watermarking:** SynthID (Google's logit modification), SynGuard
  (semantic-level guidance)
- **Supervised classifiers:** RoBERTa-based detectors, RADAR (adversarial training)

[Target: 8-10 method citations with 1 sentence each describing core technique]

\subsection{Fake News Detection}

\subsubsection{Fake News Datasets}
[PRIMARY FOCUS - Detailed coverage with emphasis on scale and domain]

Key datasets to cover:
- **LIAR** (12.8K short political claims, 6-way labels from PolitiFact)
- **LIAR-PLUS** (extended with journalist justifications)
- **FakeNewsNet** (PolitiFact + GossipCop subsets with social context)
- **FEVER** (185K fact-checked claims with Wikipedia evidence)
- **MultiFC** (multi-domain fact-checking corpus)
- **MMCFND** (multimodal multilingual for low-resource languages)
- **Fakeddit** (Reddit-based with image modality)
- **PHEME** (social media rumor dataset)
- **CoAID** (COVID-19 misinformation)
- **CONSTRAINT** (COVID-19 fake news in social media)

[Target: 8-10 dataset citations with comparison of scale, labels, and modalities]

\subsubsection{Detection Methods}
[BRIEF OVERVIEW]

Approach categories:
- **Fact-checking approaches:** Evidence retrieval and claim verification
- **Multimodal analysis:** Text-image consistency checking
- **Social context integration:** Network propagation patterns and user credibility
- **Knowledge graph-based:** External knowledge integration for reasoning

[Target: 5-8 method citations with brief descriptions]

\subsection{Positioning Our Dataset}

[1-2 paragraphs explaining what's NOVEL about our contribution:]

Key differentiators to emphasize:
1. **Source quality:** Real-world, high-quality journalistic content from NYT
   (not synthetic prompts or student essays)
2. **Temporal span:** 20+ years of articles (2000-present) reflecting evolving
   news landscape
3. **Model diversity:** 6 state-of-the-art LLMs (Gemma-2-9b, Mistral-7B,
   Qwen-2-72B, LLaMA-8B, Yi-Large, GPT-4-o)
4. **Scale and balance:** 58,502 total samples with ~7,300 per source (balanced)
5. **Dual-task support:** Enables both binary detection (Task A) and model
   attribution (Task B)
6. **Rich metadata:** Full article context with abstracts as controlled prompts

[Conclude with statement about bridging gap between real-world journalism and
modern LLM detection research]
```

---

## Key Datasets to Include (20-30 Papers Target)

### AI-Generated Text Detection Datasets (12-15 papers)

1. **RAID** - 6M generations, 11 models, 8 domains, 11 adversarial attacks
   - Reference: arxiv.org/abs/2405.07940
   - Note: Largest and most challenging benchmark, found detectors easily fooled

2. **M4** - Multi-domain (wiki-how, reddit, peerread, arxiv), 7 languages, GPT-4
   - Note: Multi-purpose dataset for detector training

3. **TuringBench** - 10K news articles, 19 LLMs, politics-focused
   - Note: Based on real news titles as prompts

4. **MGTBench** - Benchmarking framework for machine-generated text detection

5. **LLM-DetectAIve** - 236K examples including machine-humanized and human-polished labels
   - Domains: arXiv, Wikipedia, Reddit, student essays

6. **HART** - 21.5K examples, 4 categories
   - Categories: human-written, AI-refined, AI-generated, humanized AI-generated
   - Domains: student essays, arXiv abstracts, story writing, news articles

7. **FAIDSet** - 84K multilingual texts
   - Note: Includes diverse forms of human-LLM collaborative generations
   - Latest LLMs included

8. **DetectRL** - Real-world LLM-generated text detection benchmark
   - Accepted at NeurIPS 2024

9. **ArguGPT** - Argumentative essays dataset
   - Focus: Academic argumentation domain

10. **HC3** - Human-ChatGPT comparison corpus
    - Direct comparison pairs for detection research

11. **CHEAT** - Academic writing dataset
    - Focus: Student essays and academic integrity

12. **Ghostbuster** - Creative writing dataset
    - Focus: Literary and creative text generation

### Fake News Detection Datasets (8-12 papers)

13. **LIAR** - 12.8K political claims, 6-way labels (Wang, 2017)
    - Labels: pants-fire, false, barely-true, half-true, mostly-true, true
    - Source: PolitiFact.com over a decade
    - Average: 19 words per claim

14. **LIAR-PLUS** - Extended version with journalist justifications
    - Note: ~10K claims with explanations, politics domain

15. **FakeNewsNet** - Multi-dimensional repository with social context
    - PolitiFact subset: 434 real + 367 fake (balanced)
    - GossipCop subset available
    - Average: 1,459 words per article (much longer than LIAR)
    - Includes spatiotemporal information

16. **FEVER** - 185K fact-checked claims with Wikipedia evidence
    - Labels: Supported, Refuted, NotEnoughInfo
    - Claims generated by altering Wikipedia sentences
    - Fleiss kappa: 0.6841

17. **MultiFC** - Multi-domain fact-checking corpus
    - Covers diverse topics beyond politics

18. **MMCFND** - Multimodal Multilingual Caption-aware Fake News Detection
    - Focus: Low-resource languages
    - Includes text-image pairs

19. **Fakeddit** - Reddit-based dataset with images
    - Large-scale multimodal fake news dataset
    - Social media context

20. **PHEME** - Social media rumor dataset
    - Twitter-based rumors and verification

21. **CoAID** - COVID-19 misinformation dataset
    - Domain-specific: Health misinformation during pandemic

22. **CONSTRAINT** - COVID-19 fake news in social media
    - Social media posts about COVID-19
    - Real-time misinformation tracking

### Key Method Papers (Brief Mentions, 6-8 papers)

23. **RAIDAR** - Rewriting-based detection (Mao et al., 2024) - **YOUR BASELINE**
    - Key concept: LLMs make fewer edits to AI-generated text than human text
    - Uses "stubbornness" of LLMs
    - Surpasses previous methods by up to 29%
    - Reference: arxiv.org/abs/2401.12970, OpenReview ICLR 2024

24. **DetectGPT** - Zero-shot via probability curvature (Mitchell et al., 2023)
    - Exploits negative curvature in log probability space
    - No training required

25. **Fast-DetectGPT** - Optimized DetectGPT
    - 340x faster than original
    - No accuracy reduction

26. **Binoculars** - Cross-perplexity method, 90%+ accuracy (Hans et al., 2024)
    - Uses perplexity / cross-entropy between two similar LMs
    - Zero-shot and domain-agnostic
    - ICML 2024, strong at low false positive rates
    - Reference: arxiv.org/abs/2401.12070

27. **SynthID** - Google's watermarking (Uesato et al., 2024)
    - Embeds pseudorandom watermark via logit modification
    - Algorithmically verifiable with secret key
    - Vulnerable to paraphrasing

28. **SynGuard** - Enhanced watermarking with semantic guidance
    - Extends SynthID with semantic-level signals
    - 11%+ improvement in F1 against paraphrasing attacks
    - Survives meaning-preserving transformations

29. **RoBERTa-based detectors** - OpenAI detector
    - Supervised fine-tuning on labeled data
    - High accuracy in-distribution, brittle out-of-distribution

30. **RADAR** - Adversarial training framework
    - Trains detectors against paraphrasers
    - Improved robustness to meaning-preserving transformations

---

## References Management Strategy

### Current Status in `references.bib`

**✅ Already Present (Good foundation - DO NOT REMOVE):**
- **RAIDAR** (lines 28-36, 197, 262-283) - Multiple entries exist
- **DetectGPT** (lines 8-16, 197) - Already included
- **Binoculars** (line 197: `hans2024spotting`)
- **SynthID** (lines 285-304) - Google's watermarking
- **All 6 LLMs** - Gemma (39-47), Mistral (49-57), Qwen (59-67), LLaMA (69-77), Yi (79-87), GPT-4 (89-97)
- **MMCFND** (lines 353-374) - Multimodal fake news dataset
- **RADAR** (line 197: `radar_openreview`) - Partially present
- **RoBERTa** (line 197: `roberta_openai`)

**❌ Missing (~20 papers to add):**

**AI-Generated Text Detection Datasets (10 papers):**
1. RAID benchmark (arxiv.org/abs/2405.07940)
2. M4 dataset
3. TuringBench
4. MGTBench
5. LLM-DetectAIve
6. HART
7. FAIDSet
8. DetectRL
9. HC3
10. CHEAT/Ghostbuster/ArguGPT

**Fake News Detection Datasets (9 papers):**
11. LIAR (Wang, 2017)
12. LIAR-PLUS
13. FakeNewsNet
14. FEVER
15. MultiFC
16. Fakeddit
17. PHEME
18. CoAID
19. CONSTRAINT

**Detection Methods (2 papers):**
20. Fast-DetectGPT
21. SynGuard

### Recommended Approach: Incremental Addition

**✅ SELECTED STRATEGY: Option 1 - Add Missing Entries Incrementally**

**PROS:**
- Clean, organized approach
- Verify each entry before adding
- Avoid creating new duplicates
- Prioritize most-cited papers first
- Maintain existing structure

**CONS:**
- Takes 30-45 minutes
- Requires finding proper BibTeX entries

**IMPORTANT RULES:**
1. ⚠️ **DO NOT delete or clean existing entries** - Keep all current references intact
2. ✅ **Only add missing references** - Check if citation key already exists before adding
3. ✅ **Use consistent citation keys** - Follow pattern: `author2024keyword`
4. ✅ **Include essential fields only** - title, author, year, venue/journal, url/doi
5. ✅ **Add entries at end of file** - Append new references after line 491

### BibTeX Entry Addition Process

For each missing paper:

1. **Search Google Scholar** - Find the paper
2. **Click "Cite" → "BibTeX"** - Get formatted entry
3. **Verify citation key** - Check it doesn't exist in references.bib
4. **Clean the entry** - Remove unnecessary fields (month, pages, abstract)
5. **Add to end of file** - Append after existing entries
6. **Test compilation** - Ensure LaTeX compiles without errors

### Priority Order for Adding References

**High Priority (Add first - 10 papers):**
1. RAID - Main benchmark comparison
2. LIAR - Classic fake news dataset
3. FEVER - Large-scale fact-checking
4. FakeNewsNet - Social context
5. M4 - Multi-domain AI-gen dataset
6. TuringBench - News domain AI-gen
7. Fast-DetectGPT - Improved baseline method
8. SynGuard - Watermarking advancement
9. HC3 - Human-ChatGPT comparison
10. HART - Multi-category AI-gen

**Medium Priority (Add if cited - 8 papers):**
11. LLM-DetectAIve - Humanized labels
12. FAIDSet - Multilingual
13. DetectRL - NeurIPS benchmark
14. MultiFC - Multi-domain fact-checking
15. LIAR-PLUS - Extended LIAR
16. Fakeddit - Multimodal Reddit
17. CoAID - COVID misinformation
18. CONSTRAINT - COVID fake news

**Low Priority (Add if space permits - 3 papers):**
19. MGTBench - Benchmarking framework
20. PHEME - Twitter rumors
21. ArguGPT/CHEAT/Ghostbuster - Domain-specific

---

## Writing Guidelines

### Section Length Targets

- **Introduction paragraph:** 3-4 sentences
- **AI-Generated Text Detection Datasets:** 1.5-2 pages
- **AI-Generated Text Detection Methods:** 0.5-0.75 pages
- **Fake News Detection Datasets:** 1-1.5 pages
- **Fake News Detection Methods:** 0.5 pages
- **Positioning Our Dataset:** 0.5 pages (8-10 sentences)
- **Total target:** 4-5 pages for Related Work section

### Citation Style

- Use numerical citations: `\cite{mao2024raidar}`
- For multiple works: `\cite{mitchell2023detectgpt,hans2024binoculars}`
- Ensure all citations are in `references.bib`

### Paragraph Structure for Datasets

Template for each dataset mention:
```
[Dataset Name] \cite{citation} is a [scale] dataset containing [size]
[domain/type] texts. It features [key characteristic 1], [key characteristic 2],
and enables research in [application area].
```

### Paragraph Structure for Methods

Template for each method group:
```
[Method Category] approaches include [Method1] \cite{citation1}, which
[brief mechanism description], and [Method2] \cite{citation2}, which
[brief mechanism + key result].
```

---

## PROS and CONS Analysis

### PROS of Recommended Structure

1. ✅ **Dataset-centric approach** - Aligns with paper being a dataset contribution
2. ✅ **Clear 2-section hierarchy** - Follows explicit instructions (AI-gen + Fake news)
3. ✅ **Logical subsection flow** - Datasets first (primary), then methods (context)
4. ✅ **Comprehensive coverage** - 20-30 papers as requested
5. ✅ **Balanced treatment** - Both domains (AI-gen and fake news) well-represented
6. ✅ **Strong positioning** - Dedicated subsection explains dataset novelty
7. ✅ **Reader-friendly** - Similar structure in both main sections (symmetry)

### CONS to Consider

1. ⚠️ **Risk of length** - May exceed typical related work section (4-5 pages)
   - **Mitigation:** Keep method descriptions to 1 sentence each

2. ⚠️ **Domain overlap** - AI-generated text CAN BE fake news (not always distinct)
   - **Mitigation:** Acknowledge overlap in intro, focus AI-gen on detection task,
     fake news on veracity/claims

3. ⚠️ **Balance challenge** - Fake news section might overshadow AI-gen focus
   - **Mitigation:** Make AI-gen section slightly longer (2.5 pages vs 2 pages)

4. ⚠️ **Citation availability** - Some datasets may not have proper papers yet
   - **Mitigation:** Use arXiv preprints, Hugging Face dataset cards, or GitHub repos

5. ⚠️ **Redundancy risk** - Some papers appear in multiple categories
   - **Mitigation:** Cite once in most relevant location, cross-reference if needed

---

## Alternative Approaches Considered

### Alternative 1: Single Unified Section

```
\section{Related Work}
\subsection{Datasets for Content Authenticity}
\subsection{Detection Methods}
\subsection{Our Contribution}
```

**PROS:**
- Cleaner organization, less repetitive
- Natural grouping by task type
- Shorter overall length

**CONS:**
- Doesn't follow explicit 2-section instruction
- Loses domain-specific context (AI vs fake news)
- Harder to position work within two research communities

**Verdict:** ❌ Not recommended - Instructions explicitly request 2 sections

### Alternative 2: Chronological Organization

```
\section{Related Work}
\subsection{Early Fake News Detection (2015-2019)}
\subsection{Rise of AI-Generated Text (2020-2022)}
\subsection{Modern Challenges (2023-2025)}
\subsection{Our Contribution}
```

**PROS:**
- Shows evolution of field
- Natural narrative flow
- Highlights recent urgency

**CONS:**
- Doesn't follow instruction format
- Harder for readers to find specific dataset/method info
- Mixes domains chronologically (confusing)

**Verdict:** ❌ Not recommended - Too different from requested structure

### Alternative 3: Task-Based Organization

```
\section{Related Work}
\subsection{Binary Classification Datasets and Methods}
\subsection{Model Attribution Datasets and Methods}
\subsection{Our Dual-Task Dataset}
```

**PROS:**
- Directly aligns with paper's Task A and Task B
- Highly relevant to contribution
- Unique organizational angle

**CONS:**
- Doesn't match instruction structure
- Most existing work focuses on binary only
- Limited prior work on attribution (small section)

**Verdict:** ❌ Not recommended - Insufficient prior work for balanced sections

---

## Implementation Action Items

### Phase 1: Content Extraction (15 minutes)

1. ✅ Uncomment lines 3-29 from current `related_work.tex`
2. ✅ Extract useful content about:
   - RAIDAR (lines 13-15, 19)
   - SynthID (lines 3, 25)
   - DetectGPT/Binoculars (line 21)
   - Multimodal datasets (line 7)
3. ✅ Save extracted paragraphs to temporary file for reuse
4. ✅ Delete placeholder text (lines 30-54)

### Phase 2: Structure Creation (10 minutes)

5. ✅ Create section header and intro paragraph
6. ✅ Create subsection skeleton:
   - `\subsection{AI-Generated Text Detection}`
   - `\subsubsection{Detection Datasets}`
   - `\subsubsection{Detection Methods}`
   - `\subsection{Fake News Detection}`
   - `\subsubsection{Fake News Datasets}`
   - `\subsubsection{Detection Methods}`
   - `\subsection{Positioning Our Dataset}`

### Phase 3: Dataset Descriptions (45-60 minutes)

7. ✅ Write AI-generated text detection datasets (12 datasets × 2-3 sentences)
8. ✅ Write fake news detection datasets (10 datasets × 2-3 sentences)
9. ✅ Include key statistics: size, domain, labels, languages
10. ✅ Ensure citations for each dataset

### Phase 4: Method Descriptions (30 minutes)

11. ✅ Write brief AI-gen detection methods (8-10 methods, 1 sentence each)
12. ✅ Write brief fake news detection methods (5-8 methods, 1 sentence each)
13. ✅ Emphasize RAIDAR as baseline for your work
14. ✅ Group methods by approach (zero-shot, supervised, watermarking, etc.)

### Phase 5: Positioning Section (15 minutes)

15. ✅ Write 2 paragraphs explaining dataset novelty:
    - Paragraph 1: What exists (gap analysis)
    - Paragraph 2: Our contribution (how we fill the gap)
16. ✅ Emphasize 6 unique aspects listed above
17. ✅ End with forward-looking statement about enabling future research

### Phase 6: References (30-45 minutes)

18. ✅ Check existing references.bib for already present citations
19. ✅ Add ONLY missing citations to `references.bib` (append at end)
20. ✅ Use consistent citation format (see References Management Strategy above)
21. ✅ Include essential fields: title, author, year, venue, url/doi
22. ✅ Test each addition - compile to verify no LaTeX errors
23. ⚠️ **DO NOT remove or modify existing entries**

### Phase 7: Review and Polish (20 minutes)

24. ✅ Check flow between paragraphs
25. ✅ Ensure balanced coverage (AI-gen slightly more than fake news)
26. ✅ Verify all dataset statistics are accurate
27. ✅ Check for typos and LaTeX syntax errors
28. ✅ Compile document to verify all citations render correctly

**Total estimated time:** 2.75-3.25 hours

---

## Key Datasets Quick Reference Table

| Dataset | Type | Size | Domain | Year | Key Feature |
|---------|------|------|--------|------|-------------|
| RAID | AI-Gen | 6M | Multi | 2024 | 11 models, adversarial |
| M4 | AI-Gen | Large | Multi | 2024 | 7 languages, GPT-4 |
| TuringBench | AI-Gen | 10K | News | 2022 | 19 LLMs, politics |
| LLM-DetectAIve | AI-Gen | 236K | Multi | 2024 | Humanized labels |
| HART | AI-Gen | 21.5K | Multi | 2024 | 4-way labels |
| FAIDSet | AI-Gen | 84K | Multi | 2024 | Multilingual |
| LIAR | Fake | 12.8K | Politics | 2017 | 6-way labels |
| FakeNewsNet | Fake | 23K | News | 2019 | Social context |
| FEVER | Fake | 185K | Claims | 2018 | Wikipedia evidence |
| MMCFND | Fake | Multi | Multi | 2023 | Multimodal, low-resource |
| **Our Dataset** | **AI-Gen** | **58.5K** | **News** | **2025** | **NYT + 6 LLMs** |

---

## Key Method Quick Reference

| Method | Type | Approach | Accuracy | Year | Note |
|--------|------|----------|----------|------|------|
| RAIDAR | Post-hoc | Rewriting + edit distance | +29% | 2024 | **Our baseline** |
| Binoculars | Zero-shot | Cross-perplexity | 90%+ | 2024 | Best at low FPR |
| DetectGPT | Zero-shot | Probability curvature | High | 2023 | No training needed |
| Fast-DetectGPT | Zero-shot | Optimized DetectGPT | High | 2024 | 340× faster |
| SynthID | Proactive | Watermarking (logits) | N/A | 2024 | Vulnerable to paraphrasing |
| SynGuard | Proactive | Semantic watermarking | +11% F1 | 2024 | Survives paraphrasing |
| RoBERTa | Supervised | Fine-tuned classifier | High | 2019 | Poor OOD performance |
| RADAR | Supervised | Adversarial training | High | 2024 | Robust to paraphrasing |

---

## Important Notes for Writer

### What to EMPHASIZE

- **Dataset scale and diversity** - This is a dataset paper
- **Your unique contribution** - NYT source, temporal span, 6 LLMs
- **Gaps in existing work** - Most datasets use synthetic prompts, not real journalism
- **Both tasks** - Detection AND attribution (most work focuses only on detection)

### What to DE-EMPHASIZE

- **Method implementation details** - Save for Baseline section
- **Performance comparisons** - Not the focus of related work
- **Limitations of prior work** - Brief mention only, stay positive
- **Your results** - Save for Results section

### Common Pitfalls to AVOID

1. ❌ Making related work section too long (>5 pages)
2. ❌ Describing methods in detail (keep brief)
3. ❌ Criticizing prior work harshly (be respectful)
4. ❌ Forgetting to position your dataset at the end
5. ❌ Unbalanced coverage (one section much longer than other)
6. ❌ Missing citations (every claim needs a reference)
7. ❌ Redundant citations (cite each paper once in best location)

---

## Success Criteria

The Related Work section will be considered complete when:

✅ 20-30 papers cited across both domains
✅ Clear 2-section structure (AI-gen + Fake news) implemented
✅ Dataset descriptions more detailed than method descriptions
✅ All datasets include size, domain, and key features
✅ RAIDAR highlighted as baseline method
✅ Positioning subsection clearly explains novelty
✅ Balanced length (~2.5 pages AI-gen, ~2 pages fake news)
✅ All citations present in references.bib
✅ Document compiles without LaTeX errors
✅ Smooth transitions between paragraphs

---

## Next Steps

1. Review this plan with co-authors
2. Assign writing tasks if multiple people contributing
3. Gather full citation information for all 30 papers
4. Begin Phase 1 implementation (content extraction)
5. Draft section incrementally following phases
6. Schedule review meeting after Phase 7 complete

---

**Plan prepared by:** Claude Code
**Date:** October 8, 2025
**Target completion:** 2.75-3.25 hours of focused writing
**Review cycle:** 1-2 iterations expected

**References strategy:** Incremental addition only - DO NOT remove existing entries
