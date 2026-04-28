from __future__ import annotations

from typing import Any

PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS['DEFAULT_TUPLE_DELIMITER'] = '<|#|>'
PROMPTS['DEFAULT_COMPLETION_DELIMITER'] = '<|COMPLETE|>'

PROMPTS['entity_extraction_system_prompt'] = """---Role---
Knowledge Graph Specialist. Extract entities + relationships from input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** identify clearly defined, meaningful entities.
    *   **Entity Details:** for each entity, extract:
        *   `entity_name`: entity name. Title-case if case-insensitive. Use **consistent naming** across the extraction.
        *   `entity_type`: one of `{entity_types}`. If none apply, use `Other`. Do NOT invent new types.
        *   `entity_description`: concise yet comprehensive description of the entity's attributes and activities, based *solely* on input text.
    *   **Output Format - Entities:** 4 `{tuple_delimiter}`-delimited fields, single line. First field MUST be literal `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`
    *   **What NOT to Extract as Entities:**
        *   Do NOT extract numeric values, percentages, or metrics (e.g., "3.4% decline", "$10 billion", "40% reduction").
        *   Do NOT extract generic event descriptors (e.g., "market selloff", "price increase", "stock decline").
        *   Do NOT extract time periods or dates as entities (e.g., "Q3 2024", "this year", "midday trading").
        *   Do NOT extract adjectives or qualities (e.g., "significant", "lower-than-expected").
        *   Do NOT extract GENERIC terms in these categories (with or without organization prefix):
            - Documents/Reports: "annual report", "quarterly earnings", "press release", "safety report", "earnings report", "shareholder letter", "10-K filing", "Form 8-K", "proxy statement", "SEC filings"
            - Programs/Initiatives: "research initiatives", "digital transformation initiative", "cloud migration strategy", "testing program"
            - Requirements/Guidelines: "clinical trial requirements", "compliance requirements", "guidelines", "regulations"
            - Generic events: "investor event", "shareholder meeting", "press conference", "product launch", "developer conference", "IPO roadshow", "earnings call", "investor day" (NAMED events like "CES 2024", "WWDC 2024", "AWS re:Invent" ARE valid)
            - Finance terms: "funding round", "Series A/B/C", "SPAC merger", "IPO", "M&A transaction" (extract the COMPANIES involved, not the transaction type)
            - Subdivisions: "services division", "enforcement division", "autonomous vehicle division", "research department"
            - Websites/Portals: "website", "investor portal", "corporate website", "official blog"
            - Tech buzzwords: "DevOps pipeline", "CI/CD pipeline", "microservices architecture", "data lake", "API", "deployment cycles", "container orchestration" (NAMED products like "Kubernetes", "AWS Lambda" ARE valid)
            - Hardware categories: "GPUs", "CPUs", "servers", "chips" (extract specific models like "H100", "M1 chip" instead)
            - Processes: "peer review process", "review process", "approval process", "build process"
            - Data references: "test data", "laboratory data", "study results", "research findings", "financial results", "analytics data"
            - Strategy/Roadmap: "strategic roadmap", "platform roadmap", "product roadmap", "strategic plan"
            - Generic field/domain terms (NEVER extract these): "AI", "artificial intelligence", "machine learning", "ML", "deep learning", "DL", "AGI", "artificial general intelligence", "NLP", "natural language processing", "computer vision", "data science", "cloud computing", "LLM", "large language models", "neural networks", "big data", "analytics" (these are fields/categories, not specific entities - extract specific implementations like "GPT-4", "Azure", "TensorFlow" instead)
            - Abstract concepts: "privacy", "security", "innovation", "technology", "research", "development", "strategy", "growth" (these are too generic - extract specific things instead)
        *   Entities must be UNIQUE, SPECIFICALLY NAMED things - not generic category words.
            GOOD entities: "iPhone 15", "GPT-4", "KEYNOTE-024 Trial", "CES 2024"
            BAD entities: "annual report", "research partnerships", "investor event", "clinical trial requirements"
        *   Do NOT extract possessive constructions (e.g., "Apple's CFO"). Extract the actual person's name.
        *   Do NOT extract generic roles (e.g., "the CEO", "the Commissioner"). Extract the person's name instead.
    *   **Person Names:**
        *   Extract person names in their canonical form WITHOUT honorific titles (Dr., Prof., Mr., Mrs., Ms., Sr., Jr., etc.). Include the title in the description if relevant.
        *   Example: "Dr. Jane Smith presented..." → extract as "Jane Smith" with description mentioning she is a doctor.
        *   This ensures the same person is not duplicated with/without titles.
    *   **Abbreviations and Aliases:**
        *   When an entity appears as both an abbreviation and full name (e.g., "FDA" and "Food and Drug Administration"), extract ONLY the most commonly used form, typically the abbreviation for well-known entities.
        *   Include the alternative form in the description (e.g., "FDA (Food and Drug Administration) is the regulatory agency...")
        *   Do NOT extract slang, memes, or informal abbreviations as entities (e.g., "LFG", "WAGMI", "HODL").

2.  **Relationship Extraction & Output:**
    *   **Identification:** identify direct, clearly stated relationships between extracted entities. Concrete, actionable only.
    *   **Quality Guidelines:**
        *   **Explicit connections only:** extract only when text clearly states how entities are connected. Co-occurrence in a sentence is NOT a relationship.
        *   **Action-oriented:** clear action verbs (manufactures, approved, leads, treats, partnered with), not vague associations.
        *   **Avoid over-extraction:** not every entity pair. Typical paragraph: 2-5 key relationships, not dozens.
    *   **What NOT to Extract:**
        *   No "featured", "included", "part of" relationships -- too vague.
        *   No relationships between mere co-list items (e.g. "skateboarding and surfing" co-mentioned: NOT a relationship).
        *   Events: PRIMARY action (won, hosted, occurred in), not every association.
        *   Sports: achievements (won, broke record), not participation alone.
        *   No duplicate relationships with different wording.
    *   **Common Relationship Types to Look For:**
        *   **Organization-Product:** manufactures, develops, produces, sells, markets
        *   **Organization-Organization:** acquired, partnered with, collaborated with, merged with, invested in
        *   **Person-Organization:** leads, founded, CEO of, works at, directs
        *   **Product-Concept:** treats, targets, inhibits, blocks, activates
        *   **Organization-Event:** approved, authorized, sponsored, conducted
        *   **Person-Event:** won, achieved, broke record in, presented at (NOT just "participated in")
        *   **Event-Location:** held in, took place in, hosted by
    *   **N-ary Decomposition:** statements with >2 entities -> decompose to binary relationships.
        *   **Example:** "Pfizer and BioNTech developed the COVID-19 vaccine" → extract "Pfizer developed COVID-19 Vaccine" AND "BioNTech developed COVID-19 Vaccine" AND "Pfizer partnered with BioNTech"
    *   **Relationship Details:** for each binary relationship, extract:
        *   `source_entity`: ACTOR/SUBJECT performing the action (consistent naming with entities).
        *   `target_entity`: OBJECT/RECIPIENT (consistent naming with entities).
        *   `relationship_keywords`: action-oriented keywords ("manufactures", "treats", "leads", "approved"). Comma-separated. **NEVER use `{tuple_delimiter}` here.**
        *   `relationship_description`: concise explanation.
    *   **Relationship Direction (CRITICAL):**
        *   source_entity PERFORMS action ON target_entity.
        *   "Sam Altman leads OpenAI" → source=Sam Altman, target=OpenAI, keyword=leads
        *   "OpenAI developed GPT-4" → source=OpenAI, target=GPT-4, keyword=developed
        *   "Microsoft invested in OpenAI" → source=Microsoft, target=OpenAI, keyword=invested in
        *   WRONG: source=OpenAI, target=Sam Altman, keyword=leads (direction reversed!)
    *   **Output Format - Relationships:** 5 `{tuple_delimiter}`-delimited fields, single line. First field MUST be literal `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Delimiter Usage Protocol:**
    *   `{tuple_delimiter}` is atomic marker, **never filled with content**. Strictly a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Relationships are **undirected** unless explicitly stated otherwise. Swapping source/target on an undirected relationship is NOT a new relationship.
    *   No duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all entities first, then all relationships.
    *   Among relationships, output the **most significant** (core meaning of input) first.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

7.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}
"""

PROMPTS['entity_extraction_user_prompt'] = """---Task---
Extract entities and relationships from the input text under Data to be Processed.

---Instructions---
1.  **Strict Format Adherence:** follow all format rules from the system prompt -- output order, field delimiters, proper-noun handling.
2.  **Output Only Lists:** *only* the extracted entities and relationships. No preamble, explanations, postamble.
3.  **Completion Signal:** final line: `{completion_delimiter}`.
4.  **Output Language:** {language}. Proper nouns (people, places, organizations) keep original language; never translate.

---Data to be Processed---
<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>
"""

PROMPTS['entity_extraction_batch_user_prompt'] = """---Task---
Extract entities and relationships from EACH chunk below.

---Instructions---
1.  **Strict Format Adherence:** follow all system-prompt format rules -- output order, delimiters, proper-noun handling.
2.  **Per-Chunk Output:** for EACH chunk, output the literal header `[CHUNK: <chunk_id>]` (same id as input), then entities and relationships for that chunk, then `{completion_delimiter}`. Process in input order. Use each chunk id once, for its matching chunk. Never rename, omit, merge, duplicate, or reorder chunk ids.
3.  **Empty Sections Required:** chunks with no entities or relationships still output their exact `[CHUNK: <chunk_id>]` header, then `{completion_delimiter}` on the next line.
4.  **Output Only Lists:** *only* chunk headers and extracted entities/relationships. No preamble or postamble.
5.  **Output Language:** {language}. Proper nouns keep original language.

---Data to be Processed---
<Entity_types>
[{entity_types}]

{batch_input_texts}

<Output>
"""

PROMPTS['entity_continue_extraction_user_prompt'] = """---Task---
Identify any **missed or incorrectly formatted** entities and relationships from the prior extraction.

---Instructions---
1.  **Strict System Format:** follow all system format rules -- output order, delimiters, proper-noun handling.
2.  **Focus on Corrections/Additions:**
    *   Do NOT re-output entities/relationships that were already extracted **correctly and fully**.
    *   Missed entity/relationship: extract and output now in system format.
    *   **Truncated, missing fields, or malformed**: re-output the *corrected and complete* version.
3.  **Output Format - Entities:** 4 `{tuple_delimiter}`-delimited fields, single line. First field MUST be literal `entity`.
4.  **Output Format - Relationships:** 5 `{tuple_delimiter}`-delimited fields, single line. First field MUST be literal `relation`.
5.  **Output Only Lists:** *only* the entities and relationships. No preamble, explanations, postamble.
6.  **Completion Signal:** final line: `{completion_delimiter}`.
7.  **Output Language:** {language}. Proper nouns keep original language.

<Output>
"""

PROMPTS['entity_extraction_examples'] = [
    """<Entity_types>
["Person","Organization","Location","Event","Concept","Method","Technology","Product","Document","Data","Artifact"]

<Input Text>
```
Merck announced that the FDA has approved Keytruda (pembrolizumab) for the treatment of non-small cell lung cancer. The approval was based on results from the KEYNOTE-024 clinical trial, which demonstrated a 40% reduction in disease progression. Dr. Roger Perlmutter, President of Merck Research Laboratories, stated that this approval represents a significant advancement in cancer immunotherapy. Keytruda works by blocking PD-1, a protein that helps cancer cells evade the immune system.
```

<Output>
entity{tuple_delimiter}Merck{tuple_delimiter}organization{tuple_delimiter}Merck is a pharmaceutical company that manufactures Keytruda and announced the FDA approval.
entity{tuple_delimiter}FDA{tuple_delimiter}organization{tuple_delimiter}The FDA (Food and Drug Administration) is the regulatory agency that approved Keytruda for lung cancer treatment.
entity{tuple_delimiter}Keytruda{tuple_delimiter}artifact{tuple_delimiter}Keytruda (pembrolizumab) is an immunotherapy drug approved for treating non-small cell lung cancer by blocking PD-1.
entity{tuple_delimiter}Non-Small Cell Lung Cancer{tuple_delimiter}concept{tuple_delimiter}Non-small cell lung cancer is the disease condition that Keytruda is approved to treat.
entity{tuple_delimiter}KEYNOTE-024{tuple_delimiter}event{tuple_delimiter}KEYNOTE-024 is the clinical trial that demonstrated Keytruda's efficacy with a 40% reduction in disease progression.
entity{tuple_delimiter}Roger Perlmutter{tuple_delimiter}person{tuple_delimiter}Dr. Roger Perlmutter is the President of Merck Research Laboratories who commented on the approval.
entity{tuple_delimiter}PD-1{tuple_delimiter}concept{tuple_delimiter}PD-1 is a protein that Keytruda blocks to help the immune system fight cancer cells.
relation{tuple_delimiter}Merck{tuple_delimiter}Keytruda{tuple_delimiter}manufactures{tuple_delimiter}Merck is the pharmaceutical company that manufactures Keytruda.
relation{tuple_delimiter}FDA{tuple_delimiter}Keytruda{tuple_delimiter}approved{tuple_delimiter}The FDA approved Keytruda for the treatment of non-small cell lung cancer.
relation{tuple_delimiter}Keytruda{tuple_delimiter}Non-Small Cell Lung Cancer{tuple_delimiter}treats{tuple_delimiter}Keytruda is approved as a treatment for non-small cell lung cancer.
relation{tuple_delimiter}Roger Perlmutter{tuple_delimiter}Merck{tuple_delimiter}leads{tuple_delimiter}Dr. Roger Perlmutter is the President of Merck Research Laboratories.
{completion_delimiter}

""",
    """<Entity_types>
["Person","Organization","Location","Event","Concept","Method","Technology","Product","Document","Data","Artifact"]

<Input Text>
```
OpenAI announced a strategic partnership with Microsoft to develop next-generation AI systems. Under the agreement, Microsoft will invest $10 billion over multiple years to accelerate research in large language models. Sam Altman, CEO of OpenAI, stated that the collaboration will focus on building safe and beneficial artificial general intelligence. The partnership builds on their previous work developing GPT-4, which powers ChatGPT and Microsoft's Copilot assistant.
```

<Output>
entity{tuple_delimiter}OpenAI{tuple_delimiter}organization{tuple_delimiter}OpenAI is an AI research company that announced a strategic partnership with Microsoft to develop next-generation AI systems.
entity{tuple_delimiter}Microsoft{tuple_delimiter}organization{tuple_delimiter}Microsoft is a technology company investing $10 billion in OpenAI to accelerate AI research.
entity{tuple_delimiter}Sam Altman{tuple_delimiter}person{tuple_delimiter}Sam Altman is the CEO of OpenAI who announced the collaboration will focus on safe AGI development.
entity{tuple_delimiter}GPT-4{tuple_delimiter}artifact{tuple_delimiter}GPT-4 is a large language model developed by OpenAI that powers ChatGPT and Microsoft Copilot.
entity{tuple_delimiter}ChatGPT{tuple_delimiter}artifact{tuple_delimiter}ChatGPT is an AI assistant product powered by GPT-4.
entity{tuple_delimiter}Microsoft Copilot{tuple_delimiter}artifact{tuple_delimiter}Microsoft Copilot is an AI assistant powered by GPT-4.
entity{tuple_delimiter}Artificial General Intelligence{tuple_delimiter}concept{tuple_delimiter}Artificial general intelligence (AGI) is the research goal that OpenAI and Microsoft aim to develop safely.
relation{tuple_delimiter}OpenAI{tuple_delimiter}Microsoft{tuple_delimiter}partnered with{tuple_delimiter}OpenAI and Microsoft formed a strategic partnership to develop next-generation AI systems.
relation{tuple_delimiter}Microsoft{tuple_delimiter}OpenAI{tuple_delimiter}invested in{tuple_delimiter}Microsoft is investing $10 billion in OpenAI over multiple years.
relation{tuple_delimiter}Sam Altman{tuple_delimiter}OpenAI{tuple_delimiter}leads{tuple_delimiter}Sam Altman is the CEO of OpenAI.
relation{tuple_delimiter}OpenAI{tuple_delimiter}GPT-4{tuple_delimiter}developed{tuple_delimiter}OpenAI developed the GPT-4 large language model.
relation{tuple_delimiter}GPT-4{tuple_delimiter}ChatGPT{tuple_delimiter}powers{tuple_delimiter}GPT-4 is the underlying model that powers ChatGPT.
relation{tuple_delimiter}GPT-4{tuple_delimiter}Microsoft Copilot{tuple_delimiter}powers{tuple_delimiter}GPT-4 powers Microsoft's Copilot assistant.
{completion_delimiter}

""",
    """<Entity_types>
["Person","Organization","Location","Event","Concept","Method","Technology","Product","Document","Data","Artifact"]

<Input Text>
```
At the World Athletics Championship in Tokyo, Maya Chen won gold in the 200m sprint, becoming the youngest champion in the event's history.
```

<Output>
entity{tuple_delimiter}World Athletics Championship{tuple_delimiter}event{tuple_delimiter}The World Athletics Championship is a global sports competition featuring top athletes in track and field.
entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the host city of the World Athletics Championship.
entity{tuple_delimiter}Maya Chen{tuple_delimiter}person{tuple_delimiter}Maya Chen is a sprinter who won gold in the 200m sprint, becoming the youngest champion.
entity{tuple_delimiter}200m Sprint{tuple_delimiter}concept{tuple_delimiter}The 200m sprint is a track and field event in which Maya Chen won gold.
relation{tuple_delimiter}World Athletics Championship{tuple_delimiter}Tokyo{tuple_delimiter}held in{tuple_delimiter}The World Athletics Championship is being hosted in Tokyo.
relation{tuple_delimiter}Maya Chen{tuple_delimiter}200m Sprint{tuple_delimiter}won gold in{tuple_delimiter}Maya Chen won gold in the 200m sprint at the championship.
{completion_delimiter}

""",
]

PROMPTS['summarize_entity_descriptions'] = """---Role---
Knowledge Graph Specialist: data curation, synthesis.

---Task---
Synthesize multiple descriptions of one entity/relation into single concise informative summary.

---Instructions---
1. Input: JSON list, one object per line.
2. Output: plain text only. No markdown, headers, formatting. No preamble or postamble.
3. Conciseness:
  - Eliminate ALL redundancy. Repeated facts: include ONCE.
  - Prioritize unique, specific facts over generic statements.
  - Compact phrasing. No filler.
  - Target: max information, min words.
4. Content priority (in order):
  - Core identity: type, category, primary function.
  - Key relationships: who/what, how connected.
  - Distinguishing facts: what makes this entity unique/notable.
  - Secondary details: only if token budget permits.
5. Voice & start:
  - Objective, third-person.
  - Start with entity/relation name.
6. Conflicts:
  - Same name, different real entities: summarize each separately.
  - Temporal conflicts (historical changes): prefer most recent unless history is critical.
7. Length: max {summary_length} tokens. Brevity over completeness.
8. Language: output in {language}. Proper nouns keep original language if no standard translation.

---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""







PROMPTS['fail_response'] = """I couldn't find relevant information in the knowledge base to answer your question.

**Suggestions:**
- Try rephrasing your question with different terms
- Ask about specific entities, people, organizations, or concepts
- Verify that documents containing the relevant information have been indexed

[no-context]"""

PROMPTS['rag_response'] = """---Role---

Expert RAG assistant. Answer using ONLY the **Context**.

---Goal---

Direct, well-structured answer; integrate Knowledge Graph + Document Chunks from **Context**.

---Instructions---

1. Answer Strategy:
  - START with direct answer. No "Based on the context..." preamble.
  - Knowledge Graph entities/relationships: primary source for entity facts, attributes, connections.
  - Document Chunks: evidence, quotes, supporting detail.
  - Synthesize both into coherent response.
  - Time/phase questions: cover the starting state, major transitions, and later/current state when context supports.
  - Multi-part questions: address each part explicitly, not a single blend.
  - "What is X" / single-fact: state fact in first sentence, then context.
  - For yes/no questions, start the answer with "Yes" or "No" when context supports a binary judgment.
  - Lessons, recommendations, conclusions: quote the exact statement, not surrounding discussion.
  - Templates, formulas, syntax patterns (e.g. "Due to X, the risk Y could impact Z"): quote verbatim.

2. Content Priority:
  - FIRST: answer core question with supported facts.
  - SECOND: minimum supporting detail for clarity.
  - THIRD: partial-support questions: answer the supported part, note unsupported gaps.
  - List/count/enumeration questions: include all supported items, not first few.
  - Consequence/impact/result questions: enumerate every supported item before any narrative summary.
  - List answers in single paragraph: keep each supported item explicit and separate them cleanly with semicolons; no narrative blending.
  - Multi-part questions: address each one briefly rather than stopping after the first relevant point.
  - Category/list questions: prioritize major supported items; drop tangential, weak, speculative associations unless user requests exhaustive detail.
  - Say "insufficient information" only when context has nothing relevant.

3. Grounding:
  - Core facts MUST come from context. Knowledge connects supported facts only.
  - Use only context portions relevant to query; ignore unrelated portions in mixed retrieval.
  - No invented claims, examples, causes, consequences, recommendations.
  - Conflicting/mixed-topic context: ignore unrelated portions, do not blend.
  - Preserve numbers, dates, percentages, measurements exactly. No rounding, paraphrasing, generalizing.

4. Formatting & Language:
  - CRITICAL: respond in user's query language. English query -> English answer, even if sources mix languages.
  - Format as {response_type}.
  - Markdown only when it materially improves clarity.
  - `Single Paragraph`: exactly one concise paragraph; no headings, no bullets.
  - `Bullet Points`: bullets only, compact.
  - `Multiple Paragraphs`: compact, no boilerplate.

5. Citations and References:
  - When the **Reference Document List** is non-empty, add an inline citation marker `[n]` (`n` = `reference_id`) after each factual claim its chunk's `reference_id` supports.
  - Use only `reference_id` values from the provided list. No invented IDs.
  - No raw field names, JSON keys, or internal ids like `reference_id` itself; the `[n]` marker is sufficient.
  - When the **Reference Document List** is empty, do not add citations or a `### References` section.

{user_prompt}
---Context---

{context_data}
"""







PROMPTS['naive_rag_response'] = """---Role---

Expert RAG assistant. Answer using ONLY the **Context**.

---Goal---

Direct, well-structured answer; synthesize Document Chunks from **Context**.

---Instructions---

1. Answer Strategy:
  - START with direct answer. No "Based on the context..." preamble.
  - Extract relevant facts from chunks; synthesize into coherent response.
  - Time/phase questions: cover the starting state, major transitions, and later/current state when context supports.
  - "What is X" / single-fact: state fact in first sentence, then context.
  - For yes/no questions, start the answer with "Yes" or "No" when context supports a binary judgment.
  - Lessons, recommendations, conclusions: quote the exact statement, not surrounding discussion.

2. Content Priority:
  - FIRST: answer core question with supported facts.
  - SECOND: minimum supporting detail for clarity.
  - THIRD: partial-support questions: answer the supported part, note unsupported gaps.
  - List answers in single paragraph: keep each supported item explicit and separate them cleanly with semicolons; no narrative blending.
  - Multi-part questions: address each one briefly rather than stopping after the first relevant point.
  - Category/list questions: prioritize major supported items; drop tangential, weak, speculative associations unless user requests exhaustive detail.
  - Say "insufficient information" only when context has nothing relevant.

3. Grounding:
  - Core facts MUST come from context. Knowledge connects supported facts only.
  - Use only context portions relevant to query; ignore unrelated portions in mixed retrieval.
  - No invented claims, examples, causes, consequences, recommendations.
  - Conflicting/mixed-topic context: ignore unrelated portions, do not blend.

4. Formatting & Language:
  - CRITICAL: respond in user's query language. English query -> English answer, even if sources mix languages.
  - Format as {response_type}.
  - Markdown only when it materially improves clarity.
  - `Single Paragraph`: exactly one concise paragraph; no headings, no bullets.
  - `Bullet Points`: bullets only, compact.
  - `Multiple Paragraphs`: compact, no boilerplate.

5. Citations and References:
  - When the **Reference Document List** is non-empty, add an inline citation marker `[n]` (`n` = `reference_id`) after each factual claim its chunk's `reference_id` supports.
  - Use only `reference_id` values from the provided list. No invented IDs.
  - No raw field names, JSON keys, or internal ids like `reference_id` itself; the `[n]` marker is sufficient.
  - When the **Reference Document List** is empty, do not add citations or a `### References` section.

{user_prompt}
---Context---

{content_data}
"""











PROMPTS['kg_query_context'] = """
# Knowledge Graph Data (Entity)

```json
{entities_str}
```

# Knowledge Graph Data (Relationship)

```json
{relations_str}
```

# Document Chunks

Use chunks below to answer. Treat any IDs or metadata as internal bookkeeping; do not surface raw field names or raw ids.

```json
{text_chunks_str}
```

{reference_list_str}
"""

PROMPTS['naive_query_context'] = """
# Document Chunks

Use chunks below to answer. Treat any IDs or metadata as internal bookkeeping; do not surface raw field names or raw ids.

```json
{text_chunks_str}
```

{reference_list_str}
"""

PROMPTS['keywords_extraction'] = """---Role---
Expert keyword extractor for two-tiered RAG search.

---Goal---
Extract two keyword types from user query:

1. **high_level_keywords** (2-4): broad themes:
   - intent (e.g. "comparison", "relationship", "overview", "how does", "what is")
   - domain (e.g. "AI technology", "business strategy", "healthcare", "finance")
   - information type (e.g. "partnership details", "product features", "history", "impact")

2. **low_level_keywords** (1-4): specific entities EXPLICIT in query:
   - Companies/orgs: "OpenAI", "Microsoft", "FDA", "Tesla"
   - People: "Elon Musk", "Sam Altman", "Tim Cook"
   - Products/tech: "GPT-4", "iPhone", "Azure", "Keytruda"
   - Technical terms: "machine learning", "mRNA", "blockchain"
   - Locations: "Silicon Valley", "China", "California"

---Instructions---
1. **Output Format**: ONLY valid JSON. No explanatory text, no markdown code fences.
2. **Preserve Exact Names**: low-level keywords keep entity names verbatim (no "Keytruda" -> "drug").
3. **Derive from Query**: all keywords come from query itself. No invented related concepts.
4. **Intent**: for high-level, consider information TYPE (comparison? mechanism? results?).
5. **Edge Cases**: vague queries (e.g. "hello") return empty lists.
6. **Language**: keywords MUST be in {language}. Proper nouns keep original language.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""








PROMPTS['keywords_extraction_examples'] = [
    """Example 1 (Process/mechanism query):

Query: "How does the consensus algorithm work in distributed systems?"

Output:
{
  "high_level_keywords": ["consensus mechanism", "distributed coordination", "system design"],
  "low_level_keywords": ["consensus algorithm", "distributed systems"]
}

""",
    """Example 2 (Comparison query):

Query: "How does CRISPR-Cas9 gene editing compare to traditional methods?"

Output:
{
  "high_level_keywords": ["gene editing", "technology comparison", "methods comparison"],
  "low_level_keywords": ["CRISPR-Cas9"]
}

""",
    """Example 3 (Business relationship query):

Query: "What is the relationship between OpenAI and Microsoft?"

Output:
{
  "high_level_keywords": ["business relationship", "partnership", "AI technology"],
  "low_level_keywords": ["OpenAI", "Microsoft"]
}

""",
    """Example 4 (Person query):

Query: "Who is Elon Musk and what companies does he lead?"

Output:
{
  "high_level_keywords": ["biography", "leadership", "companies"],
  "low_level_keywords": ["Elon Musk"]
}

""",
    """Example 5 (Product features query):

Query: "What are the features of GPT-4?"

Output:
{
  "high_level_keywords": ["product features", "capabilities", "AI model"],
  "low_level_keywords": ["GPT-4"]
}

""",
    """Example 6 (Regulatory/approval query):

Query: "What drugs did the FDA approve for diabetes in 2024?"

Output:
{
  "high_level_keywords": ["drug approval", "regulatory approval", "diabetes treatment"],
  "low_level_keywords": ["FDA", "diabetes", "2024"]
}

""",
    """Example 7 (Quantitative/historical query):

Query: "What were the main causes of the 2008 financial crisis?"

Output:
{
  "high_level_keywords": ["financial crisis causes", "economic history", "systemic risk"],
  "low_level_keywords": ["2008 financial crisis"]
}

""",
    """Example 8 (Operational/process query):

Query: "What are the steps in the agile sprint planning process?"

Output:
{
  "high_level_keywords": ["sprint planning", "agile process", "workflow steps"],
  "low_level_keywords": ["agile", "sprint planning"]
}

""",
    """Example 9 (Pharma/CMC manufacturing query):

Query: "What is the closed system drug transfer device (CSTD) strategy in Bio?"

Output:
{
  "high_level_keywords": ["device strategy", "manufacturing process", "drug transfer"],
  "low_level_keywords": ["CSTD", "closed system drug transfer device", "Bio"]
}

""",
]

PROMPTS['orphan_connection_validation'] = """---Task---
Evaluate whether a meaningful relationship exists between two entities.

Orphan: {orphan_name} ({orphan_type}) - {orphan_description}
Candidate: {candidate_name} ({candidate_type}) - {candidate_description}
Similarity: {similarity_score}

Valid relationship types:
- direct: one uses/creates/owns the other
- industry: same sector
- competitive: direct competitors or alternatives
- temporal: versions, successors, historical connections
- dependency: one relies on / runs on the other

Output valid JSON only, no markdown:
{{"should_connect": bool, "confidence": float, "relationship_type": str|null, "relationship_keywords": str|null, "relationship_description": str|null, "reasoning": str}}

Rules:
- confidence: float in [0.0, 1.0]
- HIGH confidence (>=0.7): direct/explicit relationships only
- MEDIUM (0.4-0.69): strong implicit/industry
- LOW (<0.4): weak/tenuous
- should_connect=true only if confidence >= 0.6
- Similarity alone insufficient; explain relationship

Example (connected):
{{"should_connect": true, "confidence": 0.82, "relationship_type": "direct", "relationship_keywords": "framework, built-with", "relationship_description": "Django is a web framework written in Python", "reasoning": "Direct explicit relationship"}}

Example (not connected):
{{"should_connect": false, "confidence": 0.05, "relationship_type": null, "relationship_keywords": null, "relationship_description": null, "reasoning": "No logical connection"}}
"""







# HyDE (Hypothetical Document Embedding) prompt
# Generates a hypothetical answer to improve retrieval through semantic similarity
PROMPTS[
    'hyde_prompt'
] = """Knowledgeable assistant. Given a question, identify the specific aspect/facet asked about (e.g. physical form, mechanism, policy, cause, consequence, comparison) and write a short passage directly addressing that facet. Write as if from the reference-document section where the answer would naturally appear. Be concrete and factual; invent plausible details when needed.

Question: {query}

Write a 2-3 sentence passage on the specific aspect, using the language and framing of a knowledgeable document in that section:"""


# Conversation rewrite prompt
# Rewrites the latest user query as a standalone question using prior conversation context.
# Used at retrieval time when conversation_history is non-empty so downstream keyword extraction,
# HyDE, and vector search work on a self-contained query rather than a context-dependent fragment.
PROMPTS[
    'conversation_query_rewrite'
] = """Rewrite the latest user query as a standalone question using prior conversation history.

Rules:
- Output ONLY the rewritten query. No preamble, no explanation, no quoting.
- Resolve pronouns (it, this, that, them, these, those, he, she, they) using the most recent referent in history.
- Resolve elliptical references ("what about phase 2?", "how about for India?") by adding the topic from history.
- Preserve every concrete entity, number, date, and qualifier from the original query verbatim.
- If the query is already self-contained (no pronouns, no elliptical references), output it unchanged.
- Keep the rewritten query in the same language as the original.
- Output a single sentence; do not add a trailing period if the original lacked one.

---Conversation History---
{history}

---Latest Query---
{query}

---Standalone Rewrite---
"""

# Entity Review prompt for LLM-based entity resolution
# Used to determine if entity pairs refer to the same real-world entity
PROMPTS[
    'entity_review_system_prompt'
] = """Entity Resolution Specialist. Determine whether pairs of entity names refer to the same real-world entity.

---Guidelines---

**DO merge entities that are:**
- Abbreviations: "FDA" = "US Food and Drug Administration"
- Alternate names: "The Fed" = "Federal Reserve"
- Translations: "美联储" = "Federal Reserve"
- Typos/misspellings: "Dupixant" = "Dupixent"
- Name variations: "Jerome Powell" = "Fed Chair Powell"
- Shortened forms: "United States" = "United States of America"
- Company suffixes: "Apple" = "Apple Inc." = "Apple Inc" = "Apple Corporation"
- University variations: "Stanford" = "Stanford University" = "Stanford U"
- Government agencies: "SEC" = "Securities and Exchange Commission"

**DO NOT merge entities that are:**
- Similar but distinct: "Method 1" ≠ "Method 2"
- Parent/child concepts: "United States" ≠ "United States Stock Market"
- Related but different: "Apple Inc" ≠ "Apple Watch"
- Different instances: "Super Bowl LV" ≠ "Super Bowl LVI"
- Different semantic types: A fruit ≠ an organization
  Example: "apple" (fruit) ≠ "Apple Inc." (organization)
  Example: "Amazon" (river) ≠ "Amazon.com" (company)
- Type-mismatched entities: Always verify entity types match before confirming alias

---Output Format---

Per pair, return JSON object with:
- pair_id: pair number (1-indexed)
- same_entity: true/false
- canonical: preferred/canonical name (if same_entity=true, use the most complete/formal form)
- confidence: 0.0-1.0
- reasoning: brief decision rationale

Return JSON array of all results."""



PROMPTS['entity_review_user_prompt'] = """---Task---
Review the following entity pairs; mark which refer to the same real-world entity.

---Entity Pairs---
{pairs}

---Output---
Return JSON array, one entry per pair. Example:
[
  {{"pair_id": 1, "same_entity": true, "canonical": "Federal Reserve", "confidence": 0.95, "reasoning": "FRB is the official abbreviation for Federal Reserve Board"}},
  {{"pair_id": 2, "same_entity": false, "canonical": null, "confidence": 0.9, "reasoning": "These are distinct concepts - one is a country, the other is a financial market"}}
]"""

# Entity batch review prompt for reviewing multiple new entities against existing ones
PROMPTS['entity_batch_review_prompt'] = """---Task---
NEW entities extracted from a document. Each new entity has candidate EXISTING entities that may be the same.

Job: determine if each new entity matches any of its candidates.

---New Entities and Candidates---
{entity_candidates}

---Output Format---
Return a JSON array. For each new entity:
{{
  "new_entity": "<the new entity name>",
  "matches_existing": true/false,
  "canonical": "<existing entity name if match, else the new entity name>",
  "confidence": 0.0-1.0,
  "reasoning": "<brief explanation>"
}}

Only set matches_existing=true if you are confident they refer to the same real-world entity."""
