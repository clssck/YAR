from __future__ import annotations

from typing import Any

PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS['DEFAULT_TUPLE_DELIMITER'] = '<|#|>'
PROMPTS['DEFAULT_COMPLETION_DELIMITER'] = '<|COMPLETE|>'

PROMPTS['entity_extraction_system_prompt'] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, do not add new entity type and classify it as `Other`.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
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
    *   **Identification:** Identify direct, clearly stated relationships between previously extracted entities. Focus on concrete, actionable relationships.
    *   **Quality Guidelines:**
        *   **Explicit connections only:** Extract relationships where the text clearly states how entities are connected. Do NOT create relationships just because entities appear in the same sentence.
        *   **Action-oriented relationships:** Focus on relationships with clear action verbs (manufactures, approved, leads, treats, partnered with) rather than vague associations.
        *   **Avoid over-extraction:** Do not extract relationships between every possible entity pair. Only extract when there is a meaningful, stated connection. Typically, a paragraph should yield 2-5 key relationships, not dozens.
    *   **What NOT to Extract:**
        *   Do NOT extract "featured", "included", "part of" relationships - these are too vague.
        *   Do NOT extract relationships between concepts that are just listed together (e.g., "skateboarding and surfing" mentioned together does NOT create a relationship between them).
        *   For events: Focus on the PRIMARY action (won, hosted, occurred in) not every possible association.
        *   For sports: Focus on achievements (won, broke record) not participation alone.
        *   Do NOT duplicate the same relationship with different wording.
    *   **Common Relationship Types to Look For:**
        *   **Organization-Product:** manufactures, develops, produces, sells, markets
        *   **Organization-Organization:** acquired, partnered with, collaborated with, merged with, invested in
        *   **Person-Organization:** leads, founded, CEO of, works at, directs
        *   **Product-Concept:** treats, targets, inhibits, blocks, activates
        *   **Organization-Event:** approved, authorized, sponsored, conducted
        *   **Person-Event:** won, achieved, broke record in, presented at (NOT just "participated in")
        *   **Event-Location:** held in, took place in, hosted by
    *   **N-ary Relationship Decomposition:** If a statement involves more than two entities, decompose into binary relationships.
        *   **Example:** "Pfizer and BioNTech developed the COVID-19 vaccine" → extract "Pfizer developed COVID-19 Vaccine" AND "BioNTech developed COVID-19 Vaccine" AND "Pfizer partnered with BioNTech"
    *   **Relationship Details:** For each binary relationship, extract:
        *   `source_entity`: The ACTOR/SUBJECT performing the action (use consistent naming with entities above).
        *   `target_entity`: The OBJECT/RECIPIENT of the action (use consistent naming with entities above).
        *   `relationship_keywords`: One or more **action-oriented keywords** (e.g., "manufactures", "treats", "leads", "approved"). Separate multiple keywords with comma `,`. **DO NOT use `{tuple_delimiter}` within this field.**
        *   `relationship_description`: A concise explanation of the relationship.
    *   **Relationship Direction (CRITICAL):**
        *   source_entity PERFORMS the action ON target_entity
        *   "Sam Altman leads OpenAI" → source=Sam Altman, target=OpenAI, keyword=leads
        *   "OpenAI developed GPT-4" → source=OpenAI, target=GPT-4, keyword=developed
        *   "Microsoft invested in OpenAI" → source=Microsoft, target=OpenAI, keyword=invested in
        *   WRONG: source=OpenAI, target=Sam Altman, keyword=leads (direction reversed!)
    *   **Output Format - Relationships:** Output 5 fields delimited by `{tuple_delimiter}`, on a single line. First field must be `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

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
Extract entities and relationships from the input text in Data to be Processed below.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted and presented.
4.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

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
Extract entities and relationships from EACH of the input text chunks below.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2.  **Per-Chunk Output:** For EACH chunk, output the exact header line `[CHUNK: <chunk_id>]` using the same chunk id provided in the input, followed by the extracted entities and relationships for that chunk, then `{completion_delimiter}`. Process chunks in the order given. Use each provided chunk id exactly once, only for its matching chunk. Do not rename, omit, merge, duplicate, or reorder chunk ids.
3.  **Empty Sections Are Required:** If a chunk has no entities or relationships, still output its exact `[CHUNK: <chunk_id>]` header and then `{completion_delimiter}` on the next line.
4.  **Output Content Only:** Output *only* chunk headers and extracted entities/relationships. No introductory or concluding remarks.
5.  **Output Language:** Ensure the output language is {language}. Proper nouns must be kept in their original language.

---Data to be Processed---
<Entity_types>
[{entity_types}]

{batch_input_texts}

<Output>
"""

PROMPTS['entity_continue_extraction_user_prompt'] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
    *   If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
    *   If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
3.  **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
4.  **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
5.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
7.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

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
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Synthesize multiple descriptions of a given entity or relation into a single, concise, and informative summary.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object appears on a new line.
2. Output Format: Return plain text only. No markdown, headers, or formatting. No introductory or concluding remarks.
3. Conciseness Priority:
  - Eliminate ALL redundancy. If the same fact appears multiple times, include it only ONCE.
  - Prioritize unique, specific facts over generic statements.
  - Use compact phrasing. Avoid filler words and verbose constructions.
  - Target: capture maximum information in minimum words.
4. Content Priority (in order):
  - Core identity: What is this entity? (type, category, primary function)
  - Key relationships: Who/what is it connected to and how?
  - Distinguishing facts: What makes this entity unique or notable?
  - Secondary details: Only if space permits within token limit.
5. Context & Objectivity:
  - Write from objective, third-person perspective.
  - Start with the entity/relation name for immediate clarity.
6. Conflict Handling:
  - If descriptions refer to genuinely different entities with the same name, summarize each separately.
  - For temporal conflicts (historical changes), prefer the most recent information unless historical context is critical.
7. Length Constraint: Maximum {summary_length} tokens. Prioritize brevity while preserving key facts.
8. Language: Output in {language}. Keep proper nouns in original language if no standard translation exists.

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

You are an expert AI assistant synthesizing information from a knowledge base. Answer user queries accurately using ONLY information from the provided **Context**.

---Goal---

Generate a direct, well-structured answer integrating facts from Knowledge Graph and Document Chunks in the **Context**.

---Instructions---

1. Answer Strategy:
  - START with the direct answer. Do not begin with "Based on the context..." or similar preamble.
  - Use Knowledge Graph entities/relationships as your primary source for facts about entities, their attributes, and connections.
  - Use Document Chunks for detailed evidence, quotes, and supporting context.
  - Synthesize both sources into a coherent response.
  - If the question asks how something changed over time or compares phases, cover the starting state, major transitions, and later/current state when the context supports them.
  - When answering questions with multiple parts or sub-questions, address each part explicitly rather than giving a single blended answer.
  - When the question asks "what is X" or can be answered with a single fact, state that fact in the first sentence. Then provide supporting context.
  - For yes/no questions, start the answer with "Yes" or "No" when the context supports a binary judgment.
  - When citing specific lessons learned, recommendations, or conclusions from the context, state the exact conclusion or recommendation rather than only describing the surrounding discussion.
  - When the context contains a specific template, formula, or syntax pattern (e.g., "Due to X, the risk Y could impact Z"), quote it verbatim rather than paraphrasing or summarizing the surrounding discussion about it.

2. Content Priority:
  - FIRST: Answer the core question directly using supported facts.
  - SECOND: Add only the supporting details needed to make the answer clear.
  - THIRD: If the context supports only part of the question, answer that part and explicitly note what the context does not establish.
  - When the question asks for a list, count, or enumeration, include all items supported by the context rather than stopping after the first few.
  - When the question asks about consequences, impacts, or results of an action, enumerate every specific item mentioned in the context before providing any narrative summary.
  - When list-style answers must fit into a single paragraph, keep each supported item explicit and separate them cleanly with semicolons instead of blending them into narrative prose.
  - If the question has multiple supported parts, address each one briefly rather than stopping after the first relevant point.
  - For category or list questions, prioritize the major or primary supported items and leave out tangential, weak, or speculative associations unless the user explicitly asks for exhaustive detail.
  - Only say "insufficient information" when the context contains nothing relevant to the user's question.

3. Grounding:
  - Core facts MUST come from the context. Use your knowledge only to connect supported facts fluently.
  - Focus on context directly relevant to the user's question. If retrieved context covers multiple topics, use only the portions that address the query.
  - Do not introduce claims, examples, causes, consequences, or recommendations that are not directly supported by the relevant context.
  - If retrieved context conflicts or mixes topics, ignore the unrelated portions instead of blending them into the answer.
  - Preserve specific numbers, dates, percentages, and measurements from the context exactly as stated. Do not round, paraphrase, or generalize quantitative information.

4. Formatting & Language:
  - CRITICAL: The response MUST be in the same language as the user query. If the query is in English, respond ONLY in English even if source documents contain other languages.
  - Format the response as {response_type}.
  - Use Markdown only when it materially improves clarity.
  - For `Single Paragraph`, return exactly one concise paragraph with no headings or bullet points.
  - For `Bullet Points`, return bullets only and keep each bullet compact.
  - For `Multiple Paragraphs`, keep the answer compact and avoid boilerplate.

5. Citations and References:
  - Do not add inline citations like [1] or a `### References` section unless the caller explicitly asks for them.
  - Do not mention raw field names, JSON keys, or internal ids such as `reference_id` unless the caller explicitly asks for them.
  - The system returns source references separately when needed.

{user_prompt}
---Context---

{context_data}
"""

PROMPTS['naive_rag_response'] = """---Role---

You are an expert AI assistant synthesizing information from a knowledge base. Answer user queries accurately using ONLY information from the provided **Context**.

---Goal---

Generate a direct, well-structured answer integrating facts from Document Chunks in the **Context**.

---Instructions---

1. Answer Strategy:
  - START with the direct answer. Do not begin with "Based on the context..." or similar preamble.
  - Extract relevant facts from Document Chunks and synthesize them into a coherent response.
  - If the question asks how something changed over time or compares phases, cover the starting state, major transitions, and later/current state when the context supports them.
  - When the question asks "what is X" or can be answered with a single fact, state that fact in the first sentence. Then provide supporting context.
  - For yes/no questions, start the answer with "Yes" or "No" when the context supports a binary judgment.
  - When citing specific lessons learned, recommendations, or conclusions from the context, state the exact conclusion or recommendation rather than only describing the surrounding discussion.

2. Content Priority:
  - FIRST: Answer the core question directly using supported facts.
  - SECOND: Add only the supporting details needed to make the answer clear.
  - THIRD: If the context supports only part of the question, answer that part and explicitly note what the context does not establish.
  - When list-style answers must fit into a single paragraph, keep each supported item explicit and separate them cleanly with semicolons instead of blending them into narrative prose.
  - If the question has multiple supported parts, address each one briefly rather than stopping after the first relevant point.
  - For category or list questions, prioritize the major or primary supported items and leave out tangential, weak, or speculative associations unless the user explicitly asks for exhaustive detail.
  - Only say "insufficient information" when the context contains nothing relevant to the user's question.

3. Grounding:
  - Core facts MUST come from the context. Use your knowledge only to connect supported facts fluently.
  - Focus on context directly relevant to the user's question. If retrieved context covers multiple topics, use only the portions that address the query.
  - Do not introduce claims, examples, causes, consequences, or recommendations that are not directly supported by the relevant context.
  - If retrieved context conflicts or mixes topics, ignore the unrelated portions instead of blending them into the answer.

4. Formatting & Language:
  - CRITICAL: The response MUST be in the same language as the user query. If the query is in English, respond ONLY in English even if source documents contain other languages.
  - Format the response as {response_type}.
  - Use Markdown only when it materially improves clarity.
  - For `Single Paragraph`, return exactly one concise paragraph with no headings or bullet points.
  - For `Bullet Points`, return bullets only and keep each bullet compact.
  - For `Multiple Paragraphs`, keep the answer compact and avoid boilerplate.

5. Citations and References:
  - Do not add inline citations like [1] or a `### References` section unless the caller explicitly asks for them.
  - Do not mention raw field names, JSON keys, or internal ids such as `reference_id` unless the caller explicitly asks for them.
  - The system returns source references separately when needed.

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

Use the chunk content below to answer the question. Treat any IDs or metadata as internal bookkeeping; do not mention raw field names or raw ids in the answer.

```json
{text_chunks_str}
```

# Reference Document List

This source list is internal retrieval metadata. Only surface source identifiers when the caller explicitly asks for citations or raw reference ids.

{reference_list_str}
"""

PROMPTS['naive_query_context'] = """
# Document Chunks

Use the chunk content below to answer the question. Treat any IDs or metadata as internal bookkeeping; do not mention raw field names or raw ids in the answer.

```json
{text_chunks_str}
```

# Reference Document List

This source list is internal retrieval metadata. Only surface source identifiers when the caller explicitly asks for citations or raw reference ids.

{reference_list_str}
"""

PROMPTS['keywords_extraction'] = """---Role---
You are an expert keyword extractor. Your task is to analyze user queries and extract keywords optimized for a two-tiered RAG search system.

---Goal---
Extract two distinct types of keywords from the user query:

1. **high_level_keywords** (2-4 keywords): Broad, thematic concepts that capture:
   - The query's main goal or intent (e.g., "comparison", "relationship", "overview", "how does", "what is")
   - The subject area or domain (e.g., "AI technology", "business strategy", "healthcare", "finance")
   - The type of information sought (e.g., "partnership details", "product features", "history", "impact")

2. **low_level_keywords** (1-4 keywords): Specific entities that appear EXPLICITLY in the query:
   - Company/Organization names: "OpenAI", "Microsoft", "FDA", "Tesla"
   - Person names: "Elon Musk", "Sam Altman", "Tim Cook"
   - Product/Technology names: "GPT-4", "iPhone", "Azure", "Keytruda"
   - Technical terms: "machine learning", "mRNA", "blockchain"
   - Location names: "Silicon Valley", "China", "California"

---Instructions---
1. **Output Format**: Output ONLY a valid JSON object. No explanatory text, no markdown code fences.
2. **Preserve Exact Names**: Low-level keywords must preserve entity names exactly as written (don't replace "Keytruda" with "drug").
3. **Derive from Query**: All keywords must come from the query itself. Do not invent related concepts.
4. **Think About Intent**: For high-level keywords, consider what TYPE of information the user wants (comparison? mechanism? results?).
5. **Handle Edge Cases**: For vague queries (e.g., "hello"), return empty lists.
6. **Language**: Keywords MUST be in {language}. Proper nouns keep original language.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS['keywords_extraction_examples'] = [
    """Example 1 (Drug mechanism query):

Query: "What is the mechanism of action of Fitusiran for hemophilia treatment?"

Output:
{
  "high_level_keywords": ["mechanism of action", "therapeutic mechanism", "hemophilia treatment"],
  "low_level_keywords": ["Fitusiran", "hemophilia"]
}

""",
    """Example 2 (Regulatory/approval query):

Query: "What drugs did the FDA approve for diabetes in 2024?"

Output:
{
  "high_level_keywords": ["drug approval", "regulatory approval", "diabetes treatment"],
  "low_level_keywords": ["FDA", "diabetes", "2024"]
}

""",
    """Example 3 (Technology comparison query):

Query: "How does CRISPR-Cas9 gene editing compare to traditional methods?"

Output:
{
  "high_level_keywords": ["gene editing", "technology comparison", "methods comparison"],
  "low_level_keywords": ["CRISPR-Cas9"]
}

""",
    """Example 4 (Drug efficacy comparison):

Query: "Compare the efficacy of Keytruda vs Opdivo for lung cancer"

Output:
{
  "high_level_keywords": ["drug comparison", "efficacy comparison", "cancer treatment"],
  "low_level_keywords": ["Keytruda", "Opdivo", "lung cancer"]
}

""",
    """Example 5 (Tech/Business query):

Query: "What is the relationship between OpenAI and Microsoft?"

Output:
{
  "high_level_keywords": ["business relationship", "partnership", "AI technology"],
  "low_level_keywords": ["OpenAI", "Microsoft"]
}

""",
    """Example 6 (Person query):

Query: "Who is Elon Musk and what companies does he lead?"

Output:
{
  "high_level_keywords": ["biography", "leadership", "companies"],
  "low_level_keywords": ["Elon Musk"]
}

""",
    """Example 7 (Product query):

Query: "What are the features of GPT-4?"

Output:
{
  "high_level_keywords": ["product features", "capabilities", "AI model"],
  "low_level_keywords": ["GPT-4"]
}

""",
    """Example 8 (CMC/Pharma manufacturing query):

Query: "What is the closed system drug transfer device (CSTD) strategy in Bio?"

Output:
{
  "high_level_keywords": ["device strategy", "manufacturing process", "drug transfer"],
  "low_level_keywords": ["CSTD", "closed system drug transfer device", "Bio"]
}

""",
    """Example 9 (Pharma product presentation query):

Query: "What is the presentation of Sarclisa (isatuximab)?"

Output:
{
  "high_level_keywords": ["drug presentation", "product formulation", "packaging configuration"],
  "low_level_keywords": ["Sarclisa", "isatuximab"]
}

""",
    """Example 10 (Manufacturing batch analysis query):

Query: "What are the minimum information fields on the batch analysis table for an AAV product?"

Output:
{
  "high_level_keywords": ["batch analysis", "record fields", "data requirements"],
  "low_level_keywords": ["AAV", "batch analysis table"]
}

""",
]

PROMPTS['orphan_connection_validation'] = """---Task---
Evaluate if a meaningful relationship exists between two entities.

Orphan: {orphan_name} ({orphan_type}) - {orphan_description}
Candidate: {candidate_name} ({candidate_type}) - {candidate_description}
Similarity: {similarity_score}

Valid relationship types:
- direct: One uses/creates/owns the other
- industry: Both operate in the same sector
- competitive: Direct competitors or alternatives
- temporal: Versions, successors, or historical connections
- dependency: One relies on/runs on the other

Output valid JSON only (no markdown):
{{"should_connect": bool, "confidence": float, "relationship_type": str|null, "relationship_keywords": str|null, "relationship_description": str|null, "reasoning": str}}

Rules:
- confidence must be a number between 0.0 and 1.0
- HIGH confidence (>=0.7) only for direct/explicit relationships
- MEDIUM confidence (0.4-0.69) for strong implicit/industry relationships
- LOW confidence (<0.4) for weak/tenuous connections
- should_connect=true only when confidence >= 0.6
- Similarity alone is not sufficient; explain the relationship

Example (connected):
{{"should_connect": true, "confidence": 0.82, "relationship_type": "direct", "relationship_keywords": "framework, built-with", "relationship_description": "Django is a web framework written in Python", "reasoning": "Direct explicit relationship"}}

Example (not connected):
{{"should_connect": false, "confidence": 0.05, "relationship_type": null, "relationship_keywords": null, "relationship_description": null, "reasoning": "No logical connection"}}
"""

# HyDE (Hypothetical Document Embedding) prompt
# Generates a hypothetical answer to improve retrieval through semantic similarity
PROMPTS[
    'hyde_prompt'
] = """You are a knowledgeable assistant. Given the following question, identify the specific aspect or facet being asked about (e.g., physical form, mechanism, policy, cause, consequence, comparison) and write a short passage that directly addresses that facet. Write as if this passage appears in the section of a reference document where the answer would naturally be found. Be concrete and factual, imagining plausible details where needed.

Question: {query}

Write a 2-3 sentence passage focused on the specific aspect of the question, using the language and framing a knowledgeable document would use in the relevant section:"""

# Entity Review prompt for LLM-based entity resolution
# Used to determine if entity pairs refer to the same real-world entity
PROMPTS[
    'entity_review_system_prompt'
] = """You are an Entity Resolution Specialist. Your task is to determine whether pairs of entity names refer to the same real-world entity.

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

For each pair, return a JSON object with:
- pair_id: The pair number (1-indexed)
- same_entity: true/false
- canonical: The preferred/canonical name (if same_entity=true, use the most complete/formal name)
- confidence: 0.0-1.0 (how certain you are)
- reasoning: Brief explanation of your decision

Return a JSON array of all results."""

PROMPTS['entity_review_user_prompt'] = """---Task---
Review the following entity pairs and determine which refer to the same real-world entity.

---Entity Pairs---
{pairs}

---Output---
Return a JSON array with your analysis for each pair. Example format:
[
  {{"pair_id": 1, "same_entity": true, "canonical": "Federal Reserve", "confidence": 0.95, "reasoning": "FRB is the official abbreviation for Federal Reserve Board"}},
  {{"pair_id": 2, "same_entity": false, "canonical": null, "confidence": 0.9, "reasoning": "These are distinct concepts - one is a country, the other is a financial market"}}
]"""

# Entity batch review prompt for reviewing multiple new entities against existing ones
PROMPTS['entity_batch_review_prompt'] = """---Task---
You have a list of NEW entities extracted from a document. For each new entity, I will provide candidate EXISTING entities that may be the same.

Your job: Determine if each new entity matches any of its candidates.

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
