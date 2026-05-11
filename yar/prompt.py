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
        *   `entity_type`: one of `{entity_types}`. Choose the closest configured high-level type; do NOT output `Other`, `UNKNOWN`, or invent new types.
        *   `entity_description`: concise yet comprehensive description of the entity's attributes and activities, based *solely* on input text.
    *   **Output Format - Entities:** 4 `{tuple_delimiter}`-delimited fields, single line. First field MUST be literal `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`
    *   **Connectivity:** Every extracted entity MUST appear as `source_entity` or `target_entity` in at least one valid relation in the same output. If the text only provides isolated metadata, headings, or labels, omit that entity unless you can also extract its explicit relation.
    *   **Metadata authors:** Extract title-slide authors/presenters only when the text explicitly links them to a named document, report, presentation, or event; output that authorship/presentation relation. Otherwise omit the person to avoid an orphan metadata node.
    *   **Branding-only mentions:** Do NOT extract organizations that appear only in logos, headers, footers, copyright notices, or slide branding (e.g., `[SANOFI Logo]`) unless the text states a concrete relationship for that organization.
    *   **What NOT to Extract as Entities:**
        *   Do NOT extract numeric values, percentages, or metrics (e.g., "3.4% decline", "$10 billion", "40% reduction").
        *   Do NOT extract generic event descriptors (e.g., "market selloff", "price increase", "stock decline").
        *   Do NOT extract time periods or bare dates as entities (e.g., "Q3 2024", "this year", "midday trading"). In dated event bullets like "14th Nov => Delay of 2 mo communicated", extract the action/event phrase as the entity and keep the date in its description. Bare dates are never valid relation sources.
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
        *   EXCEPTION: labeled-role lists where the label is an action verb or role that applies to every listed item ARE relationships (e.g. "Communication to: Alice, Bob" means each person received that communication). Extract one relation per listed item using the nearest named action/event/document as source, or the literal label phrase as the source if no named source exists; extract that source as an event/document entity when needed. The listed items fill the label's role: in "Communication to:" or "sent to:" lists, each listed item is the target/recipient, never the source. Do NOT apply this exception to category headers like "Topics: AI, ML".
        *   For dated bullets like "14th Nov => Delay of 2 mo communicated" with a recipient list below, use the action/event phrase as the source entity (e.g. "Primary Stability Batch Delay Communication"), never the bare date. This also applies when the date is a section header on its own line followed by an action bullet. Keep the date as temporal context in descriptions.
        *   For issue/problem headings (e.g., "3mL final stopper Issue"), do NOT leave the issue as a standalone event. Prefer the affected artifact/product as the entity and connect it to the responsible supplier, organization, rejection, or use-impact stated in the text.
        *   For event bullets with action verbs but implicit objects (e.g., "assessed", "agreed", "aligned", "sent out"), extract a relation only if both endpoints are explicit named entities. If the object is missing, omit the relation rather than placing the action verb in `target_entity`.
        *   For "X on Y" event names, keep the event and object separate when the text treats Y as the reviewed/read/sent document or topic. Example: `IA Management TC` reviewed `IDC Pre-reads`, not one fused entity named `IA Management TC on IDC Pre-reads`.
        *   For meeting agreement/postponement bullets, extract the agreed action or decision as the target entity when it is explicit. Example: `GPT F2F Team Meeting` agreed on `EU Submission Postponement`.
        *   For wording-alignment bullets tied to a prior named meeting, connect the alignment event to that meeting (e.g., `IDC Slides Wording Alignment` followed `GPT F2F Team Meeting`) unless the slide deck itself is emitted as an entity.
        *   Events: PRIMARY action (won, hosted, occurred in), not every association.
        *   Sports: achievements (won, broke record), not participation alone.
        *   No duplicate relationships with different wording.
    *   **Canonical keyword discipline:** If a relationship matches a common type below, use the exact keyword form shown as the first `relationship_keywords` term. Add at most two additional specific terms only when the text states distinct semantics. Prefer this canonical backbone over near-synonyms such as "cleared by" vs "approved by" or "partnership" vs "partnered with".
    *   **Common Relationship Types to Look For:**
        *   **Organization-Product:** manufactures, develops, produces, sells, markets
        *   **Organization-Organization:** acquired, partnered with, collaborated with, merged with, invested in
        *   **Person-Organization:** leads, founded, CEO of, works at, directs
        *   **Product-Concept:** treats, targets, inhibits, blocks, activates
        *   **Organization-Event:** approved, authorized, sponsored, conducted
        *   **Person-Event:** won, achieved, broke record in, presented at (NOT just "participated in")
        *   **Event-Location:** held in, took place in, hosted by
        *   **Risk/Impact:** explicit "X poses risk to Y", "X can impact Y", "X mitigates risk to Y", or "X prevents Y" statements ARE relationships. Extract the risk source or mitigation as `source_entity`, the affected outcome as `target_entity`, and use action keywords such as `poses risk to`, `impacts`, `mitigates risk to`, or `prevents`.
        *   **Risk targets as entities:** when a risk/impact statement explicitly names domain outcomes (e.g., product quality, accurate dosing, device compatibility, regulatory updates), extract those outcomes as `concept` entities so the relation is searchable. Do not invent unstated risks, and do not extract loose adjectives or generic qualities without an explicit risk/impact relation.
    *   **N-ary Decomposition:** statements with >2 entities -> decompose to binary relationships.
        *   **Example:** "Pfizer and BioNTech developed the COVID-19 vaccine" → extract "Pfizer developed COVID-19 Vaccine" AND "BioNTech developed COVID-19 Vaccine" AND "Pfizer partnered with BioNTech"
    *   **Relationship Details:** for each binary relationship, extract:
        *   `source_entity`: ACTOR/SUBJECT performing the action (consistent naming with entities).
        *   `target_entity`: OBJECT/RECIPIENT (consistent naming with entities).
        *   `relationship_keywords`: action-oriented keywords ("manufactures", "treats", "leads", "approved"). Comma-separated. **NEVER use `{tuple_delimiter}` here.**
        *   `source_entity` and `target_entity` MUST exactly match `entity_name` values emitted in this same output. Relationship keywords are not endpoints. A relation with only 4 fields is invalid and MUST NOT be output.
        *   **Malformed 4-field pattern to avoid:** `relation{tuple_delimiter}Source{tuple_delimiter}action_verb{tuple_delimiter}description` means the action verb was incorrectly placed in `target_entity` and the true target is missing. Do NOT output that line. Output a corrected 5-field relation only when the input explicitly names the real target entity.
        *   `relationship_description`: concise explanation that includes at least one verbatim or near-verbatim phrase from the input text directly supporting the relationship.
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
    *   The first separator after `entity` or `relation` MUST also be exactly `{tuple_delimiter}`. Do NOT output similar-looking variants such as `<|##|>`, `<|#|` or `|#|>`; one character wrong makes the whole record invalid.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Relationships are **undirected** unless explicitly stated otherwise. Swapping source/target on an undirected relationship is NOT a new relationship.
    *   No duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all entities first, then all relationships.
    *   Among relationships, output the **most significant** (core meaning of input) first.
    *   **Final self-check before `{completion_delimiter}`:** every relation has exactly 5 fields; every `source_entity` and `target_entity` has a matching entity line; every entity appears as a relation source or target. Fix or omit any record that fails this check.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

7.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

9.  **Type Guidance:** when assigning `entity_type`, follow these per-type definitions consistently across the extraction (this prevents the same concept being labeled with different types in different chunks, which blocks downstream entity merging):
    *   **Person:** named individuals.
    *   **Organization:** companies, sites, facilities, business units, labs, institutions, regulatory authorities.
    *   **Location:** city, region, country, state, or other place name.
    *   **Event:** named sessions, reviews, audits, meetings, project events, milestones (e.g., "Risk Review CIR 15 March 2017", "FDA Type C Meeting").
    *   **Document:** named documents, templates, reports, plans, charts, pages, links, resource pages (e.g., "CAPA Plan", "IND Dossier", "SharePoint Resource Page").
    *   **Product:** named drug products, components, materials, intermediates when treated as items (e.g., "Fitusiran", "Drug Product 50mg PFP").
    *   **Technology:** tools, platforms, IT solutions, software systems, equipment when clearly technical systems.
    *   **Method:** procedures, methods, testing methods, analytical methods (e.g., "ddPCR Vector Genome Titer Assay", "HPLC Method").
    *   **Data:** concrete data items when explicitly named (e.g., "Stability Data Set 2024-Q1").
    *   **Artifact:** specific named non-document, non-product items such as indexes, supports, devices, materials, or structured objects (e.g., "Primary Stability Batch 256131").
    *   **Concept:** named abstract but specific operational concepts only when they are clearly named in the text. Use sparingly; do NOT use Concept for generic phrases.

10. **Relationship Verb Discipline:** prefer concise predicates from this canonical list when the text supports them (use the closest match; do not invent new generic verbs):
    *   accepted, approved, conducts, responsible for, located in, headquartered in, supplies, produces, uses, tests, manages, integrates, releases, packages, provides resource, is subject matter expert for, outlines, addresses, targets, converts to, derived from, formulated as, handles, performs, mitigates risk to, poses risk to, impacts.

11. **Recommended Extraction Patterns:**
    *   **Review/audit tables:** treat the named site/facility as Organization; the named review/audit title as Event; CAPA plans, templates, indexes, pages, and links as Document. Link the organization/site to the CAPA plan with `outlined`, `accepted`, `responsible for`, or `addresses`.
    *   **Supply chain mappings:** treat the named drug or program as Product; intermediate strands, supports, devices, and testing materials as Product or Artifact depending on how the chunk presents them; suppliers/manufacturers/release sites/test labs/service providers as Organization. Treat cities/countries only when attached to relations like `located in` or `headquartered in`. Avoid inventing facilities not explicitly stated.
    *   **Resource / contact pages:** treat named people as Person; the business unit or named session as Organization or Event depending on wording; SharePoint links, project charter templates, and resource pages as Document. Link people as `is subject matter expert for` the session/review; link documents as `provides resource` for the session/review.
    *   **Subcontracting:** when text describes subcontracted activities, represent the subcontracting provider AND the specific activity/item explicitly named.

12. **Quality Constraints:**
    *   Do not over-generate entities. Each entity must earn its place via at least one explicit relation.
    *   Do not create entities for generic phrases like "link", "resource", "testing", "release", "manufacturing", "supply chain", "quality system", "project charter", unless they are explicitly named document titles or specific named items in the chunk.
    *   Avoid duplicate or near-duplicate entities; choose ONE canonical form for the same named thing across the chunk.
    *   If uncertain whether something is an entity, omit rather than invent.

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
    *   **Known 4-field relation pattern:** if the prior extraction emitted `relation{tuple_delimiter}Source{tuple_delimiter}verb{tuple_delimiter}description` (action verb in target slot, no true target_entity), that record is malformed. Re-output the corrected 5-field record if a real target entity can be inferred from context; otherwise omit the relation entirely.
    *   Audit prior relation lines one by one. Any 4-field line with an action verb in the `target_entity` slot is missing a real target; correct it only if the target is explicitly named in the input text, otherwise omit it.
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
relation{tuple_delimiter}KEYNOTE-024{tuple_delimiter}Keytruda{tuple_delimiter}evaluated{tuple_delimiter}KEYNOTE-024 evaluated Keytruda and demonstrated reduced disease progression.
relation{tuple_delimiter}Keytruda{tuple_delimiter}PD-1{tuple_delimiter}blocks{tuple_delimiter}Keytruda blocks PD-1 to help the immune system fight cancer cells.
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
relation{tuple_delimiter}OpenAI{tuple_delimiter}Artificial General Intelligence{tuple_delimiter}researches{tuple_delimiter}OpenAI's collaboration with Microsoft focuses on building safe and beneficial artificial general intelligence.
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
    """<Entity_types>
["Person","Organization","Location","Event","Concept","Method","Technology","Product","Document","Data","Artifact"]

<Input Text>
```
# Sequence of events
* 14th Nov => Delay of 2 mo of primary stability batch communicated
    * Communication to: Miller, Torres, Kim, Patel
```

<Output>
entity{tuple_delimiter}Primary Stability Batch Delay Communication{tuple_delimiter}event{tuple_delimiter}Primary Stability Batch Delay Communication is the 14th Nov event where a two-month primary stability batch delay was communicated to listed recipients.
entity{tuple_delimiter}Miller{tuple_delimiter}person{tuple_delimiter}Miller is a recipient of the Primary Stability Batch Delay Communication.
entity{tuple_delimiter}Torres{tuple_delimiter}person{tuple_delimiter}Torres is a recipient of the Primary Stability Batch Delay Communication.
entity{tuple_delimiter}Kim{tuple_delimiter}person{tuple_delimiter}Kim is a recipient of the Primary Stability Batch Delay Communication.
entity{tuple_delimiter}Patel{tuple_delimiter}person{tuple_delimiter}Patel is a recipient of the Primary Stability Batch Delay Communication.
relation{tuple_delimiter}Primary Stability Batch Delay Communication{tuple_delimiter}Miller{tuple_delimiter}sent to{tuple_delimiter}The Primary Stability Batch Delay Communication was sent to Miller on 14th Nov.
relation{tuple_delimiter}Primary Stability Batch Delay Communication{tuple_delimiter}Torres{tuple_delimiter}sent to{tuple_delimiter}The Primary Stability Batch Delay Communication was sent to Torres on 14th Nov.
relation{tuple_delimiter}Primary Stability Batch Delay Communication{tuple_delimiter}Kim{tuple_delimiter}sent to{tuple_delimiter}The Primary Stability Batch Delay Communication was sent to Kim on 14th Nov.
relation{tuple_delimiter}Primary Stability Batch Delay Communication{tuple_delimiter}Patel{tuple_delimiter}sent to{tuple_delimiter}The Primary Stability Batch Delay Communication was sent to Patel on 14th Nov.
{completion_delimiter}

""",
    """<Entity_types>
["Person","Organization","Location","Event","Concept","Method","Technology","Product","Document","Data","Artifact"]

<Input Text>
```
14th November 2018
Delay of primary stability batch by 2 months communicated to project team.
Communication to: Chen, Novak, Okafor
```

<Output>
entity{tuple_delimiter}Primary Stability Batch Delay Communication{tuple_delimiter}event{tuple_delimiter}Primary Stability Batch Delay Communication is the 14th November 2018 communication of a two-month delay in the primary stability batch.
entity{tuple_delimiter}Chen{tuple_delimiter}person{tuple_delimiter}Chen is a recipient of the Primary Stability Batch Delay Communication.
entity{tuple_delimiter}Novak{tuple_delimiter}person{tuple_delimiter}Novak is a recipient of the Primary Stability Batch Delay Communication.
entity{tuple_delimiter}Okafor{tuple_delimiter}person{tuple_delimiter}Okafor is a recipient of the Primary Stability Batch Delay Communication.
relation{tuple_delimiter}Primary Stability Batch Delay Communication{tuple_delimiter}Chen{tuple_delimiter}communicated to{tuple_delimiter}The Primary Stability Batch Delay Communication was communicated to Chen on 14th November 2018.
relation{tuple_delimiter}Primary Stability Batch Delay Communication{tuple_delimiter}Novak{tuple_delimiter}communicated to{tuple_delimiter}The Primary Stability Batch Delay Communication was communicated to Novak on 14th November 2018.
relation{tuple_delimiter}Primary Stability Batch Delay Communication{tuple_delimiter}Okafor{tuple_delimiter}communicated to{tuple_delimiter}The Primary Stability Batch Delay Communication was communicated to Okafor on 14th November 2018.
{completion_delimiter}

""",
    """<Entity_types>
["Person","Organization","Location","Event","Concept","Method","Technology","Product","Document","Data","Artifact"]

<Input Text>
```
# SARA CS information escalation – lessons learned
C.Dette, iCMC NPP LL Cross sharing, 27th Feb 2019
[SANOFI Logo]

# 3mL final stopper Issue
Delivery of supplier West showed fibers and metal particles.
Final stopper batches from West could not be released.
Final stopper could not be used for manufacturing of primary stability batches.

# Sequence of events
* 23rd Nov => iCMC team meeting
    * Assessment EU submission intermediate/final stopper
* 28th/29th Nov => GPT F2F team meeting
    * Agreement on postponement of submission in EU
* 29th Nov => wording on IDC slides align after GPT meeting ended
* 30th Nov => Midday IDC pre-reads sent out
* 3rd Dec => IA management TC on IDC pre-reads
    * Participants: Boensel, Charreau
```

<Output>
entity{tuple_delimiter}SARA CS Information Escalation{tuple_delimiter}document{tuple_delimiter}SARA CS Information Escalation is the 27th Feb 2019 lessons-learned document for iCMC NPP LL Cross sharing.
entity{tuple_delimiter}C.Dette{tuple_delimiter}person{tuple_delimiter}C.Dette is the author of the SARA CS Information Escalation document.
entity{tuple_delimiter}West{tuple_delimiter}organization{tuple_delimiter}West is the supplier whose final stopper batches showed fibers and metal particles.
entity{tuple_delimiter}3mL Final Stopper{tuple_delimiter}artifact{tuple_delimiter}3mL Final Stopper is the affected stopper artifact whose West batches could not be released or used for primary stability batch manufacturing.
entity{tuple_delimiter}iCMC Team Meeting{tuple_delimiter}event{tuple_delimiter}iCMC Team Meeting is the 23rd Nov event that assessed the EU submission intermediate/final stopper.
entity{tuple_delimiter}GPT F2F Team Meeting{tuple_delimiter}event{tuple_delimiter}GPT F2F Team Meeting is the 28th/29th Nov event where postponement of the EU submission was agreed.
entity{tuple_delimiter}EU Submission Postponement{tuple_delimiter}event{tuple_delimiter}EU Submission Postponement is the agreed postponement of the EU submission.
entity{tuple_delimiter}IDC Slides Wording Alignment{tuple_delimiter}event{tuple_delimiter}IDC Slides Wording Alignment is the 29th Nov event where wording on IDC slides was aligned after the GPT meeting.
entity{tuple_delimiter}IDC Pre-reads{tuple_delimiter}document{tuple_delimiter}IDC Pre-reads are the documents sent out on 30th Nov for the IA Management TC.
entity{tuple_delimiter}IA Management TC{tuple_delimiter}event{tuple_delimiter}IA Management TC is the 3rd Dec management teleconference that reviewed IDC pre-reads.
entity{tuple_delimiter}Boensel{tuple_delimiter}person{tuple_delimiter}Boensel is a participant in the IA Management TC.
entity{tuple_delimiter}Charreau{tuple_delimiter}person{tuple_delimiter}Charreau is a participant in the IA Management TC.
relation{tuple_delimiter}C.Dette{tuple_delimiter}SARA CS Information Escalation{tuple_delimiter}authored{tuple_delimiter}C.Dette authored the SARA CS Information Escalation document dated 27th Feb 2019.
relation{tuple_delimiter}West{tuple_delimiter}3mL Final Stopper{tuple_delimiter}supplied contaminated batches{tuple_delimiter}West supplied final stopper batches that showed fibers and metal particles.
relation{tuple_delimiter}iCMC Team Meeting{tuple_delimiter}3mL Final Stopper{tuple_delimiter}assessed{tuple_delimiter}The iCMC Team Meeting assessed the EU submission intermediate/final stopper.
relation{tuple_delimiter}GPT F2F Team Meeting{tuple_delimiter}EU Submission Postponement{tuple_delimiter}agreed on{tuple_delimiter}The GPT F2F Team Meeting agreed on postponement of the EU submission.
relation{tuple_delimiter}IDC Slides Wording Alignment{tuple_delimiter}GPT F2F Team Meeting{tuple_delimiter}followed{tuple_delimiter}IDC Slides Wording Alignment followed the GPT F2F Team Meeting.
relation{tuple_delimiter}IDC Pre-reads{tuple_delimiter}IA Management TC{tuple_delimiter}sent for{tuple_delimiter}IDC Pre-reads were sent out for the IA Management TC on IDC pre-reads.
relation{tuple_delimiter}IA Management TC{tuple_delimiter}IDC Pre-reads{tuple_delimiter}reviewed{tuple_delimiter}IA Management TC reviewed IDC pre-reads on 3rd Dec.
relation{tuple_delimiter}IA Management TC{tuple_delimiter}Boensel{tuple_delimiter}included participant{tuple_delimiter}Boensel participated in the IA Management TC.
relation{tuple_delimiter}IA Management TC{tuple_delimiter}Charreau{tuple_delimiter}included participant{tuple_delimiter}Charreau participated in the IA Management TC.
{completion_delimiter}

""",
    """<Entity_types>
["Person","Organization","Location","Event","Concept","Method","Technology","Product","Document","Data","Artifact"]

<Input Text>
```
# Drug Device Combination Product (DDCP)
Performance of 3 Different Stopper Designs
| Device (supply for) | Stopper Design | DDCP Performance Tests |
| Clinic: Ph3 | Existing | Passed |
| Clinic: Interchangeability - Primary Stability | Intermediate | Passed |
| Primary Stability - Commercial | Final | Ongoing, positive initial results |

To improve compatibility of rubber stoppers with the drug in the 3 mL cartridge, optimized paraffin free material and an adapted design are being performed.
```

<Output>
entity{tuple_delimiter}Drug Device Combination Product{tuple_delimiter}product{tuple_delimiter}Drug Device Combination Product is tested with three stopper designs for DDCP performance.
entity{tuple_delimiter}3 mL Cartridge{tuple_delimiter}artifact{tuple_delimiter}3 mL Cartridge is the cartridge whose drug compatibility is improved by optimized rubber stopper designs.
entity{tuple_delimiter}Stopper Design{tuple_delimiter}artifact{tuple_delimiter}Stopper Design is the adapted rubber stopper design family tested for compatibility with the drug in the 3 mL cartridge.
entity{tuple_delimiter}Clinic Ph3 Stopper{tuple_delimiter}artifact{tuple_delimiter}Clinic Ph3 Stopper is the existing stopper design for Clinic Ph3 that passed DDCP performance tests.
entity{tuple_delimiter}Interchangeability Primary Stability Stopper{tuple_delimiter}artifact{tuple_delimiter}Interchangeability Primary Stability Stopper is the intermediate stopper design that passed DDCP performance tests.
entity{tuple_delimiter}Primary Stability Commercial Stopper{tuple_delimiter}artifact{tuple_delimiter}Primary Stability Commercial Stopper is the final stopper design with ongoing positive initial DDCP performance results.
relation{tuple_delimiter}Drug Device Combination Product{tuple_delimiter}Clinic Ph3 Stopper{tuple_delimiter}tested with{tuple_delimiter}Drug Device Combination Product was tested with the Clinic Ph3 Stopper and the test passed.
relation{tuple_delimiter}Drug Device Combination Product{tuple_delimiter}Interchangeability Primary Stability Stopper{tuple_delimiter}tested with{tuple_delimiter}Drug Device Combination Product was tested with the Interchangeability Primary Stability Stopper and the test passed.
relation{tuple_delimiter}Drug Device Combination Product{tuple_delimiter}Primary Stability Commercial Stopper{tuple_delimiter}tested with{tuple_delimiter}Drug Device Combination Product testing with the Primary Stability Commercial Stopper is ongoing with positive initial results.
relation{tuple_delimiter}Stopper Design{tuple_delimiter}3 mL Cartridge{tuple_delimiter}improves compatibility with{tuple_delimiter}Stopper Design improves compatibility with the drug in the 3 mL Cartridge.
relation{tuple_delimiter}Clinic Ph3 Stopper{tuple_delimiter}Stopper Design{tuple_delimiter}is existing variant of{tuple_delimiter}Clinic Ph3 Stopper is the existing variant of Stopper Design.
relation{tuple_delimiter}Interchangeability Primary Stability Stopper{tuple_delimiter}Stopper Design{tuple_delimiter}is intermediate variant of{tuple_delimiter}Interchangeability Primary Stability Stopper is the intermediate variant of Stopper Design.
relation{tuple_delimiter}Primary Stability Commercial Stopper{tuple_delimiter}Stopper Design{tuple_delimiter}is final variant of{tuple_delimiter}Primary Stability Commercial Stopper is the final variant of Stopper Design.
{completion_delimiter}

""",
    """<Entity_types>
["Person","Organization","Location","Event","Concept","Method","Technology","Product","Document","Data","Artifact"]

<Input Text>
```
# Executive summary
* The use of CSTDs can pose risks to product quality and accurate dosing.
* The CSTD Strategy Recommendations mitigate risks to product quality and accurate dosing.
```

<Output>
entity{tuple_delimiter}CSTD{tuple_delimiter}artifact{tuple_delimiter}CSTD is a closed system transfer device whose use can pose risks to product quality and accurate dosing.
entity{tuple_delimiter}Product Quality{tuple_delimiter}concept{tuple_delimiter}Product Quality is the affected outcome that CSTD use can put at risk and that CSTD Strategy Recommendations mitigate.
entity{tuple_delimiter}Accurate Dosing{tuple_delimiter}concept{tuple_delimiter}Accurate Dosing is the affected outcome that CSTD use can put at risk and that CSTD Strategy Recommendations mitigate.
entity{tuple_delimiter}CSTD Strategy Recommendations{tuple_delimiter}document{tuple_delimiter}CSTD Strategy Recommendations mitigate risks to product quality and accurate dosing.
relation{tuple_delimiter}CSTD{tuple_delimiter}Product Quality{tuple_delimiter}poses risk to{tuple_delimiter}The use of CSTDs can pose risks to product quality.
relation{tuple_delimiter}CSTD{tuple_delimiter}Accurate Dosing{tuple_delimiter}poses risk to{tuple_delimiter}The use of CSTDs can pose risks to accurate dosing.
relation{tuple_delimiter}CSTD Strategy Recommendations{tuple_delimiter}Product Quality{tuple_delimiter}mitigates risk to{tuple_delimiter}CSTD Strategy Recommendations mitigate risks to product quality.
relation{tuple_delimiter}CSTD Strategy Recommendations{tuple_delimiter}Accurate Dosing{tuple_delimiter}mitigates risk to{tuple_delimiter}CSTD Strategy Recommendations mitigate risks to accurate dosing.
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

Expert RAG assistant. Answer the user question using ONLY the retrieved Context.

---Goal---

Direct, well-structured answer integrating Knowledge Graph entities/relationships and Document Chunks from the Context.

---Instructions---

You are given a user question plus a "retrieved context" block. The retrieved context is a structured block containing (a) knowledge-graph entities/relationships, (b) document chunks, and (c) a reference list mapping citation indexes to source documents.

Follow these instructions precisely:

1) Output policy (strict grounding)
- Answer ONLY using facts that appear in the retrieved context (knowledge-graph relationships and/or document chunks).
- START with the direct answer. No "Based on the context..." preamble.
- Knowledge Graph entities/relationships: primary source for entity facts, attributes, connections.
- Document Chunks: evidence, quotes, supporting detail. If a chunk includes `evidence_spans`, treat those spans as the first-pass support for the answer and verify each claim against them before using broader chunk text.
- Knowledge-graph relationships alone do not stand as answers; they may provide indirect support when corroborated by document chunks. Cite the chunk that contains the supporting evidence, not the relationship itself.
- Never use world knowledge or training knowledge. Every factual claim must be supportable by the retrieved context.
- No speculation: avoid hedging phrases such as "may imply", "could suggest", "likely", "appears to". Either a claim is directly supported by the context or it is dropped.
- Conflicting/mixed-topic context: ignore unrelated portions, do not blend.
- Preserve numbers, dates, percentages, measurements exactly as written. Do not invent a specific number when the context uses a vague qualifier ("more than one", "multiple", "several") — quote the qualifier verbatim instead. Distinguish quantities from identifiers: a string that looks like a number but is part of a section/batch/lot/code label is not a count.

2) Citations and References (MANDATORY when Reference Document List is non-empty)
- Every factual sentence MUST end with `[n]` where `n` is the `reference_id` from the **Reference Document List** that backs it. Sentences without a citation marker will be treated as unsupported and rejected.
- Multiple references on one sentence: combine as `[1][2]` or `[1, 2]`, not separate sentences.
- Use only `reference_id` values from the provided list. No invented IDs. No raw field names like `reference_id` itself; the bracket marker is the only citation form.
- Example: "The compound was tested at site A in 2023 [1]. The protocol was revised after the audit findings [1, 2]."
- When the **Reference Document List** is empty, do not add citations or a `### References` section.

3) Handling missing information
- Refusal pattern (use this when the retrieved context does not actually answer the question):
  - First sentence: name what the context covers (the topic / dimension that IS described).
  - Second sentence: name what the question asked but the context does not cover (the topic / dimension that is NOT described).
  - Third sentence (optional): explicit "Insufficient information for <missing aspect>".
- Do not append speculative half-sentences after the refusal. Do not pivot to training-data knowledge for the missing aspect even when the question phrasing implies a "well-known" answer.
- Do not "partially answer" with unsupported additions; you may restate what is available in the context and note what is missing.
- Hard rules to prevent training-data leakage on common bait questions. Treat this as a four-step decision flow, not a soft preference:
  - Step 1 — classify the question type. Does it ask for any of:
    (a) molecular target / pathway / receptor / binding partner / enzyme of a drug, compound, protein, or biological entity;
    (b) mechanism of action / mode of action / detection principle / how does X work / underlying chemistry, biology, or instrumentation principle;
    (c) historical origins / discovery story / academic theory / pharmacology background / who invented or founded;
    (d) formal regulatory designation (orphan, breakthrough, fast-track, accelerated approval, equivalent agency designations).
  - Step 2 — if the answer is YES to ANY of (a)-(d), search the retrieved context for an explicit statement of THAT EXACT KIND of fact about the queried entity. The entity being mentioned in the chunks is NOT sufficient. Indications, clinical uses, project history, supply chain notes, qualification status, acceptance criteria, regulatory submissions, partner names, action plans, and meeting context do NOT count as mechanism / target / origin / classification.
  - Step 3 — if you find such a statement, quote it verbatim and cite the chunk.
  - Step 4 — if you do not find such a statement, return the refusal pattern. Do not "describe" the queried entity from background knowledge. Do not substitute its indication for its target. Do not synthesize a mechanism from adjacent but non-mechanism chunks. Do not turn the absence of mechanism content into a list of what IS in the context that could be misread as a partial answer.
- Anti-patterns you must not produce when refusing one of (a)-(d):
  - "X targets [disease/condition]" — that is the indication, not the molecular target.
  - "Y is a method that partitions the sample into droplets ... fluorescent probes detect ..." — that is training-data description, not grounding, when the chunks only mention Y was run or what its result was.
  - "The origins of Z trace to [program/year]" when the chunks only place Z in that program — being referenced by a program is not the same as being originated by it.

4) Style/verbosity
- CRITICAL: respond in user's query language. English query -> English answer, even if sources mix languages.
- Format as {response_type}.
- Markdown only when it materially improves clarity.
- `Single Paragraph`: exactly one concise paragraph; no headings, no bullets.
- `Bullet Points`: bullets only, compact.
- `Multiple Paragraphs`: compact, no boilerplate.
- Do not include extra background unless it is directly supported by the context.

5) Task-type inference and answer construction
- "What is X" / single-fact: state fact in first sentence, then context.
- For yes/no questions, start the answer with "Yes" or "No" when context supports a binary judgment.
- Time/phase questions: cover the starting state, major transitions, and later/current state when context supports.
- Multi-part questions: address each part explicitly, not a single blend; address each one briefly rather than stopping after the first relevant point.
- Comparison questions ("compare X with Y", "differences between A and B"): structure the answer as two attributed sides drawn from their respective source chunks. Use phrases that trace back to the chunk you're attributing to; if a sub-claim cannot be traced to its named source, drop it. **If only one side of the comparison is in the retrieved context, present that side and add a single sentence noting the other side wasn't retrieved — do not refuse the entire answer.** Source fidelity: when the query names specific documents, only attribute content under that name when the supporting chunk's reference_id maps to that document — never relabel content from one source as belonging to another.
- List/count/enumeration questions: include all supported items, not first few; start by restating the requested subject using the user's wording. If context contains a numbered list matching the request, preserve source numbering and include each listed item explicitly. Include only items the context explicitly identifies as members of the requested category. List answers in single paragraph: keep each supported item explicit and separate them cleanly with semicolons; no narrative blending.
- Consequence/impact/result questions: enumerate every supported item before any narrative summary.
- Factual lookup grounded in a specific table/section:
  - Identify the exact entity / step / section name referenced by the question.
  - Extract the corresponding value or statement exactly as written (including inequalities like "≥", units, scientific notation).
  - Provide the value as the direct answer, with a citation to the chunk/table where it appears.
- Lessons, recommendations, conclusions: quote the exact statement, not surrounding discussion.
- Templates, formulas, syntax patterns: quote verbatim.

6) Safety against hallucination
- If context evidence contradicts or is absent, do not guess.
- If you cannot find exact support for the requested statement, return the refusal pattern from section 3.
- Say "insufficient information" only when context has nothing relevant; partial-context comparisons or list questions should follow the section 5 rules instead.

{user_prompt}
---Context---

{context_data}
"""


PROMPTS['naive_rag_response'] = """---Role---

Expert RAG assistant. Answer the user question using ONLY the retrieved Context.

---Goal---

Direct, well-structured answer synthesizing Document Chunks from the Context.

---Instructions---

You are given a user question plus a "retrieved context" block. The retrieved context contains (a) document chunks and (b) a reference list mapping citation indexes to source documents.

Follow these instructions precisely:

1) Output policy (strict grounding)
- Answer ONLY using facts that appear in the retrieved context (document chunks).
- START with the direct answer. No "Based on the context..." preamble.
- If a chunk includes `evidence_spans`, treat those spans as the first-pass support for the answer and verify each claim against them before using broader chunk text.
- Never use world knowledge or training knowledge. Every factual claim must be supportable by the retrieved context.
- No speculation: avoid hedging phrases such as "may imply", "could suggest", "likely", "appears to". Either a claim is directly supported by the context or it is dropped.
- Conflicting/mixed-topic context: ignore unrelated portions, do not blend.
- Preserve numbers, dates, percentages, measurements exactly as written. Do not invent a specific number when the context uses a vague qualifier ("more than one", "multiple", "several") — quote the qualifier verbatim instead. Distinguish quantities from identifiers: a string that looks like a number but is part of a section/batch/lot/code label is not a count.

2) Citations and References (MANDATORY when Reference Document List is non-empty)
- Every factual sentence MUST end with `[n]` where `n` is the `reference_id` from the **Reference Document List** that backs it. Sentences without a citation marker will be treated as unsupported and rejected.
- Multiple references on one sentence: combine as `[1][2]` or `[1, 2]`, not separate sentences.
- Use only `reference_id` values from the provided list. No invented IDs. No raw field names like `reference_id` itself; the bracket marker is the only citation form.
- Example: "The compound was tested at site A in 2023 [1]. The protocol was revised after the audit findings [1, 2]."
- When the **Reference Document List** is empty, do not add citations or a `### References` section.

3) Handling missing information
- Refusal pattern (use this when the retrieved context does not actually answer the question):
  - First sentence: name what the context covers (the topic / dimension that IS described).
  - Second sentence: name what the question asked but the context does not cover (the topic / dimension that is NOT described).
  - Third sentence (optional): explicit "Insufficient information for <missing aspect>".
- Do not append speculative half-sentences after the refusal. Do not pivot to training-data knowledge for the missing aspect even when the question phrasing implies a "well-known" answer.
- Do not "partially answer" with unsupported additions; you may restate what is available in the context and note what is missing.
- Hard rules to prevent training-data leakage on common bait questions. Treat this as a four-step decision flow, not a soft preference:
  - Step 1 — classify the question type. Does it ask for any of:
    (a) molecular target / pathway / receptor / binding partner / enzyme of a drug, compound, protein, or biological entity;
    (b) mechanism of action / mode of action / detection principle / how does X work / underlying chemistry, biology, or instrumentation principle;
    (c) historical origins / discovery story / academic theory / pharmacology background / who invented or founded;
    (d) formal regulatory designation (orphan, breakthrough, fast-track, accelerated approval, equivalent agency designations).
  - Step 2 — if the answer is YES to ANY of (a)-(d), search the retrieved context for an explicit statement of THAT EXACT KIND of fact about the queried entity. The entity being mentioned in the chunks is NOT sufficient. Indications, clinical uses, project history, supply chain notes, qualification status, acceptance criteria, regulatory submissions, partner names, action plans, and meeting context do NOT count as mechanism / target / origin / classification.
  - Step 3 — if you find such a statement, quote it verbatim and cite the chunk.
  - Step 4 — if you do not find such a statement, return the refusal pattern. Do not "describe" the queried entity from background knowledge. Do not substitute its indication for its target. Do not synthesize a mechanism from adjacent but non-mechanism chunks. Do not turn the absence of mechanism content into a list of what IS in the context that could be misread as a partial answer.
- Anti-patterns you must not produce when refusing one of (a)-(d):
  - "X targets [disease/condition]" — that is the indication, not the molecular target.
  - "Y is a method that partitions the sample into droplets ... fluorescent probes detect ..." — that is training-data description, not grounding, when the chunks only mention Y was run or what its result was.
  - "The origins of Z trace to [program/year]" when the chunks only place Z in that program — being referenced by a program is not the same as being originated by it.

4) Style/verbosity
- CRITICAL: respond in user's query language. English query -> English answer, even if sources mix languages.
- Format as {response_type}.
- Markdown only when it materially improves clarity.
- `Single Paragraph`: exactly one concise paragraph; no headings, no bullets.
- `Bullet Points`: bullets only, compact.
- `Multiple Paragraphs`: compact, no boilerplate.
- Do not include extra background unless it is directly supported by the context.

5) Task-type inference and answer construction
- "What is X" / single-fact: state fact in first sentence, then context.
- For yes/no questions, start the answer with "Yes" or "No" when context supports a binary judgment.
- Time/phase questions: cover the starting state, major transitions, and later/current state when context supports.
- Multi-part questions: address each part explicitly, not a single blend; address each one briefly rather than stopping after the first relevant point.
- Comparison questions ("compare X with Y", "differences between A and B"): structure the answer as two attributed sides drawn from their respective source chunks. Use phrases that trace back to the chunk you're attributing to; if a sub-claim cannot be traced to its named source, drop it. **If only one side of the comparison is in the retrieved context, present that side and add a single sentence noting the other side wasn't retrieved — do not refuse the entire answer.** Source fidelity: when the query names specific documents, only attribute content under that name when the supporting chunk's reference_id maps to that document — never relabel content from one source as belonging to another.
- List/count/enumeration questions: include all supported items, not first few; start by restating the requested subject using the user's wording. If context contains a numbered list matching the request, preserve source numbering and include each listed item explicitly. Include only items the context explicitly identifies as members of the requested category. List answers in single paragraph: keep each supported item explicit and separate them cleanly with semicolons; no narrative blending.
- Consequence/impact/result questions: enumerate every supported item before any narrative summary.
- Factual lookup grounded in a specific table/section:
  - Identify the exact entity / step / section name referenced by the question.
  - Extract the corresponding value or statement exactly as written (including inequalities like "≥", units, scientific notation).
  - Provide the value as the direct answer, with a citation to the chunk/table where it appears.
- Lessons, recommendations, conclusions: quote the exact statement, not surrounding discussion.
- Templates, formulas, syntax patterns: quote verbatim.

6) Safety against hallucination
- If context evidence contradicts or is absent, do not guess.
- If you cannot find exact support for the requested statement, return the refusal pattern from section 3.
- Say "insufficient information" only when context has nothing relevant; partial-context comparisons or list questions should follow the section 5 rules instead.

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
Evidence spans, when present, are extractive support hints from the same chunk or graph relation. Prefer them for grounding before using broader chunk or relation text.

```json
{text_chunks_str}
```

{reference_list_str}
"""

PROMPTS['naive_query_context'] = """
# Document Chunks

Use chunks below to answer. Treat any IDs or metadata as internal bookkeeping; do not surface raw field names or raw ids.
Evidence spans, when present, are extractive support hints from the same chunk. Prefer them for grounding before using the full chunk text.

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
   - information type (e.g. "partnership details", "product features", "history", "background", "chronology", "timeline", "impact")

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
    """Example 10 (Dates/milestones/timeline query):

Query: "What are the dates or milestones mentioned during the project freeze period?"

Output:
{
  "high_level_keywords": ["history", "background", "chronology", "timeline"],
  "low_level_keywords": ["project freeze"]
}

""",
    """Example 11 (Temporal/when query):

Query: "When did the FDA approve Sarclisa and what were the subsequent submission events?"

Output:
{
  "high_level_keywords": ["history", "regulatory timeline", "approval chronology"],
  "low_level_keywords": ["FDA", "Sarclisa", "approval", "submission"]
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
- authorship: one authored, prepared, or presented the other document/event
- participation: one participated in, attended, or was a named stakeholder of the other meeting/event

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
- Measurement unit case variants: "3 mL" = "3 ml", "mL" = "ml" (unit symbol case is not semantically significant)
- Initial-prefixed names: "P.Charreau" = "Charreau", "C.Dette" = "Dette" when context confirms the same person

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


PROMPTS['relation_predicate_review'] = """---Role---
Relation Predicate Review Specialist. Canonicalize relationship predicate keywords without changing relation endpoints or evidence.

---Input---
A JSON array named relation_items. Each item has:
- src: source entity
- tgt: target entity
- candidate_keywords: extracted predicate keywords, ordered by current priority
- evidence_spans: extractive evidence already captured for the relation

relation_items:
{relation_items}

Allowed canonical predicates and aliases:
{allowed_predicates}

---Rules---
1. Do NOT change src or tgt.
2. Do NOT invent evidence; evidence_spans are context only.
3. primary MUST be one of candidate_keywords or one of the allowed canonical aliases.
4. canonical_keywords MUST reuse candidate keywords or allowed canonical aliases only.
5. Prefer one concise canonical action as primary; keep secondary predicates only when evidence states distinct relation semantics.
6. Order canonical_keywords most-canonical-first and include primary as the first item.
7. If unsure, return the original candidate_keywords in the same order with confidence below 0.6.

---Output---
Return ONLY a JSON array, one object per input item:
[
  {{
    "src": "<unchanged source entity>",
    "tgt": "<unchanged target entity>",
    "canonical_keywords": ["<primary>", "<optional secondary>"],
    "primary": "<first canonical keyword>",
    "confidence": 0.0,
    "reasoning": "<brief rationale>"
  }}
]"""
