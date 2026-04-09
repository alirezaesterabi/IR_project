# Presentation Source

## Purpose of this document

This document is a long-form narrative of what we have done in the project so far, written in a way that can later be turned into presentation notes. It is not meant to replace the `README.md`, and it is not a planning checklist like `documents/implementation_phases.md`. Instead, it explains the project as a story: what problem we started from, what we discovered, what we built, why we made certain decisions, and what remains to be completed.

All seven query types are now included in this version so the document can serve as a complete presentation source. The evaluation section is still deliberately partial, because the final evaluation write-up should be updated once the remaining evaluation work is finished.

---

## 1. Where the project started

We began with the idea that sanctions search is not a normal search problem. In a standard web or document retrieval setting, a missed result is inconvenient. In sanctions compliance, a missed result can have serious regulatory and legal consequences. That framing shaped the project from the beginning. We were not trying to build a generic search demo. We were trying to design and implement a system that could retrieve potentially relevant sanctioned entities from a large and messy real-world dataset, where names vary, aliases appear across sources, identifiers matter, and some queries are much harder than a simple keyword lookup.

The dataset choice reflected that goal. We worked with OpenSanctions, which aggregates a large number of international watchlists, sanctions lists, and related records. The attraction of the dataset was not just its size, but its complexity. The records are rich, nested, and uneven. Some entities are easy to identify because they contain strong identifiers such as IMO numbers or registration numbers. Others are much more difficult because they are described indirectly through aliases, free text, linked records, or cross-source duplication. That complexity made the dataset a good fit for an IR project because it forced us to think carefully about retrieval design rather than relying on one simple method.

From the beginning, we also understood that one retrieval strategy would probably not be enough. Some searches are deterministic and benefit from exact matching. Others are more lexical and benefit from BM25-style term weighting. Others depend on semantic similarity and would not be handled well by a purely lexical model. Because of that, the project naturally evolved into a multi-stage retrieval system rather than a single-model pipeline.

---

## 2. Understanding the dataset before building anything

The first real step was exploration. Before building a parser, index, or retrieval model, we needed to understand what kind of object an OpenSanctions record actually is. That work was done through the initial exploration notebook, where the aim was not to optimise anything yet, but to answer basic structural questions.

We needed to know what fields appear consistently, which fields are sparse, which attributes are useful for direct matching, and which ones are more suitable for descriptive search. This mattered because retrieval quality in a project like this depends heavily on document representation. If the wrong fields are flattened into searchable text, useful information can be lost. If structured identifiers are treated as ordinary text, exact-match behavior becomes unreliable. If everything is pushed into one bag of tokens without distinction, the system becomes harder to interpret and harder to improve.

That exploration phase clarified several important facts. The records were not flat rows that could simply be indexed as-is. They were nested entity objects with multiple property lists, often including names, aliases, previous names, descriptive fields, sanctions sub-objects, address information, country codes, and list-specific program identifiers. This meant that preprocessing was not going to be a trivial cleaning step. It would be a transformation stage that converted complex graph-like records into an IR-friendly document format.

The exploration stage also confirmed why this was an interesting retrieval challenge. The same entity could be represented differently across sources. Some records were rich in descriptive text, while others were dominated by identifiers and metadata. Some entity types, such as vessels, could be found through maritime identifiers. Others depended much more on names and aliases. That variety pushed us toward a design that could support multiple query behaviors rather than a one-size-fits-all ranking function.

---

## 3. Why preprocessing became the foundation of the project

Once we understood the structure of the dataset, preprocessing became the central engineering task. The raw file was too large and too nested to be useful directly. We needed a pipeline that could read the source data safely, transform each record into a searchable representation, and preserve the distinction between exact-match fields, free text, and metadata.

The first challenge was scale. The raw OpenSanctions file is large enough that loading it naively into memory would be wasteful and, depending on the environment, potentially unsafe. For that reason, the parser was designed as a streaming JSONL reader. Instead of reading the entire dataset into RAM, it yields one record at a time. That decision was practical, but it also fit the wider logic of the project: this was a system built for large-scale retrieval, so the ingestion stage needed to reflect that from the start.

The second challenge was representation. A raw record contains many fields, but not all of them should be treated equally. We therefore flattened each entity into a processed document with a few clear components:

- a stable `doc_id`
- a human-readable `caption`
- a `schema` indicating entity type
- a lexical `text_blob` for classical retrieval
- `tokens` derived from that text blob
- raw `identifiers` kept separately for exact matching
- structured `metadata` such as country, dataset, and program ID
- an `embedding_text` field for dense retrieval

This was one of the most important design decisions in the whole project. Instead of indexing raw JSON, we built an intermediate document format that was deliberately shaped around retrieval needs. That gave us a cleaner separation between the raw data model and the IR model.

Another important decision concerned how different kinds of text should be processed. We did not apply the same NLP treatment to every field. Names and aliases were normalised, but not lemmatised, because names are identity-bearing strings and overly aggressive linguistic processing can damage them. In contrast, descriptive fields such as notes and summaries were lemmatised so that lexical variation could be reduced and recall improved. Identifiers were left untouched because they need exact matching. This split was not cosmetic. It reflected a core understanding of the dataset: a vessel name, a sanction reason, and an IMO number do not behave the same way in retrieval, so they should not be processed as if they do.

Methodologically, the preprocessing pipeline had several layers. First, the parser streamed one JSON record at a time from the raw file. Second, the document builder pulled out specific groups of fields rather than flattening everything blindly. Name-like fields such as `name`, `alias`, and `previousName` were grouped together because they support entity lookup behavior. Keyword-like fields such as positions, sectors, topics, and legal forms were handled separately because they are useful lexical signals but do not require full NLP. Description-style fields such as `notes`, `description`, and `summary` were treated as free text. Nested sanctions objects were also traversed so that authority and reason information could be pulled into the searchable representation. Address sub-objects were flattened as well so that location text could still contribute to retrieval.

At the token-processing level, the method was deliberately selective rather than generic. Text was first normalised into a consistent Unicode form, lowercased, stripped of punctuation, and whitespace-normalised. For Latin-script text, accents were removed where useful, while non-Latin characters were preserved more carefully so that script-specific information was not destroyed. Name fields then passed through light normalisation only, because preserving the identity signal was more important than linguistic abstraction. Description fields went through tokenisation, stop-word filtering, and spaCy lemmatisation so that different surface forms such as "evading", "evade", and "evasion" could contribute more consistently to lexical retrieval. Identifier fields bypassed all of this and were stored as raw values. Finally, the lexical `text_blob` was assembled in a meaningful order, names first, then keywords, then description text, then sanctions and address content, while metadata such as country, datasets, and programme identifiers was stored separately for filtering and later dense enrichment.

This matters for the presentation because it shows that preprocessing was not just "cleaning the data." It was the stage where we turned a complex sanctions record into multiple retrieval views: an exact-match view for identifiers, a lexical view for BM25 and TF-IDF, a metadata view for filters, and later a natural-language-style view for dense retrieval.

We also separated structured metadata from main searchable text wherever possible. Program IDs, countries, and dataset information are often better suited for filtering, grouping, or constrained retrieval than for free-text ranking. Preserving them in metadata gave the system more control and made later query-specific logic possible, especially for jurisdiction-style queries.

By the end of this stage, we had moved from a raw nested corpus to a consistent processed corpus. That was a major turning point in the project, because everything that followed, classical retrieval, dense retrieval, pooling, and evaluation, depended on this processed representation.

---

## 4. Building a smaller working corpus before scaling

Although the long-term goal was always to work with a large real dataset, we did not want every iteration to depend on full-scale processing. That would have slowed down experimentation and made debugging unnecessarily expensive. So an important practical step was to work with subsets during development.

This was not just a convenience feature. It was part of the development strategy. Using a smaller sample allowed us to validate parser behavior, inspect processed documents manually, test preprocessing decisions, and run retrieval experiments more quickly. In an IR project, iteration speed matters because ranking quality often depends on many small choices. If every change requires a full large-scale rebuild, progress becomes much slower.

The preprocessing validation notebook played an important role here. It let us inspect individual records and confirm that the flattened output matched our intentions. That mattered because mistakes at this stage would propagate directly into the indexes. A poor document representation would not always fail loudly; it would often just produce quietly worse retrieval. The validation step reduced that risk.

---

## 5. Defining the query space instead of assuming one search behavior

A major conceptual step in the project was recognising that "search" in this domain is not a single task. Users do not always query sanctions data in the same way. Sometimes they know an exact identifier. Sometimes they know a name or alias. Sometimes they are looking for all records of the same entity across datasets. Sometimes they want a filtered set, such as entities tied to a certain jurisdiction or sanctions programme. And sometimes they describe a target indirectly in natural language.

That is why the query design phase was so important. Rather than evaluating the system on an arbitrary list of keywords, we defined multiple query types and treated them as different retrieval behaviors. This gave the project a clearer experimental structure and made it easier to justify different retrieval components.

In practice, the query-generation work separated the easier-to-derive query types from the more judgement-heavy ones. Query types 1, 2, 5, and 6 were generated in a more systematic way because they could be anchored in identifiable fields or metadata, while types 3, 4, and 7 required more deliberate phrasing and human judgement. The full query space can be explained as follows:

- Type 1: exact identifier lookup. Example: `IMO9296822`.
- Type 2: name or alias lookup. Example: `Angelica Schulte`.
- Type 3: semantic or descriptive search where the user does not know the exact entity name. Example: `Russian crude oil tanker evading sanctions in the Baltic Sea`.
- Type 4: relational or graph-style query about links between entities. Example: `What vessels is Ketee Co Ltd linked to?`
- Type 5: cross-dataset deduplication across multiple sources. Example: `SAGITTA sanctions all sources`.
- Type 6: jurisdiction or filter query constrained by authority, programme, or country. Example: `OFAC sanctioned vessels Russian oil 2025`.
- Type 7: RAG-style narrative query where the final answer is not just a ranked list, but a grounded synthesis from retrieved records. Example: `Summarise all sanctions on the SAGITTA vessel`.

The query-generation notebook reflects that distinction clearly. It generates large batches of the more structured query types and exports them in organised form. At the same time, it treats the more interpretive query types, especially types 3, 4, and 7, as cases requiring more careful human input. That was a sensible separation because not all query types can be generated with the same level of automatic confidence.

What mattered most in this phase was not just producing query files, but making explicit what the system was expected to handle. Once those query types were defined, the rest of the architecture became easier to justify. Identifier retrieval exists because type 1 exists. Lexical ranking matters because type 2 and part of type 6 depend on it. Pooling becomes necessary because type 3 and type 4 do not have obvious ground truth. Type 7 justifies a generation-oriented layer on top of retrieval rather than a ranking-only output. In that sense, the query taxonomy helped turn a broad project idea into a concrete IR problem.

---

## 6. Implementing the classical retrieval lane

With a processed corpus and an explicit query space in place, the next step was to build the classical retrieval side of the system. This was the most natural starting point because classical IR gives strong, interpretable baselines and handles a large share of sanctions search behavior well.

We implemented three main retrieval behaviors in this lane: BM25, TF-IDF, and exact identifier lookup. Each one served a different purpose.

The identifier retriever addressed the most deterministic search case. If a user has an IMO number, registration number, tax number, or similar exact field, the system should not depend on approximate lexical ranking to find the right entity. It should use the raw identifiers directly. Preserving those fields separately in preprocessing made this possible.

BM25 acted as the primary lexical ranker. This made sense because BM25 is still the standard baseline for first-stage retrieval in many IR systems, especially when exact word overlap and term weighting matter. For sanctions search, this is especially useful when the query contains names, aliases, or important domain terms that also appear in the target records.

TF-IDF was included as a baseline rather than as the expected winner. Its role was comparative. It helped establish whether more advanced lexical weighting was actually providing a benefit over a simpler vector-space representation. That is useful in a course project because it strengthens the experimental logic of the system: we are not just claiming BM25 is a good choice, we are placing it against a simpler lexical alternative.

This stage was important not only because it produced working retrievers, but because it gave the project a strong baseline layer. Before adding dense retrieval or fusion, we needed to know that the system could already perform well on exact and lexical query types. Classical retrieval formed that backbone.

---

## 7. Preparing the corpus for dense retrieval rather than forcing dense models onto lexical text

Dense retrieval was not added just to make the project look more modern. It addressed a real limitation of classical retrieval. A purely lexical model struggles when relevant documents do not share vocabulary with the query, or when the query is a descriptive paraphrase rather than a direct name or identifier match. That gap is especially visible in semantic and investigative query styles.

However, we did not simply reuse the lexical `text_blob` and call it a dense representation. Instead, we created a dedicated `embedding_text` field. This was a meaningful design decision. The lexical document representation had been built for BM25 and TF-IDF, where token overlap, weighting, and exact field content matter. Dense encoders benefit from a more natural-language-like representation that turns structured metadata into readable context.

So the project introduced an `embedding_text` builder that constructs a more descriptive string from each processed document. It includes entity caption, schema, selected metadata such as countries and sanctions programmes, some identifier information where useful, and a fallback to lexical text when the structured content is too sparse. This gave the dense models a cleaner semantic input than a raw flattened token bag would have done.

That was one of the more subtle but important project decisions. We did not treat dense retrieval as a drop-in replacement for lexical retrieval. We adapted the representation to the model family. This strengthened the overall architecture, because each retrieval lane could now operate on a document view suited to its own strengths.

---

## 8. Building the dense retrieval lane

After preparing the corpus for dense retrieval, the next step was to actually encode it and store it in a searchable vector index. This work was carried out through the dense retrieval notebooks, which moved the project from theoretical hybrid design to a working two-lane retrieval setup.

The dense pipeline encoded the corpus into sentence embeddings and stored those vectors in ChromaDB for nearest-neighbour search. Two models were considered in the project materials: a lighter MiniLM-based setup and a larger multilingual option. This reflected an important trade-off in the project. We were not only interested in retrieval quality, but also in practicality. A sanctions search system has to be able to scale to a large collection, so embedding choice affects memory, storage, batch size, and runtime as much as it affects ranking quality.

The implementation materials also show that this stage was treated seriously as an engineering problem, not just a proof-of-concept notebook. There is attention to caching embeddings, managing batch sizes, naming runs consistently, and persisting ChromaDB collections on disk. That matters because dense retrieval becomes genuinely useful only when it can be run repeatedly and compared systematically, not when it works once in an ad hoc notebook cell.

Once the embeddings and Chroma collections existed, the downstream dense search notebook could query them directly, compare models, and prepare outputs for pooling and later evaluation. At that point, the project had moved from a classical-only system to a genuine hybrid retrieval setup.

---

## 9. Why pooling was necessary for the harder query types

Not every query type has a clean, automatic answer key. This became especially clear for type 3 and type 4 queries. If a query is descriptive or relational, relevance cannot always be determined by a single exact field match. In those cases, the project needed a way to build judgement data without pretending that relevance was trivial.

That is why pooling became part of the workflow. Instead of declaring one system's outputs to be the ground truth, we used retrieval outputs from multiple systems to assemble a candidate pool for human judgement. This follows standard IR practice and fits the needs of the project well.

The pooling notebook for query types 3 and 4 reflects this logic directly. It runs lexical systems, merges their top-ranked candidates, enriches the exported output with useful context such as captions and text previews, and produces spreadsheet files for manual relevance assessment. That workflow is important because it turns difficult query types from an abstract problem into something operationally evaluable.

In narrative terms, pooling marks the point where the project stopped being only about building retrieval models and became a proper IR experiment. Once we started creating judged pools, we were no longer just asking whether a system looked plausible. We were preparing the evidence needed to compare systems more rigorously.

---

## 10. Combining retrieval lanes instead of forcing one model to do everything

Once both lexical and dense retrieval were present, fusion became the next logical step. The project used Reciprocal Rank Fusion as the mechanism for combining ranked outputs from different systems.

This was a sensible choice for two reasons. First, the lexical and dense lanes produce scores on very different scales, so naive score combination would be difficult to justify. Second, RRF works at the rank level rather than the raw score level, which makes it easier to combine heterogeneous systems without heavy calibration.

Conceptually, fusion expresses one of the central lessons of the whole project: different query behaviors benefit from different signals. Exact identifier matching, lexical overlap, and semantic similarity each capture something real, but none of them is sufficient on its own across the full query space. RRF gave us a practical way to combine these complementary strengths without pretending that one retrieval family had solved the whole problem.

This is also a strong presentation point later on, because it shows the architecture maturing. The project did not move from one model to another in a straight line. It moved from understanding the task, to building separate retrieval capabilities, to combining them in a principled way.

---

## 11. Building the ground truth because it did not exist already

One of the most important evaluation realities in this project is that there was no ready-made ground-truth benchmark waiting for us. OpenSanctions gives us a large and operationally interesting corpus, but it does not give us a complete evaluation set for our specific query types. That meant we had to build the judgement layer ourselves, at least for query types 1 to 6.

This is an important point to explain in the presentation because it shows that evaluation was not just a matter of plugging ranked lists into a metric library. Before we could calculate anything meaningful, we needed a way to define which entities counted as relevant for each type of query. The answer was not the same for every query type.

For query types 1, 2, 5, and 6, ground truth could be built in a more automatic or semi-automatic way because those query types are anchored in fields or metadata that can be checked directly.

- Type 1 ground truth is the most straightforward. If the query is an exact identifier such as an IMO number, then relevant entities are those whose identifier field matches that value exactly.
- Type 2 ground truth is less rigid, but still structured. The relevant entity can often be found by matching the query against `name`, `alias`, or `previousName`, possibly with controlled variation.
- Type 5 ground truth depends on identifying records that correspond to the same real-world entity across multiple source datasets.
- Type 6 ground truth depends on structured constraints such as programme ID, jurisdiction, schema, or country, so relevant entities can often be derived from metadata filters.

These query types are not completely trivial, but they are at least grounded in observable fields. That makes automatic or semi-automatic qrels realistic.

The situation is different for query types 3 and 4. For those, relevance cannot be derived reliably from a single field lookup.

- Type 3 queries are descriptive. A user may describe a target in natural language without naming it directly, so relevance depends on whether a retrieved entity actually matches the described situation.
- Type 4 queries are relational. The query is often asking about links, ownership relations, or sanctions-network structure, so relevance depends on how well the retrieved entity fits a graph-style information need.

For these harder query types, we needed a pooling workflow rather than an automatic answer key. The basic logic is classic IR practice: run multiple retrieval systems independently, collect their top candidates, merge them into a judging pool, and then assign manual relevance labels. In our project, this pooling stage is the bridge between retrieval experiments and usable evaluation data. Without it, claims about type 3 and type 4 performance would be much weaker.

In presentation terms, the easiest way to explain pooling is this: we cannot manually judge the whole OpenSanctions collection for every semantic or relational query, so we ask several systems to propose likely candidates first. We then take, for example, the top results from BM25, TF-IDF, and dense retrieval, merge and deduplicate them, and treat that smaller combined set as the pool for human judgement. A reviewer then labels the pooled entities using a relevance scale, which becomes the qrels for those query types. Pooling therefore reduces judging cost while still giving the evaluation a fairer basis than relying on only one retrieval system or pretending that no judgement is needed.

So the ground-truth story in this project is really a mixed strategy:

- Types 1, 2, 5, and 6: automatic or semi-automatic qrels from known fields and metadata
- Types 3 and 4: pooled candidates with manual relevance judgement
- Type 7: not ordinary qrels, because the final output is generated narrative rather than only ranked retrieval

This is worth emphasising because it reflects the broader design philosophy of the project. Different query types do not just require different retrieval models. They also require different judgement strategies.

---

## 12. What the current evaluation work means, and why this section is only partial for now

Evaluation is already structurally present in the project, but this document intentionally treats it as an in-progress section rather than a final results chapter.

The project now has the pieces needed for evaluation:

- query files organised by type
- qrels infrastructure and utilities
- notebooks for pooling and judged relevance preparation
- metric definitions for types 1 to 6 in code, and project-level metric policy for all seven types
- evaluation notebooks for comparing BM25, fused runs, and optional dense runs

Even so, this is not the point to write a final narrative about results, because the user-facing evaluation story should be updated after the remaining evaluation work is completed. At the moment, the correct narrative is that we have built the evaluation framework and prepared the data flow for it, but we are deliberately postponing the final interpretation of results.

The most useful way to explain evaluation in the presentation is by connecting each query type to the metric that best matches its retrieval goal:

- Type 1 and Type 2: `Precision@1` and `MRR`, with `MAP` as a secondary measure. These query types are about whether the correct entity appears immediately and consistently near the top.
- Type 3 and Type 4: `nDCG@10` as the primary measure, with `MAP` and `Recall@10` as secondary measures. These query types often involve graded relevance and multiple plausible results, so ranking quality within the top results matters more than a single exact hit.
- Type 5 and Type 6: `Recall@10` and `Recall@20` as the primary measures, with `MAP` as a secondary measure. These types emphasise coverage, because the user may need several relevant entities rather than just one.
- Type 7: `RAGAS faithfulness` and `answer relevance`. This query type is not evaluated like ordinary ranked retrieval because the output is a generated narrative rather than only a ranked list.

At the system level, `Recall@10` is still an important overall measure because the project is motivated by risk minimisation. In this domain, failing to retrieve a relevant entity can matter more than achieving a slightly cleaner top rank on easy queries.

That is actually a useful presentation choice. Rather than presenting incomplete results as if they were final, this document can frame evaluation as the next stage of the project. The metrics policy is already clear: different query types require different measures, and overall performance will be judged in a way that reflects the retrieval needs of the domain. But the final comparative claims should come later, once the runs and qrels are fully settled.

For now, the important point is that evaluation was not treated as an afterthought. It was designed alongside the retrieval architecture, especially through per-type query handling, pooling for the more ambiguous tasks, and a metric policy that changes with the type of information need.

---

## 13. What we have really built so far

Looking back across the work completed so far, the project can be described as a progression through five connected layers.

First, we understood the domain and the dataset. We established that sanctions retrieval is a high-recall, high-stakes problem and that OpenSanctions is large, structured, and heterogeneous enough to justify a serious IR architecture.

Second, we built a preprocessing pipeline that turns raw nested records into retrieval-ready documents. This gave the project a stable internal representation and created a clean separation between names, free text, identifiers, and metadata.

Third, we defined the query space in a structured way. That meant the system was not built against a vague idea of "search," but against a set of concrete retrieval behaviors.

Fourth, we implemented multiple retrieval lanes rather than relying on a single model family. The classical lane supports exact and lexical matching. The dense lane addresses semantic similarity and broader descriptive search.

Fifth, we connected those lanes to experimental workflow through pooling, qrels preparation, and fusion. That transformed the repository from a collection of isolated notebooks and scripts into a coherent IR project pipeline.

This is the key narrative point: the project did not grow as a random accumulation of components. It progressed from understanding the retrieval problem, to building a representation, to implementing retrieval, to preparing systematic comparison.

---

## 14. Main design ideas that shaped the project

Several decisions appear repeatedly across the code, notebooks, and documentation, and together they define the character of the project.

The first is that document representation matters as much as ranking. A large part of the work went into deciding how to flatten records, which fields to keep raw, and which forms of text should be normalised or lemmatised.

The second is that query diversity should shape system architecture. We did not assume all users ask the same kind of question, so we did not build a one-lane system.

The third is that exact identifiers are special. In many IR settings, everything is treated as text. In this project, identifiers were treated as a separate retrieval channel because that better reflects real user needs.

The fourth is that hybrid retrieval is more realistic than retrieval monoculture. Lexical and dense methods each solve different parts of the problem, and the project architecture reflects that rather than trying to force one method to dominate all scenarios.

The fifth is that evaluation needs judgement, especially for harder query types. For type 3 and type 4 queries, relevance cannot simply be inferred from metadata. Pooling was therefore necessary, not optional.

---

## 15. Limitations and honest current boundaries

It is also important that the project narrative stays honest about what has and has not been completed yet.

The preprocessing and retrieval foundations are already substantial, but the final evaluation story is not finished. Some parts of the system are implemented in polished source modules, while others still live primarily in notebooks. That is normal for a project at this stage, but it should be acknowledged in any presentation. It shows that the work has moved beyond planning, but it has not yet reached its final polished experimental endpoint.

Another limitation is that the dataset itself introduces ambiguity and unevenness. Query formulation, relevance judgement, and cross-source matching are not always clean problems in sanctions data. In a way, that is a weakness of the dataset, but it is also part of the value of the project: it forced us to engage with realistic retrieval complications instead of a toy benchmark.

Type 7 is now included in the presentation narrative as part of the full task definition, but it should still be explained carefully as a different kind of output. It is not just "another retrieval query." It changes the end product from ranked records to grounded summarisation, and therefore it also changes the evaluation logic.

---

## 16. How this document should be used later for the presentation

When this is turned into a presentation, each major section can become one or two slides:

- problem and motivation
- dataset complexity
- preprocessing pipeline
- query design
- classical retrieval
- dense retrieval
- pooling and judgement
- fusion
- evaluation framework
- lessons learned and next steps

The advantage of writing the story this way is that the presentation does not need to start from scratch. Most of the explanation is already here in spoken form. Later, the main task will be compression: selecting the strongest diagrams, examples, tables, and retrieval outputs to accompany the narrative.

The evaluation section should be revisited once the results are complete. At that point, this document can be expanded with concrete metrics, comparisons, failure cases, and final conclusions for query types 1 to 7, while still making clear that type 7 uses a different evaluation family from the ranked-retrieval tasks.

---

## 17. One-sentence project summary

So far, the project has developed from a high-level search engine design into a working retrieval pipeline for sanctions entity search: we explored a complex real-world dataset, built a streaming preprocessing system, defined seven query behaviors, implemented classical and dense retrieval components, introduced fusion and pooling, and prepared the ground for full evaluation of both ranked retrieval and grounded summarisation tasks.
