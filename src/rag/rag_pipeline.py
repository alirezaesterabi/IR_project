"""RAG pipeline for narrative summarisation of sanctioned entities (query type 7)."""

from transformers import pipeline


class RAGPipeline:
    def __init__(self, k=3, max_tokens_per_doc=100, model_name="google/flan-t5-base"):
        self.k = k
        self.max_tokens_per_doc = max_tokens_per_doc
        self.pipeline = pipeline("text2text-generation", model=model_name)

    def build_context(self, documents: list[dict]) -> str:
        blocks = []
        for doc in documents:
            metadata = doc.get("metadata", {}) or {}
            text_blob = doc.get("text_blob", "") or ""
            tokens = text_blob.split()

            name = metadata.get("name")
            if not name:
                name = tokens[0] if tokens else ""

            lines = []
            if name:
                lines.append(f"Entity: {name}")

            schema = metadata.get("schema")
            if schema:
                lines.append(f"Type: {schema}")

            country = metadata.get("country")
            if country:
                lines.append(f"Country: {country}")

            program_id = metadata.get("programId")
            if program_id:
                lines.append(f"Programme: {program_id}")

            sanctions = doc.get("sanctions")
            if sanctions:
                parts = []
                for s in sanctions:
                    authority = s.get("authority", "")
                    date = s.get("date", "")
                    parts.append(f"{authority} ({date})")
                lines.append("Sanctions: " + ", ".join(parts))

            ownership = doc.get("ownership")
            if ownership:
                owner = ownership.get("owner", "")
                owner_sanctioned = ownership.get("ownerSanctioned", False)
                sanctioned_str = "true" if owner_sanctioned else "false"
                lines.append(f"Ownership: {owner}, sanctioned: {sanctioned_str}")

            if tokens:
                summary_tokens = tokens[: self.max_tokens_per_doc]
                lines.append("Summary: " + " ".join(summary_tokens))

            blocks.append("\n".join(lines))

        return "\n---\n".join(blocks)

    def build_prompt(self, query: str, context: str) -> str:
        return (
            "Summarise the following sanctioned entities in response to the query.\n"
            "Only use information from the provided context. Do not add external facts.\n"
            "\n"
            f"Query: {query}\n"
            "\n"
            "Context:\n"
            f"{context}\n"
            "\n"
            "Summary:"
        )

    def generate(self, query: str, documents: list[dict]) -> dict:
        truncated_docs = documents[: self.k]
        context = self.build_context(truncated_docs)
        prompt = self.build_prompt(query, context)
        output = self.pipeline(prompt, max_new_tokens=200, truncation=True)
        summary = output[0]["generated_text"]
        return {
            "summary": summary,
            "context": context,
            "query": query,
        }
