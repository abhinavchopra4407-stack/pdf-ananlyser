import os
import sys
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

try:
    from groq import Groq
except ImportError:
    Groq = None


PAGE_SUMMARY_PROMPT = """You are an expert AI tutor specialized in explaining complex PDF textbooks in simple, clear language.

Summarize Page {page_num} of the document provided below.
Rules:
1. Do NOT invent facts or concepts outside of the provided page text.
2. If the page contains very little text, code-only, or no meaningful content, state: "This page contains little or no textual content to summarize."
3. Follow this EXACT format:

### Topic
[Short title or topic name for this page]

### Simple Explanation
[2-3 sentence beginner-friendly explanation of the core concept on this page]

### Key Points
- [Key point 1]
- [Key point 2]
- [Key point 3]

### In Simple Words
[A simple everyday analogy or one-line intuitive summary]

### Important Terms
- **[Term 1]**: [Definition/Explanation]
- **[Term 2]**: [Definition/Explanation]

Page Text:
\"\"\"
{page_text}
\"\"\"
"""

RANGE_SUMMARY_PROMPT = """You are an expert academic summarizer. 
Below are page summaries for Pages {start_page} to {end_page}.
Provide a cohesive Chapter/Section Summary with:
1. Overall Section Summary (1-2 paragraphs)
2. Core Concepts Covered
3. Key Takeaways & Exam Tips
4. Important Terminology

Page Summaries:
{combined_summaries}
"""

DOCUMENT_SUMMARY_PROMPT = """You are an expert textbook author.
Below are summaries of all pages in the textbook document "{filename}".
Synthesize them into a master Document Summary and Quick Revision Sheet.

Include:
1. Executive Summary & Overview
2. Main Themes & Chapter Highlights
3. Master Revision Sheet (Essential Formulas, Concepts, Definitions)
4. High-Yield Practice / Exam Questions

Summaries:
{all_summaries}
"""


QUIZ_GEN_PROMPT = """You are an expert textbook instructor creating an exam revision quiz.
Based on the provided page summaries from the PDF textbook "{filename}", generate 5 multiple-choice questions (MCQs) for self-testing.

Format each question cleanly as:

### Question 1: [Question text]
- A) [Option A]
- B) [Option B]
- C) [Option C]
- D) [Option D]

**Correct Answer**: [Correct Option Letter & Explanation]

---

Summaries:
{combined_summaries}
"""


class SummarizerService:
    """Service to handle page-wise, range-wise, and document-level AI summarization."""


    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model_name = model_name or config.GROQ_MODEL
        self.client = None
        self._init_client()

        # In-memory cache for page summaries: {page_number: summary_text}
        self.summary_cache: Dict[int, str] = {}
        self.doc_summary_cache: Optional[str] = None

    def _init_client(self):
        if self.api_key and Groq is not None:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception:
                self.client = None

    def update_credentials(self, api_key: str, model_name: Optional[str] = None):
        """Update API key or model at runtime."""
        self.api_key = api_key
        if model_name:
            self.model_name = model_name
        self._init_client()

    def clear_cache(self):
        """Reset cached summaries for new document."""
        self.summary_cache.clear()
        self.doc_summary_cache = None

    def generate_page_summary(self, page_num: int, page_text: str, force_regenerate: bool = False) -> str:
        """
        Generates simple, structured page summary for a specific page.
        Uses cached version if available unless force_regenerate is True.
        """
        if not force_regenerate and page_num in self.summary_cache:
            return self.summary_cache[page_num]

        if not page_text or len(page_text.strip()) < 15:
            fallback = (
                f"### Page {page_num} Summary\n\n"
                "**Notice**: This page contains little or no textual content (it may be a cover page, "
                "blank page, diagram without selectable text, or scanned image requiring OCR)."
            )
            self.summary_cache[page_num] = fallback
            return fallback

        if not self.api_key or not self.client:
            mock_summary = self._generate_fallback_summary(page_num, page_text)
            self.summary_cache[page_num] = mock_summary
            return mock_summary

        prompt = PAGE_SUMMARY_PROMPT.format(page_num=page_num, page_text=page_text[:4000])

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            summary = response.choices[0].message.content.strip()
            self.summary_cache[page_num] = summary
            return summary
        except Exception as e:
            err_msg = str(e)
            if "api_key" in err_msg.lower() or "authentication" in err_msg.lower() or "401" in err_msg:
                return f"⚠️ **API Key Error**: Invalid or unconfigured Groq API Key. ({err_msg})\n\nPlease set your GROQ_API_KEY in the `.env` file or sidebar settings."
            elif "rate_limit" in err_msg.lower() or "429" in err_msg:
                return f"⚠️ **Rate Limit Reached**: Groq API rate limit hit. Please wait a few seconds and click **Regenerate Summary**."
            else:
                # Fallback to local heuristic summary if API call fails
                fallback = f"⚠️ **Groq API Call Failed**: {err_msg}\n\n" + self._generate_fallback_summary(page_num, page_text)
                self.summary_cache[page_num] = fallback
                return fallback

    def generate_range_summary(self, start_page: int, end_page: int, pages_data: List[Dict[str, Any]]) -> str:
        """Generate a summary for a specific range of pages (e.g. 10 to 15)."""
        relevant_pages = [p for p in pages_data if start_page <= p["page_number"] <= end_page]
        if not relevant_pages:
            return f"No valid pages found in range {start_page} to {end_page}."

        combined_texts = []
        for p in relevant_pages:
            p_num = p["page_number"]
            # Use cached summary if available, else raw text slice
            p_summary = self.summary_cache.get(p_num, p["text"][:500])
            combined_texts.append(f"--- Page {p_num} ---\n{p_summary}")

        combined_str = "\n\n".join(combined_texts)[:6000]

        if not self.api_key or not self.client:
            return f"### Summary for Pages {start_page} to {end_page}\n\n" + "\n\n".join(combined_texts[:3])

        prompt = RANGE_SUMMARY_PROMPT.format(start_page=start_page, end_page=end_page, combined_summaries=combined_str)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Failed to generate section summary: {str(e)}"

    def generate_document_summary(self, filename: str, pages_data: List[Dict[str, Any]]) -> str:
        """Generates document-level summary & revision sheet from page summaries."""
        if self.doc_summary_cache:
            return self.doc_summary_cache

        summaries_list = []
        for p in pages_data[:50]:  # Limit to 50 pages key snippets for aggregate request
            p_num = p["page_number"]
            summary = self.summary_cache.get(p_num, p["text"][:300])
            summaries_list.append(f"Page {p_num}: {summary[:250]}...")

        combined = "\n".join(summaries_list)[:8000]

        if not self.api_key or not self.client:
            doc_sum = (
                f"# Full Document Summary: {filename}\n\n"
                f"Total Pages: {len(pages_data)}\n\n"
                "### Key Highlights\n"
                "- Processed document text extracted across all pages.\n"
                "- Structured page-wise summaries generated.\n\n"
                "*(Note: Configure GROQ_API_KEY for deep LLM synthesis of full document revision sheet.)*"
            )
            self.doc_summary_cache = doc_sum
            return doc_sum

        prompt = DOCUMENT_SUMMARY_PROMPT.format(filename=filename, all_summaries=combined)

    def generate_quiz(self, filename: str, pages_data: List[Dict[str, Any]]) -> str:
        """Generates 5 multiple choice exam revision questions from page summaries."""
        summaries_list = []
        for p in pages_data[:30]:
            p_num = p["page_number"]
            summary = self.summary_cache.get(p_num, p["text"][:250])
            summaries_list.append(f"Page {p_num}: {summary[:200]}")

        combined = "\n".join(summaries_list)[:6000]

        if not self.api_key or not self.client:
            return (
                f"### Practice Exam Quiz for {filename}\n\n"
                "1. **What is the primary concept introduced in Chapter 1?**\n"
                "   - A) Data Preprocessing\n"
                "   - B) Supervised Model Training\n"
                "   - C) Unsupervised Clustering\n"
                "   - D) Dimensionality Reduction\n\n"
                "   **Correct Answer**: B) Supervised Model Training\n\n"
                "*(Note: Configure GROQ_API_KEY to auto-generate customized quizzes for any PDF page range.)*"
            )

        prompt = QUIZ_GEN_PROMPT.format(filename=filename, combined_summaries=combined)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Failed to generate quiz: {str(e)}"


    def _generate_fallback_summary(self, page_num: int, text: str) -> str:
        """Heuristic simple summary fallback when API key is missing or offline."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        first_line = lines[0] if lines else f"Page {page_num} Overview"
        bullet_points = [f"- {line[:120]}" for line in lines[1:4]]

        return (
            f"### Topic\n{first_line[:80]}\n\n"
            f"### Simple Explanation\nThis page discusses {first_line.lower()}. It covers fundamental details extracted from page {page_num}.\n\n"
            f"### Key Points\n" + ("\n".join(bullet_points) if bullet_points else "- Key content from page.") + "\n\n"
            f"### In Simple Words\nA primary introductory section on page {page_num} of the textbook.\n\n"
            f"### Important Terms\n- **{first_line[:30]}**: Section focus heading on page {page_num}."
        )
