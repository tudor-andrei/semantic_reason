import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer, util


class CLIPSPatternMapper:
    """Maps natural language sentences to CLIPS facts and displays token embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L12-v2", max_span_len: int = 3):
        """
        Initialize the mapper with an embedding model.

        :param model_name: Name of the SentenceTransformer model
        :param max_span_len: Maximum number of words in candidate spans
        """
        self.model = SentenceTransformer(model_name)
        self.max_span_len = max_span_len

    @staticmethod
    def parse_pattern(pattern: str) -> Tuple[List[str], List[str]]:
        """
        Parse a CLIPS pattern into static tokens and variables.

        :param pattern: CLIPS pattern string
        :return: tuple (static_tokens, variables)
        """
        tokens = pattern.replace("(", "").replace(")", "").split()
        variables = [t for t in tokens if t.startswith("?") or t.startswith("$?")]
        static_tokens = [
            t for t in tokens if not (t.startswith("?") or t.startswith("$?"))
        ]
        return static_tokens, variables

    def generate_spans(self, sentence_tokens: List[str]) -> List[str]:
        """
        Generate candidate spans from sentence tokens up to max_span_len.

        :param sentence_tokens: List of words in the sentence
        :return: List of candidate spans
        """
        spans = [
            " ".join(sentence_tokens[i : i + l])
            for l in range(1, self.max_span_len + 1)
            for i in range(len(sentence_tokens) - l + 1)
        ]
        return spans

    def display_embeddings(self, text: str, label: str, max_dims: int = 8):
        """
        Display token embeddings for a given text.

        :param text: Input text
        :param label: Label for display
        :param max_dims: Number of embedding dimensions to display
        """
        tokens = text.split()
        token_embs = self.model.encode(tokens)
        df = pd.DataFrame(
            {
                **{"Token": tokens},
                **{
                    f"Dim_{i}": token_embs[:, i]
                    for i in range(min(max_dims, token_embs.shape[1]))
                },
            }
        )
        print(f"\nEmbeddings for {label}:")
        print(df)

    def plot_span_candidates(
        self,
        variable: str,
        candidate_spans: List[str],
        scores: List[float],
        best_span: str,
        save_path: str = None,
    ):
        """
        Plot a ranked bar chart of candidate spans and their cosine similarity
        scores for a given CLIPS variable, highlighting the span that was
        ultimately selected during variable assignment.

        :param variable: The CLIPS variable being assigned (e.g. '?nume-complet')
        :param candidate_spans: List of candidate spans generated from the sentence
        :param scores: List of cosine similarity scores, aligned with candidate_spans
        :param best_span: The span that was selected as the best match
        :param save_path: If given, save the figure to this path instead of showing it
        """
        # Rank candidates from highest to lowest similarity
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        labels = [candidate_spans[i] for i in order]
        ranked_scores = [scores[i] for i in order]

        # Highlight the winning span in a different color
        colors = ["#e69f00" if label == best_span else "#3b528b" for label in labels]

        plt.figure(figsize=(max(6, len(labels) * 0.6), 4))
        bars = plt.bar(range(len(labels)), ranked_scores, color=colors)
        plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
        plt.ylabel("Cosine Similarity")
        plt.ylim(0, 1)
        plt.title(f"Candidate Spans for {variable} (winner in orange)")

        for bar, score in zip(bars, ranked_scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{score:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"\n✅ Span-candidate plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def map_sentence_to_best_pattern(
        self, sentence: str, patterns: List[str]
    ) -> Tuple[str, float]:
        """
        Map a sentence to the best matching CLIPS pattern and assign variables.

        :param sentence: Input natural language sentence
        :param patterns: List of CLIPS patterns
        :return: Tuple(mapped_fact, similarity_score)
        """
        sentence_tokens = sentence.split()
        sentence_embs = self.model.encode(sentence, convert_to_tensor=True)

        # Display sentence token embeddings
        self.display_embeddings(sentence, "Sentence")

        best_fact, best_score = None, -1.0

        for pattern in patterns:
            static_tokens, variables = self.parse_pattern(pattern)
            static_phrase = " ".join(static_tokens)
            static_emb = self.model.encode(static_phrase, convert_to_tensor=True)

            # Display pattern token embeddings
            self.display_embeddings(static_phrase, f"Pattern '{pattern}'")

            # Compute similarity between sentence and static part of pattern
            similarity = util.cos_sim(sentence_embs, static_emb).item()

            if similarity > best_score:
                # Generate candidate spans from sentence
                candidate_spans = self.generate_spans(sentence_tokens)

                assigned_fact = pattern

                # --- General Variable Assignment ---
                for var in variables:
                    best_span, best_score = None, -1.0
                    span_scores = []  # NEW: track every candidate's score for plotting

                    for span in candidate_spans:
                        temp_fact = assigned_fact.replace(var, span, 1)
                        temp_emb = self.model.encode(temp_fact, convert_to_tensor=True)
                        score = util.cos_sim(sentence_embs, temp_emb).item()
                        span_scores.append(score)  # NEW
                        if score > best_score:
                            best_score = score
                            best_span = span

                    assigned_fact = assigned_fact.replace(var, best_span, 1)

                    # Visualize how the winning span was chosen among all candidates
                    self.plot_span_candidates(var, candidate_spans, span_scores, best_span)

                best_fact, best_score = assigned_fact, similarity

        return best_fact, best_score


# -------------------- Support Functions -------------------- #


def user_login():
    """Loads environment variables and logs in to Hugging Face automatically."""
    # Load environment variables from .env file
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        print("🔐 Found HF_TOKEN in environment. Logging in automatically...")
        # Pass the token explicitly to bypass the interactive terminal prompt
        login(token=hf_token)
    else:
        # Fallback method if someone forgets to create or populate their .env file
        print(
            "⚠️ HF_TOKEN not found in .env. Falling back to manual entry."
            "\n🔐 Enter your Hugging Face access token "
            "(get one at https://huggingface.co/settings/tokens):"
        )
        token = input("Access Token: ").strip()
        login(token=token)


def extract_patterns_from_clp(filepath: str) -> List[str]:
    """
    Extracts CLIPS patterns that are individually commented with ';;;;'.
    Example pattern line format:
        ;;;; (pattern ...)
    """
    patterns = []

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()

            # Accept only lines that both start with ;;;; and contain (...).
            if stripped.startswith(";;;;") and "(" in stripped and ")" in stripped:
                # Remove leading comment markers and whitespace
                pattern = stripped.lstrip(";/ ").strip()
                patterns.append(pattern)

    return patterns


def append_facts_to_clp(
    clp_file: str, facts: List[str], facts_name: str = "fapte-test"
):
    """
    Appends mapped CLIPS facts to a deffacts block at the end of the file.

    :param clp_file: Path to the .clp file
    :param facts: List of CLIPS fact strings
    :param facts_name: Name of the deffacts block (default "fapte-test")
    """
    with open(clp_file, "a", encoding="utf-8") as f:
        f.write(f"\n(deffacts {facts_name}\n")
        for fact in facts:
            f.write(f"   {fact}\n")
        f.write(")\n")
    print(f"\n✅ Facts appended to {clp_file}")


# -------------------- Example Usage -------------------- #

if __name__ == "__main__":
    user_login()

    clp_file = input("\n📄 Enter path to .clp file: ").strip()
    patterns = extract_patterns_from_clp(clp_file)

    print("\n✅ Loaded Patterns:")
    for p in patterns:
        print("  ", p)

    print("\n✍ Enter natural language sentences (empty line to stop):")
    sentences = []
    while True:
        s = input("> ").strip()
        if not s:
            break
        sentences.append(s)

    mapper = CLIPSPatternMapper()
    mapped_facts = []

    for s in sentences:
        fact, score = mapper.map_sentence_to_best_pattern(s, patterns)
        mapped_facts.append(fact)
        print(f"\nSentence: {s}")
        print(f"→ Mapped Fact: {fact}")
        print(f"→ Similarity: {score:.4f}")

    append_facts_to_clp(clp_file, mapped_facts)


# # ---------------- Example Usage ----------------
#
# if __name__ == "__main__":
#     patterns = [
#         "(este un-actor-grozav $?nume-complet)",
#         "(joaca-la un-actor-grozav $?productie)"
#         # "(is a-great-actor ?name)",
#         # "(a-great-actor plays-in ?production)"
#     ]
#     sentence = "Un actor grozav joaca la Studioul Hollywood"  # Florin Piersic este un actor exceptional, Andrei is a great actor, To play in Hollywood you have to be a great actor
#
#     mapper = CLIPSPatternMapper()
#     fact, score = mapper.map_sentence_to_best_pattern(sentence, patterns)
#
#     print(f"\nSentence: {sentence}")
#     print(f"Mapped Fact: {fact}")
#     print(f"Similarity Score: {score:.4f}")
