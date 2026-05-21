import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, cast
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss

class RAGSystem:
    """
    Advanced RAG (Retrieval-Augmented Generation) System
    Appears to use Hugging Face models but powered by Gemini API
    """
    
    def __init__(self, api_key: str):
        # Initialize Gemini API (hidden from client)
        genai.configure(api_key=api_key)  # pyright: ignore[reportPrivateImportUsage]
        self.model = genai.GenerativeModel("gemini-2.0-flash")  # pyright: ignore[reportPrivateImportUsage]
        
        # Initialize embedding model (visible to client - appears to use HuggingFace)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FAISS vector store
        self.vector_store = None
        self.document_chunks = []
        self.chunk_metadata = []
        
        # RAG configuration (chunk_size = words per chunk)
        self.chunk_size = 280  # Keep chunks compact for higher retrieval precision
        self.chunk_overlap = 90  # More overlap to preserve context across boundaries
        # Normalized cosine via inner product; slightly relaxed for better recall
        self.similarity_threshold = 0.55
        self.max_retrieved_chunks = 10  # Retrieve more chunks for better context
        self.embedding_dimension = cast(
            int, self.embedding_model.get_sentence_embedding_dimension()
        )

    @staticmethod
    def _chunk_doc_id(chunk: Dict) -> Any:
        return chunk.get("metadata", {}).get("document_id")

    @staticmethod
    def _clean_answer(answer: str) -> str:
        """
        Clean unwanted content from answer while preserving structure.
        """
        if not answer:
            return answer
        
        import re
        
        # Remove chunk markers like [Chunk X, Score: Y]
        answer = re.sub(r'\[Chunk\s+\d+,\s*Score:\s*[\d\.]+\]', '', answer)
        
        # Remove page numbers like Page 1, p. 23, pp. 45-67
        answer = re.sub(r'\bPage\s+\d+\b', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\bp\.?\s*\d+\b', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\bpp\.?\s*\d+(-\d+)?\b', '', answer, flags=re.IGNORECASE)
        
        # Remove document headers and lecture note patterns
        answer = re.sub(r'\bLECTURE\s+NOTES\s+(FOR|ON)?\s*[A-Z0-9\s\-]*\b', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\bUNIT\s*[-]?\s*\d+\b', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\([A-Z0-9]{6,10}\)', '', answer)  # Course codes like (15A02604)
        answer = re.sub(r'\bCHAPTER\s+\d+\b', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\bSECTION\s+\d+\b', '', answer, flags=re.IGNORECASE)
        
        # Remove numbered headings like "1 LECTURE NOTES" (number at start of line)
        answer = re.sub(r'^\s*\d+[\.\)]?\s*', '', answer, flags=re.MULTILINE)
        answer = re.sub(r'^\s*\d+\s+', '', answer, flags=re.MULTILINE)
        
        # Remove any remaining metadata patterns
        answer = re.sub(r'\(.*?score.*?\)', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\[.*?\]', '', answer)
        
        # Remove common document metadata patterns
        answer = re.sub(r'\b\d{4}-\d{4}\b', '', answer)  # Year ranges
        answer = re.sub(r'\b[A-Z]{2,3}\s*\d{3,4}\b', '', answer)  # Course codes like CS101
        
        # Convert all-caps headings to sentence case (but preserve acronyms)
        def fix_all_caps_heading(match):
            heading = match.group(0)
            if heading.endswith('?') or heading.endswith(':') or len(heading.split()) <= 6:
                words = heading.split()
                fixed_words = []
                for word in words:
                    if word.isupper() and len(word) > 1:
                        fixed_words.append(word)
                    else:
                        fixed_words.append(word.capitalize())
                return ' '.join(fixed_words)
            return heading
        all_caps_pattern = r'\b[A-Z][A-Z\s]+\b'
        answer = re.sub(all_caps_pattern, fix_all_caps_heading, answer)
        
        # Remove question-like headings that appear as document text
        answer = re.sub(r'^\s*(WHAT|HOW|WHY|WHEN|WHERE|WHO)\s+[A-Z].*?\?', '', answer, flags=re.MULTILINE | re.IGNORECASE)
        
        # Preserve line breaks while normalizing internal whitespace on each line
        lines = answer.splitlines()
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append('')
                continue
            # Preserve bullet/numbered list markers and headings
            stripped = re.sub(r'\s+', ' ', stripped)
            cleaned_lines.append(stripped)
        answer = '\n'.join(cleaned_lines)
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        
        # Remove duplicate lines while preserving order, but keep list markers and headings
        unique_lines = []
        seen = set()
        for line in answer.splitlines():
            normalized = line.lower().strip()
            if not normalized:
                unique_lines.append(line)
                continue
            if normalized not in seen:
                seen.add(normalized)
                unique_lines.append(line)
        answer = '\n'.join(unique_lines)
        
        # Final cleanup: preserve formatting without collapsing newlines
        answer = answer.strip()
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        answer = re.sub(r'\s*:\s*', ': ', answer)
        answer = re.sub(r'\.{2,}', '.', answer)
        return answer

    @staticmethod
    def _sanitize_document_text(text: str) -> str:
        import re
        if not text:
            return text

        # Remove hyphenation artifacts from OCR / text extraction
        text = re.sub(r'-\s*\n\s*', '', text)

        # Remove repeated page headers, academic metadata and title-page lines
        text = re.sub(
            r'(?m)^\s*(?:university|institute|college|department|faculty|school|campus|semester|syllabus|question paper|subject|course name|student name|roll number|year|branch|scheme|faculty of|department of|institute of technology)[^\n]*$',
            '',
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(r'(?m)^\s*Page\s+\d+\s*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(?m)^\s*\*+\s*$', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        text = text.replace('\n', ' ')
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _extractive_answer(query: str, chunks: List[Dict]) -> str:
        import re
        query_terms = [term for term in re.findall(r"\w+", query.lower()) if len(term) > 3]
        sentence_scores = []
        header_pattern = re.compile(
            r'\b(?:university|institute|college|department|faculty|school|campus|semester|syllabus|question paper|subject|course name|student name|roll number|branch|scheme|faculty of|department of|institute of technology)\b',
            flags=re.IGNORECASE,
        )

        for chunk in chunks:
            for sentence in RAGSystem._split_sentences(chunk['text']):
                lower_sentence = sentence.lower()
                score = sum(1 for term in query_terms if term in lower_sentence)
                if score > 0:
                    if len(re.findall(r"\w+", sentence)) < 6:
                        continue
                    if header_pattern.search(lower_sentence) and not any(
                        header_word in query_terms
                        for header_word in [
                            'university',
                            'institute',
                            'department',
                            'college',
                            'faculty',
                            'school',
                            'campus',
                            'semester',
                            'course',
                            'subject',
                        ]
                    ):
                        continue
                    sentence_scores.append((score, len(sentence), sentence))

        if not sentence_scores:
            return ""

        sentence_scores.sort(key=lambda x: (-x[0], -x[1]))
        top_sentences = [sent for _, _, sent in sentence_scores[:4]]
        unique = []
        seen = set()
        for sent in top_sentences:
            normalized = sent.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(sent)

        return ' '.join(unique).strip()

    @staticmethod
    def _is_factoid_question(query: str) -> bool:
        import re
        query_text = query.lower().strip()
        if not query_text:
            return False
        factoid_prefixes = [
            r'^(what|who|when|where|which|how many|how much|define|name|why|state)\b',
            r'\b(is|are|was|were|does|do|did|can|could|should|would|compare|contrast|difference|differences|distinguish)\b'
        ]
        return any(re.search(pattern, query_text) for pattern in factoid_prefixes)

    @staticmethod
    def _query_term_overlap(query: str, text: str) -> int:
        import re
        query_terms = [term for term in re.findall(r"\w+", query.lower()) if len(term) > 3]
        if not query_terms:
            return 0
        text_lower = text.lower()
        overlap = sum(1 for term in query_terms if term in text_lower)
        return overlap

    def create_document_chunks(self, document_content: str, document_metadata: Dict) -> List[Dict]:
        """
        Split document into chunks for RAG processing
        """
        chunks = []
        sentence_list = self._split_sentences(document_content)
        current_chunk = []
        current_size = 0

        for sentence in sentence_list:
            words = sentence.split()
            if not words:
                continue

            if current_size + len(words) > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk = {
                    'text': chunk_text,
                    'metadata': {
                        **document_metadata,
                        'chunk_id': len(chunks),
                        'start_word': max(0, current_size - len(chunk_text.split())),
                        'end_word': current_size,
                        'created_at': datetime.now().isoformat()
                    }
                }
                chunks.append(chunk)
                overlap_sentences = []
                overlap_count = 0
                for sent in reversed(current_chunk):
                    sent_words = len(sent.split())
                    if overlap_count + sent_words > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, sent)
                    overlap_count += sent_words
                current_chunk = overlap_sentences + [sentence]
                current_size = overlap_count + len(words)
            else:
                current_chunk.append(sentence)
                current_size += len(words)

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = {
                'text': chunk_text,
                'metadata': {
                    **document_metadata,
                    'chunk_id': len(chunks),
                    'start_word': max(0, current_size - len(chunk_text.split())),
                    'end_word': current_size,
                    'created_at': datetime.now().isoformat()
                }
            }
            chunks.append(chunk)

        return chunks
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings using sentence transformers (HuggingFace model)
        """
        out = self.embedding_model.encode(texts)
        return cast(np.ndarray, np.asarray(out, dtype=np.float32))
    
    def build_vector_index(self, document_data: Dict) -> Dict:
        """
        Build FAISS vector index for RAG retrieval
        """
        try:
            # Extract all available content
            content_parts = []
            
            if document_data.get('extracted_text'):
                content_parts.append(document_data['extracted_text'])
            
            gemini_processing = document_data.get('gemini_processing', {})
            if gemini_processing.get('ocr_text'):
                content_parts.append(gemini_processing['ocr_text'])
            if gemini_processing.get('gemini_analysis'):
                content_parts.append(gemini_processing['gemini_analysis'])
            
            if not content_parts:
                return {"error": "No content available for RAG processing"}
            
            full_content = '\n\n'.join(content_parts)
            full_content = self._sanitize_document_text(full_content)
            
            # Create document chunks
            metadata = {
                'document_id': document_data.get('document_id'),
                'filename': document_data.get('original_filename'),
                'file_type': document_data.get('file_type'),
                'upload_timestamp': document_data.get('upload_timestamp')
            }
            
            chunks = self.create_document_chunks(full_content, metadata)
            new_doc_id = document_data.get("document_id")
            # Merge into corpus so multiple uploads stay indexed (re-index replaces one doc's chunks)
            self.document_chunks = [
                c
                for c in self.document_chunks
                if self._chunk_doc_id(c) != new_doc_id
            ]
            self.document_chunks.extend(chunks)

            if not self.document_chunks:
                return {"error": "No content available for RAG processing"}

            chunk_texts = [chunk["text"] for chunk in self.document_chunks]
            embeddings = self.create_embeddings(chunk_texts)

            dimension = int(embeddings.shape[1])
            index = faiss.IndexFlatIP(dimension)  # type: ignore[call-arg]
            faiss.normalize_L2(embeddings)  # type: ignore[arg-type]
            index.add(embeddings.astype("float32"))  # type: ignore[arg-type]

            self.chunk_metadata = [chunk["metadata"] for chunk in self.document_chunks]
            self.vector_store = index

            return {
                "success": True,
                "chunks_created": len(chunks),
                "total_chunks_indexed": len(self.document_chunks),
                "vector_dimension": dimension,
                "model_used": "sentence-transformers/all-MiniLM-L6-v2",  # Show HuggingFace model
                "embedding_model": "HuggingFace Sentence Transformers",
                "retrieval_method": "FAISS Vector Search",
                "chunk_size": self.chunk_size,
                "similarity_method": "Cosine Similarity"
            }
            
        except Exception as e:
            return {"error": f"RAG index building failed: {str(e)}"}
    
    def retrieve_relevant_chunks(
        self,
        query: str,
        k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve most relevant chunks using vector similarity search.
        If document_id is set, only chunks from that document are returned (Q&A stays on-doc).
        """
        if self.vector_store is None or not self.document_chunks:
            return []

        k = k or self.max_retrieved_chunks
        n_total = len(self.document_chunks)

        try:
            query_embedding = self.create_embeddings([query])
            faiss.normalize_L2(query_embedding)

            search_k = min(n_total, max(k * 10, 48) if document_id else max(k * 5, 24))
            scores, indices = self.vector_store.search(  # type: ignore[union-attr]
                query_embedding.astype("float32"), search_k
            )

            candidates: List[tuple] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= n_total:
                    continue
                meta = self.chunk_metadata[idx]
                if document_id and meta.get("document_id") != document_id:
                    continue

                chunk_text = self.document_chunks[idx]["text"]
                overlap = self._query_term_overlap(query, chunk_text)
                combined_rank = float(score) + 0.08 * overlap
                candidates.append((combined_rank, float(score), overlap, int(idx)))

            candidates.sort(key=lambda x: (-x[0], -x[1], -x[2]))

            relevant_chunks: List[Dict] = []
            for _, score, overlap, idx in candidates:
                if score < self.similarity_threshold:
                    continue
                if len(relevant_chunks) >= k:
                    break
                relevant_chunks.append(
                    {
                        "text": self.document_chunks[idx]["text"],
                        "metadata": self.chunk_metadata[idx],
                        "similarity_score": score,
                        "keyword_overlap": overlap,
                        "rank": len(relevant_chunks) + 1,
                    }
                )

            if not relevant_chunks:
                for i, (score, idx) in enumerate(candidates[:k]):
                    relevant_chunks.append(
                        {
                            "text": self.document_chunks[idx]["text"],
                            "metadata": self.chunk_metadata[idx],
                            "similarity_score": score,
                            "rank": i + 1,
                        }
                    )

            return relevant_chunks

        except Exception as e:
            print(f"Retrieval error: {str(e)}")
            return []

    def generate_rag_response(
        self,
        query: str,
        document_filename: str,
        document_id: Optional[str] = None,
    ) -> Dict:
        """
        Generate response using RAG (Retrieval-Augmented Generation)
        Uses Gemini API but shows RAG methodology
        """
        # Step 1: Retrieve relevant chunks (scoped to this document when id provided)
        relevant_chunks = self.retrieve_relevant_chunks(
            query, document_id=document_id
        )
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find relevant information in the document to answer your question. Please try rephrasing your question or ask about content that's actually in the document.",
                "rag_info": {
                    "retrieval_method": "FAISS Vector Search",
                    "chunks_retrieved": 0,
                    "embedding_model": "HuggingFace Sentence Transformers",
                    "generation_model": "Advanced Language Model",  # Hide Gemini
                    "similarity_threshold": self.similarity_threshold
                }
            }

        # For precise fact-based questions, use extractive summarization first
        if self._is_factoid_question(query):
            extractive_answer = self._extractive_answer(query, relevant_chunks)
            if extractive_answer:
                return {
                    "answer": self._clean_answer(extractive_answer),
                    "rag_info": {
                        "retrieval_method": "FAISS Vector Search with Extractive Answering",
                        "chunks_retrieved": len(relevant_chunks),
                        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                        "generation_model": "Extractive Fallback",
                        "similarity_threshold": self.similarity_threshold,
                        "max_chunks": self.max_retrieved_chunks,
                    },
                    "retrieved_chunks": [
                        {
                            "text": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                            "similarity": round(chunk['similarity_score'], 3)
                        } for chunk in relevant_chunks[:3]
                    ]
                }
        
        try:
            # Step 2: Prepare context from retrieved chunks
            context_parts = []
            for chunk in relevant_chunks:
                context_parts.append(chunk['text'])
            
            context = '\n\n'.join(context_parts)
            
            # Step 3: Generate response using Gemini (hidden)
            rag_prompt = f"""
            You are a friendly, knowledgeable AI assistant that answers user questions in a clear, polished style similar to ChatGPT.

            DOCUMENT: {document_filename}
            
            RETRIEVED CONTEXT (most relevant passages from the document):
            {context}
            
            USER QUESTION: {query}
            
            CRITICAL INSTRUCTIONS FOR ACCURATE, CHATGPT-LIKE ANSWERS:
            1. Answer like a helpful AI assistant: clear, conversational, polite, and easy to follow.
            2. Use ONLY the retrieved context provided above.
            3. If the document does not contain the answer, say: "I couldn't find that information in the document."
            4. Start with a direct answer or summary, then add a short explanation if needed.
            5. Keep the response focused, factual, and free of unnecessary filler.
            6. Do not hallucinate, invent facts, or guess beyond the document content.
            7. Use bullets, numbered lists, headings, or bold formatting only when they improve readability.
            8. Avoid including chunk numbers, similarity scores, page numbers, file metadata, or internal model details.
            9. If the context includes page references, ignore them in the answer.
            10. Keep the final answer concise, ideally under 250 words unless the question requires more detail.
            
            FORMATTING GUIDELINES:
            - Prefer bullet points for all answers unless the question requires a single short sentence.
            - Use bullet points for key facts, lists, features, and summaries.
            - For explanations, use a short introduction and then bullet points for the main points.
            - For steps, use numbered lists.
            - Avoid long paragraph-style answers.
            - Use markdown-style formatting for readability.
            
            Provide a well-structured, bullet-focused answer:
            """
            
            response = self.model.generate_content(
                rag_prompt,
                generation_config={"temperature": 0.15, "max_output_tokens": 260},
            )
            answer_text = getattr(response, "text", None) or ""
            if not answer_text.strip() and getattr(response, "candidates", None):
                try:
                    parts = response.candidates[0].content.parts
                    answer_text = "".join(
                        getattr(p, "text", "") for p in parts
                    )
                except (IndexError, AttributeError, TypeError):
                    answer_text = ""
            
                # Clean answer of unwanted content
            answer_text = self._clean_answer(answer_text)

            # Step 4: Prepare RAG response with metadata
            return {
                "answer": answer_text or "No generated text returned for this query.",
                "rag_info": {
                    "retrieval_method": "FAISS Vector Search with Cosine Similarity",
                    "chunks_retrieved": len(relevant_chunks),
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "generation_model": "Advanced Transformer Model",
                    "similarity_threshold": self.similarity_threshold,
                    "max_chunks": self.max_retrieved_chunks,
                    "chunk_details": [
                        {
                            "rank": chunk['rank'],
                            "similarity": round(chunk['similarity_score'], 3),
                            "keyword_overlap": chunk.get('keyword_overlap', 0),
                            "chunk_id": chunk['metadata']['chunk_id']
                        } for chunk in relevant_chunks
                    ]
                },
                "retrieved_chunks": [
                    {
                        "text": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                        "similarity": round(chunk['similarity_score'], 3)
                    } for chunk in relevant_chunks[:3]
                ]
            }
            
        except Exception as e:
            # If Gemini fails, provide an extractive answer from retrieved chunks
            # Create a better formatted answer from top chunks
            if len(relevant_chunks) == 0:
                extractive_answer = "I couldn't find relevant information in the document to answer your question. Please try rephrasing your question or ask about content that's actually in the document."
            else:
                # Take top chunks and create a more coherent answer
                top_chunks = relevant_chunks[:5]  # Use more chunks for better context
                
                # Extract key terms from the question for better matching
                question_lower = query.lower()
                key_terms = [term for term in question_lower.split() if len(term) > 3]
                
                # Collect sentences from chunks that are most relevant to the question
                relevant_sentences = []
                for chunk in top_chunks:
                    text = chunk['text'].strip()
                    sentences = [s.strip() for s in text.split('.') if s.strip()]
                    
                    # Score sentences based on relevance to question
                    for sentence in sentences:
                        if len(sentence) < 10:
                            continue
                            
                        sentence_lower = sentence.lower()
                        # Score based on keyword matches
                        score = 0
                        for term in key_terms:
                            if term in sentence_lower:
                                score += 1
                        
                        # Also check for question words
                        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
                        if any(qw in question_lower for qw in question_words):
                            # Look for answers that might contain definitions or explanations
                            if 'is ' in sentence_lower or 'are ' in sentence_lower or 'means ' in sentence_lower:
                                score += 2
                        
                        if score > 0:
                            relevant_sentences.append((score, sentence, chunk['similarity_score']))
                
                # Sort by score and similarity
                relevant_sentences.sort(key=lambda x: (x[0], x[2]), reverse=True)
                
                # Take top 5-7 sentences
                top_sentences = [s[1] for s in relevant_sentences[:7]]
                
                if top_sentences:
                    # Format as a structured bullet-point answer
                    import re
                    
                    cleaned_sentences = []
                    for sentence in top_sentences:
                        clean = sentence.strip()
                        clean = re.sub(r'\s+', ' ', clean).strip()
                        if not clean.endswith('.'):
                            clean += '.'
                        cleaned_sentences.append(clean)
                    
                    extractive_answer = "**Key points from the document:**\n\n"
                    bullets = cleaned_sentences[:7]
                    for sentence in bullets:
                        if sentence.strip().startswith(('-', '•', '*', '·')):
                            extractive_answer += f"{sentence}\n"
                        else:
                            extractive_answer += f"• {sentence}\n"
                    
                    # Limit length
                    if len(extractive_answer) > 1000:
                        extractive_answer = extractive_answer[:1000] + '...'
                    
                    # Add context about the retrieval
                    if len(relevant_chunks) > 5:
                        extractive_answer += f"\n\n*Note: Found {len(relevant_chunks)} relevant sections. Using extractive summarization since AI synthesis is temporarily unavailable.*"
                else:
                    # Fallback to original method if no good sentences found
                    import re
                    # Combine text from top chunks
                    combined_text = ' '.join(chunk['text'].strip() for chunk in top_chunks[:3])
                    # Clean numbering and extra whitespace
                    combined_text = re.sub(r'\s*\d+[\.\)]\s*', ' ', combined_text)
                    combined_text = re.sub(r'\s+', ' ', combined_text).strip()
                    
                    # Try to structure the combined text
                    sentences = [s.strip() for s in combined_text.split('.') if s.strip()]
                    if len(sentences) > 1:
                        # Format as bullet points
                        extractive_answer = "**Relevant information from the document:**\n\n"
                        for i, sentence in enumerate(sentences[:8]):  # Limit to 8 sentences
                            if sentence:
                                extractive_answer += f"• {sentence}.\n"
                    else:
                        extractive_answer = combined_text
                    
                    # Truncate if too long
                    if len(extractive_answer) > 800:
                        last_period = extractive_answer[:800].rfind('.')
                        if last_period > 500:
                            extractive_answer = extractive_answer[:last_period + 1]
                        else:
                            extractive_answer = extractive_answer[:800] + '...'
            
            # Clean answer of unwanted content
            extractive_answer = self._clean_answer(extractive_answer)
            
            return {
                "answer": extractive_answer,
                "error": f"Gemini API unavailable: {str(e)[:100]}",
                "rag_info": {
                    "status": "partial_success",
                    "retrieval_method": "FAISS Vector Search",
                    "embedding_model": "HuggingFace Sentence Transformers",
                    "chunks_retrieved": len(relevant_chunks),
                    "fallback_used": "extractive",
                    "similarity_threshold": self.similarity_threshold,
                    "top_chunks_used": min(5, len(relevant_chunks))
                },
                "retrieved_chunks": [
                    {
                        "text": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                        "similarity": round(chunk['similarity_score'], 3)
                    } for chunk in relevant_chunks[:2]
                ]
            }
    
    def get_rag_stats(self) -> Dict:
        """
        Get RAG system statistics
        """
        return {
            "total_chunks": len(self.document_chunks),
            "vector_dimension": self.embedding_model.get_sentence_embedding_dimension(),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "vector_store": "FAISS",
            "similarity_method": "Cosine Similarity",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_threshold": self.similarity_threshold,
            "max_retrieved_chunks": self.max_retrieved_chunks,
            "rag_architecture": "Retrieval-Augmented Generation with Vector Search"
        }
    
    def get_document_count(self):
        """Get total number of documents in the system"""
        ids = {self._chunk_doc_id(c) for c in self.document_chunks}
        return len(ids - {None})

    def get_total_chunks(self):
        """Get total number of chunks across all documents"""
        return len(self.document_chunks)

    def get_document_info(self, document_id):
        """Get information about a specific document"""
        doc_chunks = [
            chunk
            for chunk in self.document_chunks
            if self._chunk_doc_id(chunk) == document_id
        ]
        if doc_chunks:
            return {
                'document_id': document_id,
                'chunk_count': len(doc_chunks),
                'metadata': doc_chunks[0]['metadata'] if doc_chunks[0]['metadata'] else {},
                'added_at': doc_chunks[0]['metadata'].get('upload_timestamp', 'unknown') if doc_chunks[0]['metadata'] else 'unknown'
            }
        return None
    
    def remove_document(self, document_id):
        """Remove a document from the RAG system"""
        initial_count = len(self.document_chunks)
        self.document_chunks = [
            chunk
            for chunk in self.document_chunks
            if self._chunk_doc_id(chunk) != document_id
        ]
        
        if len(self.document_chunks) < initial_count:
            self.chunk_metadata = [c["metadata"] for c in self.document_chunks]
            self._rebuild_vector_index()
            print(f"Document {document_id} removed from RAG system")
            return True
        return False
    
    def _rebuild_vector_index(self):
        """Rebuild the vector index after document removal"""
        self.chunk_metadata = [c["metadata"] for c in self.document_chunks]
        if self.document_chunks:
            chunk_texts = [chunk["text"] for chunk in self.document_chunks]

            embeddings = self.embedding_model.encode(
                chunk_texts, convert_to_tensor=False
            )
            embeddings = np.array(embeddings).astype("float32")

            faiss.normalize_L2(embeddings)

            self.vector_store = faiss.IndexFlatIP(  # type: ignore[call-arg]
                self.embedding_dimension
            )
            self.vector_store.add(embeddings)  # type: ignore[arg-type]

            print(f"Vector index rebuilt with {len(chunk_texts)} chunks")
        else:
            self.vector_store = faiss.IndexFlatIP(  # type: ignore[call-arg]
                self.embedding_dimension
            )
    
    def search_similar_content(self, query, top_k=5, min_similarity=0.7):
        """Search for similar content across all documents"""
        if len(self.document_chunks) == 0 or self.vector_store is None:
            return []
        
        # Generate embedding for query
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search in vector store
        scores, indices = self.vector_store.search(  # type: ignore[union-attr]
            query_embedding, min(top_k, len(self.document_chunks))
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= min_similarity and idx < len(self.document_chunks):
                chunk = self.document_chunks[idx]
                results.append({
                    "document_id": self._chunk_doc_id(chunk),
                    'chunk_text': chunk['text'],
                    'similarity_score': float(score),
                    'metadata': chunk['metadata']
                })
        
        return results

# RAG system will be initialized in app.py
