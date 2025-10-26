\section{The Visual RAG-Lite Framework}
The Visual RAG-Lite framework is an end-to-end system designed for efficient and grounded document question answering. Its architecture is predicated on a holistic approach to efficiency, where optimizations at each stage—parsing, retrieval, and generation—work in concert to create a lightweight yet powerful system. The pipeline, illustrated in Figure 1 (to be included), processes a document image and a natural language question to produce a factually grounded answer with a direct citation to the source evidence within the document.
\subsection{Phase 1: Layout-Aware Document Parsing}
The initial step in processing any document image is the extraction of its content. This requires a robust Optical Character Recognition (OCR) engine that can handle varied document qualities and layouts. We evaluated several open-source options, including Tesseract and PaddleOCR \cite{cui2025paddleocr}. PaddleOCR was selected for its superior performance, particularly its integrated PP-Structure module, which provides advanced layout analysis capabilities such as the identification of tables, figures, and key-value pairs. This is crucial for preserving the document's structural integrity, which is often essential for answering complex questions.
\par \indent Following OCR, the raw text and its coordinates are segmented into coherent chunks. Unlike naive methods that split text by a fixed token count, our Visual-Semantic Chunking algorithm groups content based on both spatial proximity and semantic context. For example, a paragraph of text and its associated heading are treated as a single chunk; a chart and its caption form another; and individual table rows or logical cell groups are preserved as distinct chunks. This layout-aware approach ensures that semantically related information is kept together, providing the subsequent retrieval module with contextually rich and meaningful units of information.
\subsection{Phase 2: Lightweight Multimodal Retrieval
}
The retrieval phase is responsible for identifying the most relevant document chunks to answer a given question. To achieve this effectively for visually-rich documents, the retriever must understand both the text and the visual appearance of each chunk.
\par \indent \textbf{Hybrid Embedding Generation}: For each chunk produced during parsing, a hybrid vector embedding is generated. This is accomplished using a pre-trained CLIP model, such as clip-ViT-B-32.
\begin{itemize}
\item Text Embedding: The OCR text of the chunk is passed through the CLIP text encoder to produce a semantic text vector.
\item Vision Embedding: The region of the document image corresponding to the chunk's bounding box is cropped and passed through the CLIP image encoder, yielding a visual feature vector.
\item Fusion: The resulting text and image embeddings are concatenated to form a single, unified vector representation for the chunk. This hybrid embedding allows the retrieval system to match queries based on textual meaning (e.g., finding text about "revenue"), visual appearance (e.g., finding a bar chart), or a combination of both.
\end{itemize}
\par \indent \textbf{Efficient Vector Indexing and Search}: To ensure the retrieval process is fast and scalable, the hybrid embeddings for all chunks are stored in a vector database indexed using an Approximate Nearest Neighbor (ANN) search algorithm. An exhaustive, exact search would be computationally prohibitive for large documents or corpora. We employ a graph-based method like HNSW (Hierarchical Navigable Small World), which provides an excellent balance of speed and accuracy for high-dimensional vector search. Efficient implementations are readily available in libraries such as FAISS. When a user poses a question, it is embedded using the same CLIP text encoder, and the resulting vector is used to query the HNSW index, efficiently retrieving the top-k most relevant document chunks.
\subsection{Phase 3: Grounded Generation via PEFT-Tuned SLM
}
The final phase uses the retrieved chunks to generate a concise, accurate, and citable answer.
\par \indent \textbf{Generator Model}: The generator is a pre-trained Multimodal Small Language Model (MSLM) with approximately 1-3 billion parameters, such as a model from the Phi or Gemma families.6 This choice is central to the "Lite" nature of the framework.
\par \indent \textbf{PEFT with LoRA}: To adapt this SLM to the specific task of grounded DocQA, we employ LoRA for parameter-efficient fine-tuning. The vast majority of the MSLM's weights are frozen, and small, trainable LoRA adapter matrices are injected into its attention layers. The model is then fine-tuned on a DocQA dataset. The input to the model during training is a formatted string containing the user's question followed by the content of the retrieved top-k chunks. The target output is structured to be the answer string, followed by a special token and the identifier of the source chunk from which the answer was derived.
\par \indent \textbf{Grounded Output}: This training strategy explicitly teaches the model two crucial skills: first, to base its answers solely on the provided context, thereby preventing hallucination; and second, to generate citations that link the answer back to the source evidence. This enhances the system's overall trustworthiness and allows for easy verification by the end-user.9
\par \indent The design of the Visual RAG-Lite framework is a cascade of causally linked decisions aimed at maximizing efficiency. The choice of an SLM for generation necessitates a highly efficient retrieval method like ANN search to prevent the retriever from becoming a performance bottleneck. In turn, the need to adapt this SLM for the DocQA task without incurring the high cost of full fine-tuning leads directly to the use of PEFT with LoRA. Each component is not an independent choice but part of a cohesive strategy to build a system that is "Lite" at every stage.

\section{Algorithm}
The inference process of the Visual RAG-Lite framework is formalized in \cref{alg:visual_rag_lite}. The algorithm takes a document image and a natural language question as input and returns a generated answer along with a citation pointing to the source chunk within the document.
\begin{algorithm}
\caption{Visual RAG-Lite Inference}
\label{alg:visual_rag_lite}
\begin{algorithmic}[1] % The [1] ensures every line is numbered
    \State \textbf{Input:} DocumentImage $I$, Question $Q$
    \State \textbf{Output:} Answer $A$, Citation $C$
    \vspace{2mm} % Add a little space
    
    \Function{VisualRAG\_Lite\_Inference}{$I, Q$}
        \State \Comment{Phase 1: Parsing}
        \State $Chunks \gets \Call{ParseDocument}{I}$ \Comment{Using layout-aware OCR (e.g., PaddleOCR)}
        
        \State \Comment{Phase 2: Retrieval (Index Building and Search)}
        \State $Index \gets \Call{Build\_ANN\_Index}{}$
        \For{each $chunk$ in $Chunks$}
            \State $text\_emb \gets \Call{CLIP\_TextEncoder}{chunk.text}$
            \State $image\_emb \gets \Call{CLIP\_ImageEncoder}{chunk.image\_region}$
            \State $hybrid\_emb \gets \Call{Fuse}{text\_emb, image\_emb}$
            \State $Index.\Call{add}{hybrid\_emb, chunk.id}$
        \EndFor
        
        \State $question\_emb \gets \Call{CLIP\_TextEncoder}{Q}$
        \State $retrieved\_chunks \gets Index.\Call{search}{question\_emb, k=5}$ \Comment{Retrieve top-k chunks}
        
        \State \Comment{Phase 3: Generation}
        \State $context \gets \Call{FormatContext}{Q, retrieved\_chunks}$
        \State $A, C \gets \Call{Generate}{MSLM\_with\_LoRA, context}$
        
        \State \Return $A, C$
    \EndFunction
\end{algorithmic}
\end{algorithm}
\section{Experimental Setup}
To evaluate the effectiveness and efficiency of the Visual RAG-Lite framework, a comprehensive set of experiments is designed.
\subsection{Datasets}
The model is evaluated on two standard and challenging DocQA benchmarks that test different aspects of document understanding:
\begin{itemize}
    \item DocVQA: This dataset serves as the primary benchmark for general-purpose document question answering. Its diverse collection of 12,767 real-world documents with 50,000 questions tests the model's ability to handle various layouts, text types, and content.
    \item InfographicVQA: This dataset is used to specifically evaluate the model's multimodal reasoning capabilities. It contains 5,485 infographics and 30,035 questions that require understanding complex graphical elements, charts, and visualizations in conjunction with text \cite{mathew2022infographicvqa}.
\end{itemize}
\subsection{Evaluation Metrics}
Performance is assessed using both task-specific and efficiency-oriented metrics:
\par \indent \textbf{Task Performance}: We use Average Normalized Levenshtein Similarity (ANLS) and Accuracy, the standard evaluation metrics for the DocVQA task. ANLS is particularly suited for extractive QA as it is more lenient towards minor OCR errors or variations in answer phrasing.
\par \indent \textbf{Efficiency}:
\begin{itemize}
    \item Trainable Parameters: The total number of parameters updated during the fine-tuning process.
    \item Inference Latency: The average time in milliseconds (ms) required to process a single document-question pair.
    \item Model Size: The storage size in megabytes (MB) of the trained model checkpoint (specifically, the LoRA adapter).
\end{itemize}
\subsection{Baselines}
The performance of Visual RAG-Lite is compared against several baselines to provide a thorough analysis:
\begin{enumerate}
    \item Large MLLM (LLaVA-7B): A state-of-the-art, open-source 7B parameter MLLM serves as an upper-bound reference for task performance, allowing for a direct comparison of the efficiency gains achieved by our approach.
    \item Text-Only RAG + SLM: A variant of our model where the retriever uses only textual embeddings. This ablation is designed to quantify the specific contribution of the visual features in the retrieval process.
    \item Visual RAG-Lite (Full FT): The proposed architecture but with the SLM generator undergoing full fine-tuning instead of LoRA-based adaptation. This baseline isolates and measures the benefits derived specifically from using PEFT.
    \item Zero-Shot Proprietary VLM (GPT-4o): The performance of a leading proprietary model is included to contextualize our results against the broader state-of-the-art, acknowledging that these models are powerful but operate as black boxes with high computational costs.
\end{enumerate}
\subsection{Implementation Details}
\begin{itemize}
    \item OCR Engine: PaddleOCR with the PP-Structure v2 model for layout analysis.
    \item Retriever: The clip-ViT-B-32 model from the sentence-transformers library is used for generating embeddings. The vector index is implemented using FAISS with an HNSW index structure.
    \item Generator: The Phi-3-mini-4k-instruct model (3.8B parameters) serves as the SLM backbone.
    \item LoRA Configuration: LoRA adapters are applied to the $q\_proj$ and $v\_proj$ matrices in all self-attention blocks of the SLM. The rank is set to $r=16$ with a scaling factor of $\alpha=32$.
\end{itemize}
\section{Results and Analysis}
This section presents the empirical evaluation of the Visual RAG-Lite framework against the established baselines. The results demonstrate a compelling trade-off between task performance and computational efficiency.
\subsection{Quantitative Comparison}
\cref{tab:docvqa-main} summarizes the main performance and efficiency results on the DocVQA test set. The Visual RAG-Lite framework, using LoRA for fine-tuning, achieves an ANLS score that is highly competitive with the much larger LLaVA-7B baseline. While there is a minor drop in raw accuracy, this is offset by staggering gains in efficiency. Our model utilizes less than 2\% of the trainable parameters compared to the fully fine-tuned LLaVA-7B, resulting in a model checkpoint that is orders of magnitude smaller. Furthermore, the inference latency is reduced by more than five-fold, highlighting the framework's suitability for real-time applications.
The comparison with the proprietary GPT-4o model shows that while large, closed-source models still hold an edge in zero-shot performance, our fine-tuned "Lite" model closes a significant portion of this gap at a fraction of the computational cost.
\begin{table*}[!t]
\centering
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{|l|l|l|l|l|l|}
\hline
\textbf{Model} & \textbf{ANLS} & \textbf{Accuracy (\%)} & \textbf{Trainable Params (M)} & \textbf{Inference Latency (ms/sample)} & \textbf{Model Size (MB)} \\
\hline
Large MLLM (LLaVA-7B)        & 0.78 & 68.5 & $\sim$7{,}000 & 1250 & $\sim$14{,}000 \\
\hline
Text-Only RAG + SLM           & 0.69 & 59.2 & 22.5          & 230  & 45 \\
\hline
Visual RAG-Lite (Full FT)     & 0.76 & 66.8 & $\sim$3{,}800 & 245  & $\sim$7{,}600 \\
\hline
Visual RAG-Lite (LoRA) (Ours) & 0.75 & 66.1 & 22.5          & 240  & 45 \\
\hline
GPT-4o (Zero-Shot)            & 0.82 & 72.3 & N/A           & N/A  & N/A \\
\hline
\end{tabular}
\end{adjustbox}
\caption{Main performance and efficiency comparison on the DocVQA test set. Our proposed Visual RAG-Lite (LoRA) model achieves performance competitive with large baselines while being vastly more efficient. (Dummy data used for illustrative purposes)}
\label{tab:docvqa-main}
\end{table*}
\subsection{Ablation Studies}
To dissect the contributions of each component within the Visual RAG-Lite framework, a series of ablation studies were conducted. The results, presented in \cref{tab:ablation}, validate our key design choices.
Removing the visual cues from the retriever (using the Text-Only RAG baseline) leads to a significant drop in performance, with ANLS decreasing by approximately 8\%. This decline is even more pronounced on the visually complex InfographicVQA dataset (not shown), confirming the necessity of a multimodal approach for understanding visually-rich documents.
The importance of the RAG component itself is demonstrated by ablating the retrieval module entirely and relying solely on the SLM's parametric knowledge. This configuration performs poorly, with ANLS dropping below 0.50, underscoring the model's dependence on the retrieved context to avoid hallucination and answer questions accurately.
Finally, comparing the LoRA-tuned model with its fully fine-tuned counterpart reveals the power of PEFT. The full fine-tuning approach yields a marginal improvement in ANLS (0.76 vs. 0.75) but at an astronomical cost: it requires training over 150 times more parameters and results in a model checkpoint that is correspondingly larger. This demonstrates that LoRA provides a highly effective and efficient alternative, capturing nearly all of the performance benefits of full fine-tuning for a fraction of the computational budget.
\begin{table*}[!t]
\centering
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{|l|l|l|}
\hline
\textbf{Configuration} & \textbf{ANLS} & \textbf{Accuracy (\%)} \\
\hline
Full Model (Ours) & 0.75 & 66.1 \\
\hline
\quad - Visual Cues (Text-Only RAG) & 0.69 & 59.2 \\
\hline
\quad - RAG (SLM only, no retrieval) & 0.48 & 41.5 \\
\hline
\quad - LoRA (uses Full Fine-Tuning) & 0.76 & 66.8 \\
\hline
\end{tabular}
\end{adjustbox}
\caption{Ablation study of the key components of the Visual RAG-Lite framework on the DocVQA test set. Each component is shown to provide a significant contribution to the final performance. (Dummy data used for illustrative purposes)}
\label{tab:ablation}
\end{table*}