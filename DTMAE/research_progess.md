# Research Proposal: Semantically Aligned Audio Tokenizer with Dynamic Token Masking

**Motivation:** To enhance the compatibility of acoustic tokenizers with multimodal Large Language Models (LLMs) and various downstream tasks.

**Target Conferences:** ICML, InterSpeech

---

## 1. Related Work

### 1.1. Neural Audio Codecs (e.g., SoundStream, DAC, Encodec)
*   **Contribution:** Pioneered the RVQ-GAN based architecture for neural audio codecs.
*   **Limitations:** The use of Residual Vector Quantization (RVQ) complicates LLM modeling, as it requires multiple generation steps to produce all RVQ tokens.

### 1.2. Acoustic Tokenizers with a Single Quantizer (e.g., WavTokenizer, BigCodec)
*   **Contribution:** Employed a single quantizer, which significantly simplified LLM integration by enabling 1-stage language modeling.
*   **Limitations:** These models still focus excessively on acoustic (local, random) information, lacking the semantic context that LLMs need to effectively understand the tokens.

### 1.3. Self-Supervised Learning (SSL) (e.g., wav2vec, HuBERT, WavLM, Audio-MAE)
*   **Contribution:** Produce strong semantic representations useful for many downstream tasks. "Semantic tokens" can be obtained by applying quantization to the output features of these models.

### 1.4. Semantic Distillation for Acoustic Tokens (e.g., X-codec1, 2)
*   **Contribution:** Injected semantic features into acoustic tokens by concatenating SSL model features and adding an auxiliary semantic reconstruction loss, *without sacrificing acoustic reconstruction quality*.
*   **Limitations:** Semantic tokens derived from SSL models often exhibit high temporal correlation (redundancy). It is hypothesized that X-codec likely faces a similar issue.

### 1.5. Dynamic Token Pooling (e.g., ToMe, ToMeSD, DTP)
*   **Contribution:** ToMe and ToMeSD demonstrated that heuristic token merging (bipartite soft matching) can improve ViT's throughput without performance loss. DTP showed a differentiable approach for training tokenizer-free LMs.

### 1.6. Variable Frame Rate (VFR) Tokens (e.g., CodecSlime, FlexiCodec)
*   **Contribution:** These approaches utilize the flexibility of variable frame rate tokens.
*   **CodecSlime:** First trains a codec with random temporal downsampling and then finds optimal merging boundaries using Dynamic Programming to reduce Word Error Rate (WER).
*   **FlexiCodec:** Uses a pre-trained ASR Encoder and a cosine similarity threshold `sim(x_{i-1}, x_i)` to determine dynamic boundaries. It claims that dynamic token merging can mitigate semantic information loss at extremely low frame rates (<10Hz).

---

## 2. Proposed Idea: Dynamic Token Masking (DTM)

The core idea is to perform **Acoustic Reconstruction via Dynamic Token Masking (DTM)**. Instead of simply pooling or merging tokens, we propose to replace redundant tokens with a special `<MASK>` token and task the decoder with reconstructing them.

This is similar to Masked Autoencoders (MAE), but our primary objective is to first maximize the model's **acoustic reconstruction ability**, not semantic feature extraction. Simultaneously, we want the model to learn a degree of semantic information through this masked reconstruction task.

### 2.1. Approaches to Dynamic Token Selection
We focus on heuristic approaches due to the training instability of fully differentiable methods.

1.  **Bipartite Matching (ToMe-style)**
2.  **Top-K Selection**
3.  **Greedy Merging**
4.  **Dynamic Programming (DP)**
5.  **Path Length Equalization (PLE):** A novel and efficient method.

### 2.2. Path Length Equalization (PLE)

Most token merging algorithms are based on cosine similarity, but methods other than Top-K significantly increase training and inference time. Top-K often fails, hypothesized to be due to the slow drifting of acoustic features.

PLE efficiently solves this problem. It operates as follows:
1.  Calculate the pairwise distance: `d_i = 1 - sim(x_{i-1}, x_i)`
2.  Compute the accumulated distance: `s_i = sum_{j=1 to i} d_j`
3.  Divide the accumulated distance `s` into `k` equal intervals. The tokens at the boundaries of these intervals are marked as frontiers (kept), while others are considered redundant (to be masked).

PLE achieves state-of-the-art performance with negligible computational overhead.

---

## 3. Experimental Setup

*   **Architecture:** Conformer or Transformer-based model with an iSTFT-based decoder (Vocos-like) for reconstruction, enhanced with a GAN.
*   **Model Design:**
    `GT Audio -> Feature Extractor -> Encoder(L1) -> DTM -> Encoder(L2) -> VQ -> Decoder(L2) -> Upsampling w/ <MASK> -> Decoder(L1) -> iSTFT -> Reconstructed Audio`
*   **Model Size:** 30M (base), 300M (default)
*   **Dataset:** LibriSpeech-960
*   **Frame Rates:** Level 1 @ 100Hz, Level 2 @ 50Hz

#### Configuration (for same bitrate comparison):
*   **With DTM:** Single VQ layer with 16,384 codes.
    *   Bitrate: 700bps (content) + 100bps (positional info for masks) = **800bps total**.
*   **Without DTM:** DTM and Upsampling layers are replaced with Conv down/up-sampling. Single VQ layer with 65,536 codes.
    *   Bitrate: **800bps total**.

### 3.1. Evaluation
*   **Current Reconstruction Metrics:** PESQ, STOI, WER, Speaker Similarity (SSIM), Mel Cepstral Distance (MCD).
*   **TODO:** Implement a semantic evaluation pipeline (referencing CodecBench or SSL literature).

---

## 4. Preliminary Findings

1.  **"Upsampling with <MASK> tokens"** significantly outperforms simple repetition-based upsampling for all tested dynamic token pooling algorithms. This validates the **Dynamic Token Masking** approach.
2.  Masking is conceptually similar to a weighted averaging form of pooling. The entire pooling/un-pooling process is effectively replaced by masking.
3.  **With DTM >> Without DTM** in reconstruction quality, even at the same bitrate.
4.  DTM training strategies were compared:
    *   (1) Training with random boundaries, inference with PLE.
    *   (2) Training and inference both with PLE.
    *   **Result:** For reconstruction, `(2) > (1) > Without DTM`.
    *   **Hypothesis to test:** Will strategy (1) prove superior to (2) in semantic evaluation?
5.  The 300M model with DTM scales well, outperforming the 30M model and achieving SOTA results at a comparable bitrate.

\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{LibriSpeech test-clean reconstruction metrics (count = 2620). Bold = global best; underline = best within its subgroup; higher is better except MCD and WER.}
\begin{tabular}{lcccccccc}
\toprule
\textbf{Inference setup} & \textbf{kbps} & PESQ$_\text{WB}$ ↑ & PESQ$_\text{NB}$ ↑ & STOI ↑ & MCD ↓ & Spk-Sim ↑ & WER ↓ & UTMOS ↑ \\
\midrule
Ground Truth & - & 4.644 & 4.549 & 1.000 & - & 1.000 & 2.077 & 4.086 \\
\midrule
\multicolumn{9}{l}{\textbf{(i) Fixed pattern masking}} \\
\quad Fixed stride & 0.8 & 2.351 & 3.070 & 0.919 & 3.933 & 0.699 & 3.483 & 4.033 \\
\midrule
\multicolumn{9}{l}{\textbf{(ii) PLE training}} \\
\quad PLE & 0.8 & \underline{2.668} & \underline{3.343} & \underline{0.934} & \underline{\textbf{3.683}} & \underline{0.781} & \underline{2.876} & \underline{\textbf{4.216}} \\
\quad PLE @25hz & 0.4 & 2.072 & 2.782 & 0.895 & 4.672 & 0.575 & 4.742 & 4.107 \\
\midrule
\multicolumn{9}{l}{\textbf{(iii) Random masking ($p=0.5$) training}} \\
\quad PLE & 0.8 & \underline{2.564} & \underline{3.251} & \underline{0.927} & \underline{3.795} & \underline{0.728} & \underline{3.036} & \underline{4.194} \\
\quad Top-K & 0.8 & 2.096 & 2.846 & 0.897 & 4.567 & 0.709 & 6.260 & 3.501 \\
\quad Greedy & 0.8 & 2.512 & 3.216 & 0.924 & 3.903 & 0.726 & 3.161 & 4.134 \\
\midrule
\quad X-Codec2 & 0.8 & 2.430 & 3.040 & 0.920 & - & 0.820 & 2.470 & 4.130 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{LibriSpeech test-clean (4--10s filtered) reconstruction metrics (count = 1237). Bold = global best; underline = best within its subgroup; higher is better except MCD and WER.}
\begin{tabular}{lcccccccc}
\toprule
\textbf{Inference setup} & \textbf{kbps} & PESQ$_\text{WB}$ ↑ & PESQ$_\text{NB}$ ↑ & STOI ↑ & MCD ↓ & Spk-Sim ↑ & WER ↓ & UTMOS ↑ \\
\midrule
Ground Truth & - & 4.644 & 4.549 & 1.000 & - & 1.000 & 2.181 & 4.103 \\
\midrule
\multicolumn{9}{l}{\textbf{(i) Fixed pattern masking}} \\
\quad Fixed stride & 0.8 & 2.351 & 3.068 & 0.920 & 3.925 & 0.706 & 3.627 & 4.047 \\
\midrule
\multicolumn{9}{l}{\textbf{(ii) PLE training}} \\
\quad PLE & 0.8 & \underline{\textbf{2.650}} & \underline{3.329} & \underline{\textbf{0.933}} & \underline{3.679} & \underline{0.785} & \underline{3.187} & \underline{\textbf{4.225}} \\
\quad PLE @25hz & 0.4 & 2.068 & 2.774 & 0.895 & 4.670 & 0.575 & 5.136 & 4.128 \\
\midrule
\multicolumn{9}{l}{\textbf{(iii) Random masking ($p=0.5$) training}} \\
\quad PLE & 0.8 & \underline{2.552} & \underline{3.243} & \underline{0.927} & \underline{3.795} & \underline{0.731} & 3.441 & \underline{4.207} \\
\quad Top-K & 0.8 & 2.087 & 2.835 & 0.898 & 4.549 & 0.715 & 6.528 & 3.524 \\
\quad Greedy & 0.8 & 2.509 & 3.213 & 0.925 & 3.895 & 0.730 & \underline{3.359} & 4.149 \\
\midrule
\multicolumn{9}{l}{\textbf{(iv) FlexiCodec}} \\
\quad FlexiCodec @12.5Hz & 1.30 & - & \underline{\textbf{3.350}} & - & \underline{\textbf{2.760}} & 0.85 & \underline{\textbf{2.230}} & \underline{4.220} \\
\quad FlexiCodec @8.3Hz & 0.85 & - & 3.030 & - & 3.100 & \underline{0.780} & 2.280 & 4.210 \\
\quad FlexiCodec @6.25Hz & 0.64 & - & 2.760 & - & 3.420 & 0.710 & 2.530 & 4.180 \\
\midrule
\quad X-Codec2 & 0.8 & - & 2.770 & - & 3.650 & \textbf{0.820} & 2.800 & 4.080 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{LibriSpeech test-clean upsampling ablation (PLE 50Hz; count = 2620).}
\begin{tabular}{lcccccccc}
\toprule
\textbf{Upsampling} & \textbf{kbps} & PESQ$_\text{WB}$ ↑ & PESQ$_\text{NB}$ ↑ & STOI ↑ & MCD ↓ & Spk-Sim ↑ & WER ↓ & UTMOS ↑ \\
\midrule
Mask token upsampling & 0.8 & \textbf{2.668} & \textbf{3.343} & \textbf{0.934} & \textbf{3.683} & 0.781 & \textbf{2.876} & \textbf{4.216} \\
Repeat upsampling & 0.8 & 2.618 & 3.305 & 0.932 & 3.684 & \textbf{0.794} & 3.133 & 4.179 \\
\bottomrule
\end{tabular}
\end{table}

---

## 5. Critical Questions & Future Work

### 5.1. MUST DO
*   **Connecting DTM to Semantic Learning:** The most critical question is: *Can we establish a strong link between DTM and semantic learning?*
    *   **Problem:** The large receptive field of `Encoder(L1)` might prevent the masked reconstruction from being a consistent SSL signal (i.e., it's too easy for the model).
    *   **Challenge:** However, removing or replacing `Encoder(L1)` with a simple CNN encoder significantly degrades reconstruction performance, which is our primary goal.
    *   **Potential Solutions:** How can we make masked reconstruction a stronger SSL signal?
        *   Use a very low learning rate for `Encoder(L1)`.
        *   Employ a momentum update mechanism for `Encoder(L1)`. Are these sufficient?
*   **Implement Semantic Evaluation Pipeline.**

### 5.2. OPTIONAL
*   **Variable Frame Rate:** The current DTP implementation selects a fixed number of tokens.
    *   **Needed:** Implement a threshold-based PLE and add variable-length support to the level 2 Transformer (e.g., via Flash Attention).
*   **2D Patch Tokens:** Explore 2D patch tokens from spectrograms, similar to how RVQ can be viewed as 2D.
*   **General Audio Domain:** Extend training and evaluation to general audio datasets (e.g., AudioSet) since PLE does not rely on speech-specific priors.
*   **Causal/Streaming Support:** Investigate adapting the model for streaming.
    *   **Concern:** PLE can be implemented causally, but the MAE-style decoder is inherently non-causal.

### 5.3. Terminology
*   **Paper Title:** Needs a compelling name.
*   **Key Concepts:**
    *   Dynamic Token Masking (DTM)?
    *   Model Name (DTMAE)?
    *   Upsampling with `<MASK>` tokens?
    *   Level 1, Level 2 -> Acoustic, Semantic levels?
    *   Path Length Equalization (PLE)?

### ONGOING work
*   5.2.1: VFR implementation - currently supported via tau estimation in batch-topk manner, and the training efficiency remains by utilizing varlen kernel of flash attention-v2.
    **problem** current way of estimating tau to get desired global reduction ratio(r) is not stable. ex. if we target r=0.5, estimate the tau based on training data, and using estimated tau gives r=0.6(over-reduction) on the test data. It seems quite random.
    **update** almost done. we use robbins-monro algorithm to estimate tau during training for every algorithm, and just adjust a bit on test-time to match the target r. everything seems alright now.

*   5.1: This part is the most important to-do. we haven't done anything yet.