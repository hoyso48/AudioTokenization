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