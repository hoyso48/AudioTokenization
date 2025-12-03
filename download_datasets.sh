#!/bin/bash

# Config: Where to save the data
# Change this if you have a larger disk mounted elsewhere!
DATA_ROOT="/home/hoyso/projects/datasets"

mkdir -p "$DATA_ROOT"
echo "📂 Downloading datasets to: $DATA_ROOT"

# ==========================================
# 1. LibriSpeech (ASR, PR)
# ==========================================
echo "⬇️  [1/4] Downloading LibriSpeech..."
mkdir -p "$DATA_ROOT/LibriSpeech"
cd "$DATA_ROOT"

# Download parts
wget -c https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget -c https://www.openslr.org/resources/12/dev-clean.tar.gz
wget -c https://www.openslr.org/resources/12/test-clean.tar.gz

# Extract
echo "📦 Extracting LibriSpeech..."
tar -zxf train-clean-100.tar.gz
tar -zxf dev-clean.tar.gz
tar -zxf test-clean.tar.gz

# Cleanup (Optional: uncomment to save space)
# rm train-clean-100.tar.gz dev-clean.tar.gz test-clean.tar.gz

# ==========================================
# 2. Speech Commands (KS)
# ==========================================
echo "⬇️  [2/4] Downloading Speech Commands..."
SC_ROOT="$DATA_ROOT/speech_commands_v0.01"
SC_TEST_ROOT="$DATA_ROOT/speech_commands_test_set_v0.01"
mkdir -p "$SC_ROOT" "$SC_TEST_ROOT"

wget -c http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz -P "$DATA_ROOT"
wget -c http://download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz -P "$DATA_ROOT"

echo "📦 Extracting Speech Commands..."
tar -zxf "$DATA_ROOT/speech_commands_v0.01.tar.gz" -C "$SC_ROOT"
tar -zxf "$DATA_ROOT/speech_commands_test_set_v0.01.tar.gz" -C "$SC_TEST_ROOT"

# ==========================================
# 3. VoxCeleb1 (SID)
# ==========================================
echo "⬇️  [3/4] Downloading VoxCeleb1..."
VC_ROOT="$DATA_ROOT/VoxCeleb1"
mkdir -p "$VC_ROOT/dev" "$VC_ROOT/test"

# Dev Set (Split parts from HuggingFace mirror as recommended by S3PRL)
cd "$VC_ROOT/dev"
wget -c https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav_partaa
wget -c https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav_partab
wget -c https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav_partac
wget -c https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav_partad

echo "📦 Merging and Unzipping VoxCeleb1 Dev..."
cat vox1_dev_wav_part* > vox1_dev_wav.zip
unzip -q -o vox1_dev_wav.zip
# mv wav/* . # Structure fix might be needed depending on unzip result

# Test Set
cd "$VC_ROOT/test"
wget -c https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_test_wav.zip
echo "📦 Unzipping VoxCeleb1 Test..."
unzip -q -o vox1_test_wav.zip

# ==========================================
# 4. Fluent Speech Commands (IC)
# ==========================================
echo "⬇️  [4/4] Downloading Fluent Speech Commands..."
cd "$DATA_ROOT"
wget -c "https://huggingface.co/datasets/leo19941227/fluent_speech_commands/resolve/main/fluent.tar.gz"
echo "📦 Extracting Fluent Speech Commands..."
tar -zxf fluent.tar.gz

echo "✅ All public datasets downloaded!"
echo "⚠️  NOTE: IEMOCAP (for ER task) requires manual request from https://sail.usc.edu/iemocap/"