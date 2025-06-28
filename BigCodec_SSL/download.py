import os
import torchaudio
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="../../datasets", help="Path to save the dataset")
    args = parser.parse_args()
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 모든 서브셋 리스트
    subsets = [
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
        "train-clean-100",
        "train-clean-360",
        "train-other-500"
    ]

    # 각 서브셋 별로 다운로드
    for subset in subsets:
        print(f"Downloading {subset}...")
        dataset = torchaudio.datasets.LIBRITTS(save_dir, 
        url = subset, 
        folder_in_archive= 'LibriTTS', 
        download = True)
        print(dataset)
        print(f"Completed downloading {subset}")