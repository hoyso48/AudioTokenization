root = "/mnt/disks/persist/data/LibriTTS"

train_list = ['train-clean-100', 'train-clean-360', 'train-other-500']
dev_list = ['dev-clean', 'dev-other']
test_list = ['test-clean', 'test-other']

train_txt = 'filelists/libritts_train_all.txt'
dev_txt = 'filelists/libritts_dev_all.txt'
test_txt = 'filelists/libritts_test_all.txt'

ext = '.wav'

import os
from pathlib import Path

def find_files(base_path, extension):
    files = []
    for root, dirs, filenames in os.walk(base_path):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files

if __name__ == "__main__":
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(train_txt), exist_ok=True)

    # 학습 데이터 파일 리스트 생성
    with open(train_txt, 'w') as f:
        for subset in train_list:
            path = os.path.join(root, subset)
            files = find_files(path, ext)
            f.write('\n'.join(files) + '\n')

    # 검증 데이터 파일 리스트 생성
    with open(dev_txt, 'w') as f:
        for subset in dev_list:
            path = os.path.join(root, subset)
            files = find_files(path, ext)
            f.write('\n'.join(files) + '\n')

    # 테스트 데이터 파일 리스트 생성
    with open(test_txt, 'w') as f:
        for subset in test_list:
            path = os.path.join(root, subset)
            files = find_files(path, ext)
            f.write('\n'.join(files) + '\n')