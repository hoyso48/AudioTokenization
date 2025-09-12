import os
import argparse
import multiprocessing
from joblib import Parallel, delayed
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from tqdm import tqdm
import json


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def authenticate_with_google_drive(
    client_secret_path: str,
    credentials_path: str | None = None,
) -> GoogleDrive:
    """Authenticate with Google Drive using cached credentials when available."""
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile(client_secret_path)

    # Default credentials cache location under ~/.config/pydrive2, derive name from client_id
    if credentials_path is None:
        try:
            with open(client_secret_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            installed = cfg.get("installed") or {}
            client_id = installed.get("client_id") or "default"
        except Exception:
            client_id = "default"
        safe_client_id = client_id.replace(".", "_").replace("-", "_")
        credentials_path = os.path.join(
            os.path.expanduser("~/.config/pydrive2"), f"credentials_{safe_client_id}.json"
        )

    if os.path.exists(credentials_path):
        gauth.LoadCredentialsFile(credentials_path)

    if gauth.credentials is None:
        gauth.CommandLineAuth()
    elif getattr(gauth, "access_token_expired", False):
        gauth.Refresh()
    else:
        gauth.Authorize()

    _ensure_dir(credentials_path)
    gauth.SaveCredentialsFile(credentials_path)

    return GoogleDrive(gauth)


def resolve_remote_folder_spec_to_id(
    drive: GoogleDrive,
    folder_spec: str,
) -> str:
    """Resolve a folder spec to an id. Accepts raw id or 'root/path/to/dir'.

    Does not create missing segments; raises FileNotFoundError if not found.
    """
    if '/' not in folder_spec and folder_spec != 'root':
        return folder_spec

    segments = [seg for seg in folder_spec.strip('/').split('/') if seg and seg != '.']
    if segments and segments[0] == 'root':
        segments = segments[1:]
    current_parent_id = 'root'

    def find_child_by_title(drive, parent_id, title, mime_type=None):
        query = f"'{parent_id}' in parents and trashed=false"
        if mime_type is not None:
            query += f" and mimeType='{mime_type}'"
        for child in drive.ListFile({'q': query}).GetList():
            if child.get('title') == title:
                return child
        return None

    for segment in segments:
        existing = find_child_by_title(
            drive, current_parent_id, segment, mime_type='application/vnd.google-apps.folder'
        )
        if existing is None:
            raise FileNotFoundError(
                f"Remote path segment not found under parent {current_parent_id}: {segment}"
            )
        current_parent_id = existing['id']

    return current_parent_id

def download_one_file(file, target_directory):
    """Download a single file to the target directory."""
    file_path = os.path.join(target_directory, file['title'])
    print(f"[Thread] Downloading: {file['title']} -> {file_path}")
    file.GetContentFile(file_path)


def download_single_file_by_id(file_id: str, target_directory: str, drive: GoogleDrive):
    os.makedirs(target_directory, exist_ok=True)
    f = drive.CreateFile({'id': file_id})
    f.FetchMetadata(fields='title')
    download_one_file(f, target_directory)

def download_files_in_folder(folder_id, target_directory, drive):
    """
    Recursively download all files in the given folder and its subfolders.
    The folder structure is preserved.
    
    Args:
        folder_id (str): ID of the Google Drive folder
        target_directory (str): Local path to store downloaded files
        drive (GoogleDrive): An authenticated GoogleDrive instance
    """
    os.makedirs(target_directory, exist_ok=True)

    # search for files that are not in trash
    query = f"'{folder_id}' in parents and trashed=false"
    file_list = drive.ListFile({'q': query}).GetList()

    # separate files and folders
    files = []
    folders = []
    for f in file_list:
        if f['mimeType'] == 'application/vnd.google-apps.folder':
            folders.append(f)
        else:
            files.append(f)

    # download current folder's files in parallel
    num_cores = multiprocessing.cpu_count()
    if files:
        Parallel(n_jobs=num_cores, backend='threading')(
            delayed(download_one_file)(f, target_directory) for f in tqdm(files)
        )

    # recursive call for subfolders
    for folder in folders:
        subfolder_path = os.path.join(target_directory, folder['title'])
        download_files_in_folder(folder['id'], subfolder_path, drive)

def main(folder_id, target_dir, redownload, client_secret_pth, credentials_pth, file_spec=None):
    """
    Main workflow:
      1) If target_dir exists and redownload is False, skip downloading.
      2) Otherwise, authenticate with Google Drive and download files from folder_id.
    """
    if os.path.exists(target_dir) and not redownload:
        print("Target directory already exists and redownload=False. Skipping download.")
        return

    # Authenticate with Google Drive (cached credentials)
    drive = authenticate_with_google_drive(client_secret_pth, credentials_pth)

    # If a single file is requested
    if file_spec:
        # Accept direct file id or path 'root/path/to/file'
        if '/' not in file_spec:
            download_single_file_by_id(file_spec, target_dir, drive)
            print("Download completed.")
            return
        # Resolve path to file by traversing parent dirs then matching file title
        parent_path, file_name = os.path.split(file_spec)
        parent_id = resolve_remote_folder_spec_to_id(drive, parent_path)
        query = f"'{parent_id}' in parents and trashed=false"
        for child in drive.ListFile({'q': query}).GetList():
            if child.get('mimeType') != 'application/vnd.google-apps.folder' and child.get('title') == file_name:
                download_one_file(child, target_dir)
                print("Download completed.")
                return
        raise FileNotFoundError(f"File not found at path: {file_spec}")

    # Resolve path-like folder spec to id; do not create if missing
    resolved_folder_id = resolve_remote_folder_spec_to_id(drive, folder_id)

    # Perform folder download
    download_files_in_folder(resolved_folder_id, target_dir, drive)
    print("Download completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from Google Drive to a local directory.")
    parser.add_argument("--folder_id", default="13xmw-8nSh8fHeJEoUaEi3DUA7-KmRThX",
                        help="Drive folder ID or path 'root/path/to/dir' (ignored if --file is set).")
    parser.add_argument("--file", default=None,
                        help="Single file to download: file ID or path 'root/path/to/file'.")
    parser.add_argument("--target_dir", default="../data/omaps2",
                        help="Local directory to store the downloaded files (default: ../data/omaps2).")
    parser.add_argument("--redownload", action="store_true", default=False,
                        help="If set to True, re-download files even if target_dir already exists (default: False).")
    parser.add_argument("--client_secret_pth", default="./client_secrets.json",
                        help="Path to the client_secrets.json file for Google API authentication (default: ./client_secrets.json).")

    args = parser.parse_args()
    main(args.folder_id, args.target_dir, args.redownload, args.client_secret_pth, None, args.file)
