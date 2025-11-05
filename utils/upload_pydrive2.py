import os
import argparse
import multiprocessing
from typing import Optional
import json

from joblib import Parallel, delayed
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from tqdm import tqdm


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def authenticate_with_google_drive(
    client_secret_path: str,
    credentials_path: Optional[str] = None,
) -> GoogleDrive:
    """Authenticate with Google Drive using cached credentials when available.

    If cached credentials are not found or are invalid/expired, performs interactive
    auth and saves refreshed credentials to ``credentials_path`` for future runs.
    """
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

    # Try load cached credentials
    if os.path.exists(credentials_path):
        gauth.LoadCredentialsFile(credentials_path)

    # Ensure offline access and manage saving ourselves to avoid backend config error
    gauth.settings['get_refresh_token'] = True
    gauth.settings['save_credentials'] = False  # we'll call SaveCredentialsFile() explicitly

    has_refresh = (
        getattr(gauth, 'credentials', None) is not None
        and getattr(gauth.credentials, 'refresh_token', None)
    )

    if gauth.credentials is None:
        gauth.CommandLineAuth()
    elif not has_refresh:
        # Existing token without refresh token: re-consent to obtain one
        gauth.CommandLineAuth()
    elif getattr(gauth, "access_token_expired", False):
        gauth.Refresh()
    else:
        gauth.Authorize()

    # Persist credentials for reuse
    _ensure_dir(credentials_path)
    gauth.SaveCredentialsFile(credentials_path)

    return GoogleDrive(gauth)


def resolve_remote_folder_spec_to_id(
    drive: GoogleDrive,
    folder_spec: str,
    create_missing: bool = False,
) -> str:
    """Resolve a remote folder spec to a folder id.

    Accepts either a plain folder id or a path like 'root/path/to/dir'.
    If ``create_missing`` is True, missing path segments will be created.
    """
    # Treat as plain id if no path separators and not equal to 'root'
    if '/' not in folder_spec and folder_spec != 'root':
        return folder_spec

    # Normalize path and start from root
    segments = [seg for seg in folder_spec.strip('/').split('/') if seg and seg != '.']
    if segments and segments[0] == 'root':
        segments = segments[1:]
    current_parent_id = 'root'

    for segment in segments:
        existing = find_child_by_title(
            drive,
            current_parent_id,
            segment,
            mime_type='application/vnd.google-apps.folder',
        )
        if existing is not None:
            current_parent_id = existing['id']
            continue

        if create_missing:
            current_parent_id = ensure_remote_folder(drive, current_parent_id, segment)
        else:
            raise FileNotFoundError(
                f"Remote path segment not found under parent {current_parent_id}: {segment}"
            )

    return current_parent_id


def list_children_in_parent(drive: GoogleDrive, parent_id: str, mime_type: Optional[str] = None):
    """List all non-trashed children under a parent, optionally filtered by mimeType."""
    query = f"'{parent_id}' in parents and trashed=false"
    if mime_type is not None:
        query += f" and mimeType='{mime_type}'"
    return drive.ListFile({'q': query}).GetList()


def find_child_by_title(
    drive: GoogleDrive,
    parent_id: str,
    title: str,
    mime_type: Optional[str] = None,
):
    """
    Find the first child under parent with exact title and optional mime type.
    Returns the GoogleDriveFile object or None if not found.
    """
    # Narrow on parent and trashed, then filter by title (avoids query escaping issues)
    children = list_children_in_parent(drive, parent_id, mime_type=mime_type)
    for child in children:
        if child.get('title') == title:
            return child
    return None


def ensure_remote_folder(drive: GoogleDrive, parent_id: str, folder_name: str) -> str:
    """
    Ensure a folder with folder_name exists under parent_id. Return its id.
    """
    existing = find_child_by_title(
        drive,
        parent_id,
        folder_name,
        mime_type='application/vnd.google-apps.folder',
    )
    if existing is not None:
        return existing['id']

    folder = drive.CreateFile(
        {
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [{'id': parent_id}],
        }
    )
    folder.Upload()
    return folder['id']


def upload_one_file(
    local_file_path: str,
    parent_id: str,
    drive: GoogleDrive,
    overwrite: bool = False,
):
    """Upload a single file to the specified parent folder in Drive."""
    title = os.path.basename(local_file_path)

    existing = find_child_by_title(drive, parent_id, title)
    if existing is not None and not overwrite:
        print(f"[Skip] {title} already exists in parent {parent_id}")
        return existing['id']

    if existing is not None and overwrite:
        print(f"[Update] {title} -> parent {parent_id}")
        existing.SetContentFile(local_file_path)
        existing.Upload()
        return existing['id']

    print(f"[Upload] {title} -> parent {parent_id}")
    remote_file = drive.CreateFile({'title': title, 'parents': [{'id': parent_id}]})
    remote_file.SetContentFile(local_file_path)
    remote_file.Upload()
    return remote_file['id']


def upload_directory(
    local_directory: str,
    parent_id: str,
    drive: GoogleDrive,
    overwrite: bool = False,
) -> str:
    """
    Recursively upload a local directory to Google Drive under parent_id.
    Returns the id of the top-level created (or found) remote folder.
    """
    abs_local = os.path.abspath(local_directory)
    folder_name = os.path.basename(abs_local)
    remote_folder_id = ensure_remote_folder(drive, parent_id, folder_name)

    # Collect immediate children
    files_in_dir = []
    subdirs_in_dir = []
    with os.scandir(abs_local) as it:
        for entry in it:
            if entry.is_file():
                files_in_dir.append(entry.path)
            elif entry.is_dir():
                subdirs_in_dir.append(entry.path)

    # Upload files in this directory in parallel
    if files_in_dir:
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores, backend='threading')(
            delayed(upload_one_file)(file_path, remote_folder_id, drive, overwrite)
            for file_path in tqdm(files_in_dir)
        )

    # Recurse into subdirectories
    for subdir in subdirs_in_dir:
        upload_directory(subdir, remote_folder_id, drive, overwrite)

    return remote_folder_id


def main(
    local_dir: Optional[str],
    local_file: Optional[str],
    client_secret_pth: str,
    parent_spec: str,
    overwrite: bool,
):
    if local_file:
        if not os.path.isfile(local_file):
            raise FileNotFoundError(f"Local file does not exist: {local_file}")
    else:
        if not local_dir or not os.path.isdir(local_dir):
            raise FileNotFoundError(f"Local directory does not exist: {local_dir}")

    drive = authenticate_with_google_drive(client_secret_pth, None)

    # Allow 'root/path/to/dir' and auto-create missing segments
    parent_id = resolve_remote_folder_spec_to_id(
        drive, parent_spec, create_missing=True
    )

    if local_file:
        file_id = upload_one_file(local_file, parent_id, drive, overwrite=overwrite)
        print(f"Upload completed. Remote file id: {file_id}")
        return

    top_folder_id = upload_directory(local_dir, parent_id, drive, overwrite=overwrite)
    print(f"Upload completed. Remote folder id: {top_folder_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Recursively upload a local folder to Google Drive, preserving folder structure."
        )
    )
    parser.add_argument(
        "--local_dir",
        default="../data/omaps2",
        help="Path to the local directory to upload (default: ../data/omaps2).",
    )
    parser.add_argument(
        "--local_file",
        default=None,
        help=(
            "Path to a single local file to upload. If set, overrides --local_dir."
        ),
    )
    parser.add_argument(
        "--client_secret_pth",
        default="./client_secrets.json",
        help=(
            "Path to client_secrets.json for Google API auth (default: ./client_secrets.json)."
        ),
    )
    parser.add_argument(
        "--parent_id",
        default="root",
        help=(
            "Destination folder id or path like 'root/path/to/dir' (default: root)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help=(
            "If set, overwrite existing files with the same name instead of skipping."
        ),
    )

    args = parser.parse_args()
    main(
        args.local_dir,
        args.local_file,
        args.client_secret_pth,
        args.parent_id,
        args.overwrite,
    )


