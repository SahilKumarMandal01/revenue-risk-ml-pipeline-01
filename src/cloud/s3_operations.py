import subprocess
import sys
from pathlib import Path
from typing import List, Union

from src.custom_exception import CustomException
from src.custom_logging import logging


PathLike = Union[str, Path]


class S3Sync:
    """
    Production-grade AWS S3 sync utility.

    Responsibilities:
    - Normalize Path / str inputs
    - Execute AWS CLI safely
    - Provide structured logging
    - Surface subprocess errors clearly
    """

    # --------------------------------------------------
    # INTERNAL COMMAND RUNNER
    # --------------------------------------------------

    def _run(self, command: List[PathLike]) -> None:
        """
        Execute AWS CLI command safely.

        Converts all command arguments to string
        to prevent PosixPath-related failures.
        """
        try:
            normalized_command: List[str] = [str(arg) for arg in command]

            logging.info("[S3] %s", " ".join(normalized_command))

            result = subprocess.run(
                normalized_command,
                check=True,
                capture_output=True,
                text=True,
            )

            if result.stdout:
                logging.info(result.stdout.strip())

            if result.stderr:
                logging.debug(result.stderr.strip())

        except subprocess.CalledProcessError as exc:
            logging.error("[S3 ERROR]")
            logging.error(exc.stderr)
            raise CustomException(exc, sys) from exc

        except Exception as exc:
            logging.exception("Unexpected error during S3 command execution.")
            raise CustomException(exc, sys) from exc

    # --------------------------------------------------
    # DOWNLOAD SINGLE FILE
    # --------------------------------------------------

    def download_file(self, s3_uri: str, local_path: PathLike) -> None:
        """
        Download a single file from S3.

        Args:
            s3_uri (str): S3 URI (e.g., s3://bucket/key)
            local_path (PathLike): Local file destination
        """
        try:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            self._run(
                [
                    "aws",
                    "s3",
                    "cp",
                    s3_uri,
                    local_path,
                ]
            )

            logging.info(
                "Downloaded file from %s to %s", s3_uri, str(local_path)
            )

        except Exception as exc:
            logging.exception("Failed to download file from S3.")
            raise CustomException(exc, sys) from exc

    # --------------------------------------------------
    # UPLOAD SINGLE FILE
    # --------------------------------------------------

    def upload_file(self, local_path: PathLike, s3_uri: str) -> None:
        """
        Upload a single file to S3.

        Args:
            local_path (PathLike): Local file path
            s3_uri (str): Destination S3 URI (e.g., s3://bucket/key)
        """
        try:
            local_path = Path(local_path)

            if not local_path.exists():
                raise FileNotFoundError(f"File not found: {local_path}")

            self._run(
                [
                    "aws",
                    "s3",
                    "cp",
                    local_path,
                    s3_uri,
                ]
            )

            logging.info(
                "Uploaded file from %s to %s", str(local_path), s3_uri
            )

        except Exception as exc:
            logging.exception("Failed to upload file to S3.")
            raise CustomException(exc, sys) from exc

    # --------------------------------------------------
    # SYNC LOCAL → S3
    # --------------------------------------------------

    def sync_folder_to_s3(
        self,
        folder: PathLike,
        aws_bucket_url: PathLike,
    ) -> None:
        """
        Sync local folder to S3 bucket.

        Args:
            folder (PathLike): Local folder path
            aws_bucket_url (PathLike): S3 bucket URI
        """
        try:
            folder = Path(folder)

            if not folder.exists():
                raise FileNotFoundError(f"Folder not found: {folder}")

            self._run(
                [
                    "aws",
                    "s3",
                    "sync",
                    folder,
                    aws_bucket_url,
                ]
            )

            logging.info(
                "Synced local folder %s to S3 %s",
                str(folder),
                str(aws_bucket_url),
            )

        except Exception as exc:
            logging.exception("Failed to sync folder to S3.")
            raise CustomException(exc, sys) from exc

    # --------------------------------------------------
    # SYNC S3 → LOCAL
    # --------------------------------------------------

    def sync_folder_from_s3(
        self,
        folder: PathLike,
        aws_bucket_url: PathLike,
    ) -> None:
        """
        Sync S3 folder to local directory.

        Args:
            folder (PathLike): Local folder path
            aws_bucket_url (PathLike): S3 bucket URI
        """
        try:
            folder = Path(folder)
            folder.mkdir(parents=True, exist_ok=True)

            self._run(
                [
                    "aws",
                    "s3",
                    "sync",
                    aws_bucket_url,
                    folder,
                ]
            )

            logging.info(
                "Synced S3 %s to local folder %s",
                str(aws_bucket_url),
                str(folder),
            )

        except Exception as exc:
            logging.exception("Failed to sync folder from S3.")
            raise CustomException(exc, sys) from exc