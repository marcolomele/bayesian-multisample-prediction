import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config


class HumanMicrobiomeDataset:
    """
    Download and manage the Human Microbiome Project dataset from AWS S3.
    
    The dataset is publicly available at s3://human-microbiome-project/
    No AWS credentials are required.
    """
    
    def __init__(self, download_dir="./data/microbiome", download_all=False):
        """
        Initialize the Human Microbiome Project dataset downloader.
        
        Args:
            download_dir (str): Directory to download the dataset to
            download_all (bool): If True, download all files. If False, only list available files.
        """
        self.bucket_name = "human-microbiome-project"
        self.region = "us-west-2"
        self.download_dir = download_dir
        
        # Configure boto3 to use unsigned requests (no AWS credentials needed)
        self.s3_client = boto3.client(
            's3',
            region_name=self.region,
            config=Config(signature_version=UNSIGNED)
        )
        
        # Create download directory if it doesn't exist
        os.makedirs(self.download_dir, exist_ok=True)
        
        if download_all:
            self.download_all()
    
    def list_files(self, prefix=""):
        """
        List all files in the S3 bucket.
        
        Args:
            prefix (str): S3 prefix to filter files (e.g., "16s/" or "wgs/")
        
        Returns:
            list: List of file keys in the bucket
        """
        files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    files.append(obj['Key'])
        
        return files
    
    def download_file(self, s3_key, local_path=None):
        """
        Download a single file from S3.
        
        Args:
            s3_key (str): S3 key (path) of the file to download
            local_path (str): Local path to save the file. If None, uses s3_key as filename.
        
        Returns:
            str: Path to the downloaded file
        """
        if local_path is None:
            # Create local path preserving directory structure
            local_path = os.path.join(self.download_dir, s3_key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        print(f"Downloading {s3_key}...")
        self.s3_client.download_file(self.bucket_name, s3_key, local_path)
        print(f"Downloaded to {local_path}")
        
        return local_path
    
    def download_all(self, prefix=""):
        """
        Download all files from the S3 bucket.
        
        Args:
            prefix (str): S3 prefix to filter files (e.g., "16s/" or "wgs/")
        
        Returns:
            list: List of downloaded file paths
        """
        files = self.list_files(prefix)
        downloaded = []
        
        print(f"Found {len(files)} files to download")
        
        for s3_key in files:
            try:
                local_path = self.download_file(s3_key)
                downloaded.append(local_path)
            except Exception as e:
                print(f"Error downloading {s3_key}: {e}")
        
        return downloaded
    
    def download_by_type(self, data_type="16s"):
        """
        Download files by data type.
        
        Args:
            data_type (str): Type of data to download. Options: "16s", "wgs", "isolates"
        
        Returns:
            list: List of downloaded file paths
        """
        prefix_map = {
            "16s": "16s/",
            "wgs": "wgs/",
            "isolates": "isolates/"
        }
        
        if data_type not in prefix_map:
            raise ValueError(f"Unknown data type: {data_type}. Choose from {list(prefix_map.keys())}")
        
        return self.download_all(prefix=prefix_map[data_type])
    
    def get_bucket_info(self):
        """
        Get information about the S3 bucket structure.
        
        Returns:
            dict: Information about available prefixes and file counts
        """
        files = self.list_files()
        
        # Group by prefix
        prefixes = {}
        for file_key in files:
            parts = file_key.split('/')
            if len(parts) > 1:
                prefix = parts[0] + '/'
                prefixes[prefix] = prefixes.get(prefix, 0) + 1
            else:
                prefixes['root/'] = prefixes.get('root/', 0) + 1
        
        return {
            'total_files': len(files),
            'prefixes': prefixes,
            'sample_files': files[:10] if len(files) > 10 else files
        }


if __name__ == "__main__":
    # Example usage
    dataset = HumanMicrobiomeDataset(download_dir="./data/microbiome")
    
    # First, explore what's available
    print("Exploring bucket structure...")
    info = dataset.get_bucket_info()
    print(f"Total files: {info['total_files']}")
    print(f"Available prefixes: {info['prefixes']}")
    print(f"\nSample files:")
    for f in info['sample_files']:
        print(f"  - {f}")
    
    # Uncomment to download specific data types:
    # dataset.download_by_type("16s")  # Download 16S marker gene data
    # dataset.download_by_type("wgs")  # Download whole genome shotgun data
    # dataset.download_by_type("isolates")  # Download isolate genomes