import boto3
from io import BytesIO
from typing import Optional, Dict, Any, List


class S3yandex():
    def __init__(self, endpoint_url, aws_access_key_id, aws_secret_access_key):
        self.endpoint_url = endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key,
        self.session = boto3.session.Session()

    def get_client(self):
        return self.session.client(service_name='s3', endpoint_url=self.endpoint_url,
                                   aws_access_key_id=self.aws_access_key_id,
                                   aws_secret_access_key=self.aws_secret_access_key)

    def load_s3_files(self, bucket: str, prefix: str, suffix: str) -> List[str]:
        """List files in a given S3 bucket with a specified prefix and suffix."""
        try:
            s3_client = self.get_client()
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            files = [content['Key'] for content in response.get('Contents', []) if content['Key'].endswith(suffix)]
            if not files:
                print(f"No files found in bucket {bucket} with prefix {prefix} and suffix {suffix}")
            else:
                print(f"Files found in bucket {bucket} with prefix {prefix} and suffix {suffix}: {files}")
            return files
        except Exception as e:
            print(f"Error listing files in bucket {bucket} with prefix {prefix} and suffix {suffix}: {e}")
            return []