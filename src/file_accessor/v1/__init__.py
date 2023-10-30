# from enum import Enum


# class Storage(Enum):
#     LOCAL = 1
#     MINIO = 2


# class S3:
#     def __init__(self, storage: Storage) -> None:
#         self.storage = storage

#         if self.storage == Storage.MINIO:
#             pass
import boto3

if __name__ == "__main__":
    MINIO_REGION = "us-east-1"
    API_URL = "http://10.10.30.41:32295"
    MINIO_ID = "ml-data-storage"
    MINIO_PW = "surro0701!"

    s3 = boto3.client(
        "s3",
        endpoint_url=API_URL,
        aws_access_key_id=MINIO_ID,
        aws_secret_access_key=MINIO_PW,
        region_name=MINIO_REGION,
        use_ssl=False,
    )

    pass
