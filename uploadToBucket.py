from google.cloud import storage


import argparse




def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)

    for blob in blobs:
        print(blob.name)

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-fn", "--Filename", help = "Path/Filename of source file")
parser.add_argument("-dest", "--Destination", help = "Path/Filename of destination file")
parser.add_argument("-list", "--List", help = "List files in bucket")
# Read arguments from command line
args = parser.parse_args()

if args.Filename:
    upload_blob("aiml-textgen-tts",args.Filename,args.Destination)

if args.List:
    list_blobs("aiml-textgen-tts")

