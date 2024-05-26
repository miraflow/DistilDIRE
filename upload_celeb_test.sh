# !/bin/bash
cd datasets

aws s3 sync distil-test-dalle2-celebahq/ s3://truemedia-dataset/distil-dire-dataset/celebahq-test-dalle2
aws s3 sync distil-test-if-celebahq/ s3://truemedia-dataset/distil-dire-dataset/celebahq-test-if
aws s3 sync distil-test-midjourney-celebahq/ s3://truemedia-dataset/distil-dire-dataset/celebahq-test-midjourney
aws s3 sync distil-test-sdv2-celebahq/ s3://truemedia-dataset/distil-dire-dataset/celebahq-test-sdv2