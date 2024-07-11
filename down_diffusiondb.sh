#! /bin/bash
#!/bin/bash

# Base URL
base_url="https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part-"

# Loop from 1 to 2000
for i in $(seq 1 2000); do
  # Format the number with leading zeros to 6 digits
  part_id=$(printf "%06d" $i)
  
  # Construct the full URL
  url="${base_url}${part_id}.zip"
  # Download the file
  sudo aria2c -x 16 -s 16 -d /mnt/8T -o "part-${part_id}.zip" "$url"
done

