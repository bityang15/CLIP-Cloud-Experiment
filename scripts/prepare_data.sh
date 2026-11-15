#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/clip-exp
python_bin="$root/env/bin/python"
mkdir -p "$root/downloads" "$root/datasets/oxford_flowers" "$root/weights" "$root/records/B_stage1"
exec >> "$root/records/B_stage1/download.log" 2>&1
trap 'result=$?; printf "%s\n" "$result" > "$root/records/B_stage1/download.exit"' EXIT

date -Iseconds
"$python_bin" "$root/scripts/fetch_range.py" \
    'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz' \
    "$root/downloads/dtd-r1.0.1.tar.gz" fff73e5086ae6bdbea199a49dfb8a4c1
"$python_bin" "$root/scripts/fetch_range.py" \
    'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz' \
    "$root/downloads/102flowers.tgz" 52808999861908f626f3c1f4e79d11fa
curl --fail --location --retry 3 --connect-timeout 30 \
    'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat' \
    --output "$root/datasets/oxford_flowers/imagelabels.mat"
tar -xzf "$root/downloads/dtd-r1.0.1.tar.gz" -C "$root/datasets"
tar -xzf "$root/downloads/102flowers.tgz" -C "$root/datasets/oxford_flowers"
"$python_bin" "$root/scripts/fetch_range.py" \
    'https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt' \
    "$root/weights/ViT-B-16.pt" 5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f
if [ ! -s "$root/datasets/dtd/split_zhou_DescribableTextures.json" ]; then
    "$python_bin" -m gdown --no-cookies --timeout 30 --retries 1 '1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x' -O "$root/datasets/dtd/split_zhou_DescribableTextures.json"
fi
if [ ! -s "$root/datasets/oxford_flowers/cat_to_name.json" ]; then
    "$python_bin" -m gdown --no-cookies --timeout 30 --retries 1 '1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0' -O "$root/datasets/oxford_flowers/cat_to_name.json"
fi
if [ ! -s "$root/datasets/oxford_flowers/split_zhou_OxfordFlowers.json" ]; then
    "$python_bin" -m gdown --no-cookies --timeout 30 --retries 1 '1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT' -O "$root/datasets/oxford_flowers/split_zhou_OxfordFlowers.json"
fi
printf '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f  %s\n' "$root/weights/ViT-B-16.pt" | sha256sum -c -
sha256sum "$root/weights/ViT-B-16.pt" > "$root/records/weight_sha256.txt"
mkdir -p /root/.cache/clip
if [ ! -e /root/.cache/clip/ViT-B-16.pt ]; then
    ln -s "$root/weights/ViT-B-16.pt" /root/.cache/clip/ViT-B-16.pt
fi
sha256sum /root/.cache/clip/ViT-B-16.pt
date -Iseconds
