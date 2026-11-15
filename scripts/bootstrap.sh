#!/usr/bin/env bash
set -euo pipefail

project_dir=/root/autodl-tmp/clip-exp
python_bin="$project_dir/env/bin/python"
constraints_file="$project_dir/scripts/constraints.txt"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1
export OUTDATED_IGNORE=1
package_index=https://pypi.tuna.tsinghua.edu.cn/simple

exec >> "$project_dir/records/bootstrap.log" 2>&1
trap 'result=$?; printf "%s\n" "$result" > "$project_dir/records/bootstrap.exit"' EXIT

date -Iseconds
"$python_bin" --version
"$python_bin" -m pip install --index-url "$package_index" \
    pip==24.3.1 setuptools==69.5.1 wheel==0.45.1
"$python_bin" -m pip install --index-url "$package_index" \
    --constraint "$constraints_file" numpy pillow
"$python_bin" -m pip install --index-url "$package_index" \
    --constraint "$constraints_file" torch==2.4.1 torchvision==0.19.1
"$python_bin" -m pip install --index-url "$package_index" \
    --constraint "$constraints_file" scipy scikit-learn pandas matplotlib
sed -i 's/^tb-nightly$/tensorboard==2.18.0/' "$project_dir/code/Dassl.pytorch/requirements.txt"
git -C "$project_dir/code/Dassl.pytorch" diff -- requirements.txt \
    > "$project_dir/records/dassl_dependency.patch"
"$python_bin" -m pip install --index-url "$package_index" \
    --constraint "$constraints_file" --no-build-isolation \
    --editable "$project_dir/code/Dassl.pytorch"
"$python_bin" -m pip install --index-url "$package_index" \
    --constraint "$constraints_file" \
    --requirement "$project_dir/code/PromptSRC/requirements.txt" \
    --requirement "$project_dir/code/TCPT/requirements.txt"

"$python_bin" -m pip check
"$python_bin" -m pip freeze > "$project_dir/records/pip_freeze.txt"
CONDA_PKGS_DIRS="$project_dir/.cache/conda" \
    conda list --prefix "$project_dir/env" --explicit > "$project_dir/records/conda_explicit.txt"

cd "$project_dir/code/PromptSRC"
"$python_bin" train.py --help > "$project_dir/records/promptsrc_help.txt"
cd "$project_dir/code/TCPT"
"$python_bin" train.py --help > "$project_dir/records/tcpt_help.txt"
date -Iseconds
printf 'Environment installation and both training entry-point imports passed.\n'
