#!/usr/bin/env bash
set -euo pipefail

version="${1:?usage: scripts/build-release-archive.sh <version> [os] [arch]}"
target_os="${2:-$(uname -s)}"
target_arch="${3:-$(uname -m)}"

case "${target_os}" in
  Darwin|darwin) target_os="darwin" ;;
  Linux|linux) target_os="linux" ;;
  *) echo "unsupported target os: ${target_os}" >&2; exit 1 ;;
esac

case "${target_arch}" in
  arm64|aarch64) target_arch="arm64" ;;
  amd64|x86_64) target_arch="amd64" ;;
  *) echo "unsupported target arch: ${target_arch}" >&2; exit 1 ;;
esac

host_os="$(uname -s)"
host_arch="$(uname -m)"
case "${host_os}" in
  Darwin|darwin) host_os="darwin" ;;
  Linux|linux) host_os="linux" ;;
esac
case "${host_arch}" in
  arm64|aarch64) host_arch="arm64" ;;
  amd64|x86_64) host_arch="amd64" ;;
esac

if [ "${host_os}" != "${target_os}" ] || [ "${host_arch}" != "${target_arch}" ]; then
  echo "PyInstaller builds must run on the target platform (${target_os}/${target_arch}); got ${host_os}/${host_arch}" >&2
  exit 1
fi

project="agent-insights"
binary="agent-insights"
archive="${project}_${version}_${target_os}_${target_arch}.tar.gz"

rm -rf build dist/pyinstaller dist/release
mkdir -p dist/release

python -m PyInstaller \
  --clean \
  --onefile \
  --name "${binary}" \
  --paths src \
  --specpath build/pyinstaller-spec \
  --distpath dist/pyinstaller \
  packaging/pyinstaller/agent-insights.py

dist_binary="dist/pyinstaller/${binary}"
version_output="$("${dist_binary}" --version)"
expected_version="${binary} v${version}"
if [ "${version_output}" != "${expected_version}" ]; then
  echo "version mismatch: got '${version_output}', want '${expected_version}'" >&2
  exit 1
fi
printf '%s\n' "${version_output}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

cp "${dist_binary}" "${tmpdir}/${binary}"
chmod 0755 "${tmpdir}/${binary}"
tar -C "${tmpdir}" -czf "dist/release/${archive}" "${binary}"

if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "dist/release/${archive}" > "dist/release/${archive}.sha256"
else
  sha256sum "dist/release/${archive}" > "dist/release/${archive}.sha256"
fi
