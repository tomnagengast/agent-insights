# Release

Releases are tag driven. A pushed semver tag builds standalone macOS
`agent-insights` archives, creates a GitHub release with checksums, and publishes
the Homebrew cask to `tomnagengast/homebrew-tap`.

## Release setup

The Homebrew tap lives at `tomnagengast/homebrew-tap`. Keep
`HOMEBREW_TAP_GITHUB_TOKEN` set in the `tomnagengast/agent-insights` repository
secrets. The token needs contents write access to that tap repository.

The release workflow uses the repository `GITHUB_TOKEN` for the GitHub release
itself.

## Local checks

Run the standard checks:

```sh
python -m compileall src
python -m pip install -e '.[release]'
scripts/build-release-archive.sh 0.1.2
```

Verify the generated executable:

```sh
dist/pyinstaller/agent-insights --version
tar -tzf dist/release/agent-insights_0.1.2_darwin_arm64.tar.gz
```

## Cut a release

Merge the release commit to `main`, then create and push the next semver tag:

```sh
git tag -a v0.1.2 -m "v0.1.2"
git push origin v0.1.2
```

The `release` workflow only publishes from tag refs matching `v*.*.*`.

## Homebrew

The workflow publishes a Homebrew cask. Install the latest release with:

```sh
brew tap tomnagengast/tap
brew install --cask tomnagengast/tap/agent-insights-cli
```
