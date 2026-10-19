# Reproducible gfx950 CI image

This directory defines the environment used by Avelang's self-hosted ROCm CI runner. It does not contain repository source, runner registration state, or credentials.

## Pinned inputs

- Base image: `docker.io/lmsysorg/sglang:v0.5.22-rocm10-mi35x` at digest `sha256:1438dd8b9000adaefb5828edb1cc5713992815d4296f4ad1a1e98dee4e9c7e5f`
- ROCm LLVM: commit `8f497e0992fb7513f7f78a6f6b6f1056c375e961`
- LLVM configuration: Clang, MLIR, and LLD; X86 and AMDGPU targets; RTTI enabled; static LLVM component libraries
- GitHub Actions Runner: `2.337.0`, verified by the upstream release SHA-256
- Apt and Python packages: exact versions in `apt-packages.txt` and `requirements-ci.txt`

The image is tagged locally as `localhost/avelang-ci:rocm10-llvm23-gfx950-v1` by default.

## Build without sudo

Run from the repository checkout on the Podman host:

```bash
LLVM_BUILD_JOBS=32 ./ci/build-image.sh
```

An optional first argument overrides the image tag. The LLVM build is intentionally performed from the pinned source commit rather than copied from the mutable development container.

## Verify

The verification script starts a disposable GPU-enabled container, checks the pinned toolchain and runner, builds Avelang with Ninja, runs CTest, and runs `python -m pytest test/`:

```bash
AVE_OCI_RUNTIME=/data/home/xiaoyan/.local/crun-package/crun-local \
  ./ci/verify-image.sh
```

The custom runtime is needed on the current rootless Podman host so `--group-add keep-groups` preserves access to `/dev/kfd` and `/dev/dri`.

## Durable runner service

The pristine runner distribution is installed at `/opt/actions-runner`, while registration state and job workspaces remain under `~/dev/ave_action_runner`. Registration tokens and credentials are never baked into the image.

After the runner has been registered once, install and start the rootless user service:

```bash
./ci/install-runner-service.sh
```

The installer refuses to replace the container while a `Runner.Worker` process is active. It enables user lingering, installs `avelang-ci-runner.service`, and starts the runner in `ave-kda-mi355`. The service recreates the container from the pinned image after a failure, starts at boot without an interactive login, and stops the runner gracefully.

Inspect it with:

```bash
systemctl --user status avelang-ci-runner.service
journalctl --user -u avelang-ci-runner.service --since today
```

## Updating pins

Treat every pin change as an environment upgrade:

1. Update the base digest, source commit, runner checksum, or package manifests.
2. Increment the version in `image.env`.
3. Build the image from scratch when validating changed dependency layers.
4. Run `verify-image.sh` and require both CTest and pytest to pass before switching the registered runner.
