# ML Development Infrastructure

This repository is meant to give hobbyists a professional-grade ML development system. It organizes your projects, uses Weights and Biases to track experiments, and launches ML jobs to clusters. It also includes a development Dockerfile.

## Prerequisites

**Docker**: You should have Docker installed on your system.

## Usage

You'll want to clone or download the repo.

Next you should start the development docker container. For convenience, a bash script will install some aliases to start, enter, and remove the container. Run the `setup.sh` script located in `scripts`, like:

```
./scripts/setup.sh
```

Now, running `dstart` will create the development container. After that, `dinto` will enter it. You can remove the container with `dremove` and restart with `drestart`.

If you'd prefer not to install these aliases, just run the scripts they would map to directly in order to build and enter the container:

```
./scripts/docker/start.sh && ./scripts/docker/into.sh
```

### Not on git usage

It is recommended to use git inside the docker container. If you're on a Mac, you might be using the osxkeychain credential helper, which is awkward. There are a number of other failure modes as well. However, if you switch to SSH URLs (recommended) using `git remote set-url origin git@github.com:gkamer8/ml-infra.git`, your problems should go away. The `start.sh` script mounts your SSH credentials.
