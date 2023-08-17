import git
import click


def get_git_root():
    repo = git.Repo('.', search_parent_directories=True)
    return repo.working_tree_dir


@click.command()
@click.option(
    "--dict", "-d", "info_url",
    type=(str, str),
    multiple=True,
    default=[("base_url", "https://webrobots.io"),
             ("extension", "kickstarter-datasets"),
             ("data_format", "Kickstarter_2023.*\\.zip")])
@click.argument(
    "s3_bucket_name",
    type=str,
    required=False,
    default="kickstarter-bucket")
@click.option(
    "--dict", "-d", "info_data",
    type=(str, str),
    multiple=True,
    default=[("fnames", None),
             ("path_local_in", None),
             ("path_local_out", f"{get_git_root()}/data/raw"),
             ("path_s3_in", None),
             ("path_s3_out", "data/raw"),
             ("prefix_name", "kickstarter")])
@click.pass_context
def gather_downloader(ctx, info_url,
                      s3_bucket_name,
                      info_data):
    return ctx.params


@click.command()
@click.argument(
    "s3_bucket_name",
    type=str,
    required=False,
    default="kickstarter-bucket")
@click.option(
    "--dict", "-d", "info_data",
    type=(str, str),
    multiple=True,
    default=[("fnames", None),
             ("path_local_in", f"{get_git_root()}/data/raw"),
             ("path_local_out", f"{get_git_root()}/data/interim"),
             ("path_s3_in", "data/raw"),
             ("path_s3_out", "data/interim"),
             ("prefix_name", "kickstarter")])
@click.option(
    "--dict", "-d", "info_pipe",
    type=(str, str),
    multiple=True,
    default=[("fnames", None),
             ("path_local_in", None),
             ("path_local_out", f"{get_git_root()}/models/interim"),
             ("path_s3_in", None),
             ("path_s3_out", "models/interim"),
             ("prefix_name", "model")])
@click.argument(
    "test_size",
    type=float,
    required=False,
    default=0.15)
@click.argument(
    "seed",
    type=int,
    required=False,
    default=1234)
@click.pass_context
def gather_cleaner(ctx, s3_bucket_name,
                   info_data, info_pipe,
                   test_size, seed):
    return ctx.params


@click.command()
@click.argument(
    "s3_bucket_name",
    type=str,
    required=False,
    default="kickstarter-bucket")
@click.option(
    "--dict", "-d", "info_data",
    type=(str, str),
    multiple=True,
    default=[("fnames", None),
             ("path_local_in", f"{get_git_root()}/data/interim"),
             ("path_local_out", f"{get_git_root()}/data/processed"),
             ("path_s3_in", "data/interim"),
             ("path_s3_out", "data/processed"),
             ("prefix_name", "kickstarter")])
@click.option(
    "--dict", "-d", "info_pipe",
    type=(str, str),
    multiple=True,
    default=[("fnames", None),
             ("path_local_in", None),
             ("path_local_out", f"{get_git_root()}/models/processed"),
             ("path_s3_in", None),
             ("path_s3_out", "models/processed"),
             ("prefix_name", "model")])
@click.pass_context
def gather_build_features(ctx, s3_bucket_name,
                          info_data, info_pipe):
    return ctx.params


@click.command()
@click.argument(
    "s3_bucket_name",
    type=str,
    required=False,
    default="kickstarter-bucket")
@click.option(
    "--dict", "-d", "info_data",
    type=(str, str),
    multiple=True,
    default=[("fnames", None),
             ("path_local_in", f"{get_git_root()}/data/processed"),
             ("path_local_out", None),
             ("path_s3_in", f"{get_git_root()}/data/processed"),
             ("path_s3_out", None),
             ("prefix_name", "kickstarter")])
@click.option(
    "--dict", "-d", "info_pipe",
    type=(str, str),
    multiple=True,
    default=[("fnames", None),
             ("path_local_in", None),
             ("path_local_out", f"{get_git_root()}/models/trained"),
             ("path_s3_in", None),
             ("path_s3_out", "models/trained"),
             ("prefix_name", "model")])
@click.argument(
    "sweep_config",
    type=click.Path(exists=True),
    required=False,
    default=f"{get_git_root()}/src/models/sweep_config.yaml",
)
@click.argument(
    "n_sweeps",
    type=int,
    required=False,
    default=5)
@click.argument(
    "seed",
    type=int,
    required=False,
    default=1234)
@click.pass_context
def gather_train(ctx, s3_bucket_name,
                 info_data, info_pipe,
                 sweep_config,
                 n_sweeps, seed):
    return ctx.params


@click.command()
@click.argument(
    "s3_bucket_name",
    type=str,
    required=False,
    default="kickstarter-bucket")
@click.option(
    "--dict", "-d", "info_data",
    type=(str, str),
    multiple=True,
    default=[("fnames", None),
             ("path_local_in", f"{get_git_root()}/data/processed"),
             ("path_local_out", None),
             ("path_s3_in", f"{get_git_root()}/data/processed"),
             ("path_s3_out", None),
             ("prefix_name", "kickstarter")])
@click.option(
    "--dict", "-d", "info_pipe",
    type=(str, str),
    multiple=True,
    default=[("fnames", None),
             ("path_local_in", f"{get_git_root()}/models/trained"),
             ("path_local_out", f"{get_git_root()}/models/registry"),
             ("path_s3_in", None),
             ("path_s3_out", "models/registry"),
             ("prefix_name", "model")])
@click.argument(
    "n_best",
    type=int,
    required=False,
    default=3)
@click.pass_context
def gather_register_model(ctx, s3_bucket_name,
                          info_data, info_pipe,
                          n_best):
    return ctx.params

