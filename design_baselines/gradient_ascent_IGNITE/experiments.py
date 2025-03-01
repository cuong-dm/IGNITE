from ray import tune
import click
import ray
import os


@click.group()
def cli():
    """A group of experiments for training Conservative Score Models
    and reproducing our ICLR 2021 results.
    """


#############


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-dkitty')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--epsilon', type=float, default=1e-3)
@click.option('--eta_lambda_', type=float, default=1e-3)
@click.option('--rho', type=float, default=0.05)
@click.option('--r', type=float, default=0.05)
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples, lambda_, epsilon, eta_lambda_, rho, r):
    """Evaluate Naive Gradient Ascent on DKittyMorphology-Exact-v0
    """

    # Final Version

    from design_baselines.gradient_ascent_IGNITE import gradient_ascent_IGNITE
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent_IGNITE, config={
        "lambda_": lambda_,
        "eta_lambda_": eta_lambda_,
        "epsilon": epsilon,
        "rho": rho,
        "r": r,
        "logging_dir": "data",
        "task": "DKittyMorphology-Exact-v0",
        "task_kwargs": {"relabel": False},
        "normalize_ys": True,
        "normalize_xs": True,
        "model_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "aggregation_method": 'mean',
        "solver_samples": 128, "do_evaluation": True,
        "solver_lr": 0.01,
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-ant')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--epsilon', type=float, default=1e-3)
@click.option('--eta_lambda_', type=float, default=1e-3)
@click.option('--rho', type=float, default=0.05)
@click.option('--r', type=float, default=0.05)
def ant(local_dir, cpus, gpus, num_parallel, num_samples, lambda_, epsilon, eta_lambda_, rho, r):
    """Evaluate Naive Gradient Ascent on AntMorphology-Exact-v0
    """

    # Final Version

    from design_baselines.gradient_ascent_IGNITE import gradient_ascent_IGNITE
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent_IGNITE, config={
        "lambda_": lambda_,
        "eta_lambda_": eta_lambda_,
        "epsilon": epsilon,
        "rho": rho,
        "r": r,
        "logging_dir": "data",
        "task": "AntMorphology-Exact-v0",
        "task_kwargs": {"relabel": False},
        "normalize_ys": True,
        "normalize_xs": True,
        "model_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "aggregation_method": 'mean',
        "solver_samples": 128, "do_evaluation": True,
        "solver_lr": 0.01,
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--epsilon', type=float, default=1e-3)
@click.option('--eta_lambda_', type=float, default=1e-3)
@click.option('--rho', type=float, default=0.05)
@click.option('--r', type=float, default=0.05)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples, lambda_, epsilon, eta_lambda_, rho, r):
    """Evaluate Naive Gradient Ascent on HopperController-Exact-v0
    """

    # Final Version

    from design_baselines.gradient_ascent_IGNITE import gradient_ascent_IGNITE
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent_IGNITE, config={
        "lambda_": lambda_,
        "eta_lambda_": eta_lambda_,
        "epsilon": epsilon,
        "rho": rho,
        "r": r,
        "logging_dir": "data",
        "task": "HopperController-Exact-v0",
        "task_kwargs": {"relabel": False},
        "normalize_ys": True,
        "normalize_xs": True,
        "model_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "aggregation_method": 'mean',
        "solver_samples": 128, "do_evaluation": True,
        "solver_lr": 0.01,
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-superconductor')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--epsilon', type=float, default=1e-3)
@click.option('--eta_lambda_', type=float, default=1e-3)
@click.option('--rho', type=float, default=0.05)
@click.option('--r', type=float, default=0.05)
@click.option('--oracle', type=str, default="RandomForest")
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples, lambda_, epsilon, eta_lambda_, rho, r, oracle):
    """Evaluate Naive Gradient Ascent on Superconductor-RandomForest-v0
    """

    # Final Version

    from design_baselines.gradient_ascent_IGNITE import gradient_ascent_IGNITE
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent_IGNITE, config={
        "lambda_": lambda_,
        "eta_lambda_": eta_lambda_,
        "epsilon": epsilon,
        "rho": rho,
        "r": r,
        "logging_dir": "data",
        "task": f"Superconductor-{oracle}-v0",
        "task_kwargs": {"relabel": False},
        "normalize_ys": True,
        "normalize_xs": True,
        "model_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 50,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "aggregation_method": 'mean',
        "solver_samples": 128, "do_evaluation": True,
        "solver_lr": 0.01,
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-chembl')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--epsilon', type=float, default=1e-3)
@click.option('--eta_lambda_', type=float, default=1e-3)
@click.option('--rho', type=float, default=0.05)
@click.option('--r', type=float, default=0.05)
@click.option('--assay-chembl-id', type=str, default='CHEMBL3885882')
@click.option('--standard-type', type=str, default='MCHC')
def chembl(local_dir, cpus, gpus, num_parallel, num_samples, lambda_, epsilon, eta_lambda_, rho, r,
           assay_chembl_id, standard_type):
    """Evaluate Naive Gradient Ascent on ChEMBL-ResNet-v0
    """

    # Final Version

    from design_baselines.gradient_ascent_IGNITE import gradient_ascent_IGNITE
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent_IGNITE, config={
        "lambda_": lambda_,
        "eta_lambda_": eta_lambda_,
        "epsilon": epsilon,
        "rho": rho,
        "r": r,
        "logging_dir": "data",
        "task": f"ChEMBL_{standard_type}_{assay_chembl_id}"
                f"_MorganFingerprint-RandomForest-v0",
        "task_kwargs": {"relabel": False,
                        "dataset_kwargs": dict(
                            assay_chembl_id=assay_chembl_id,
                            standard_type=standard_type)},
        "normalize_ys": True,
        "normalize_xs": True,
        "model_noise_std": 0.0,
        "val_size": 200,
        "use_vae": False,
        "vae_beta": 0.01,
        "vae_epochs": 50,
        "vae_batch_size": 128,
        "vae_hidden_size": 64,
        "vae_latent_size": 256,
        "vae_activation": "relu",
        "vae_kernel_size": 3,
        "vae_num_blocks": 3,
        "vae_lr": 0.0003,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "aggregation_method": 'mean',
        "solver_samples": 128, "do_evaluation": True,
        "solver_lr": 0.01,
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-gfp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--epsilon', type=float, default=1e-3)
@click.option('--eta_lambda_', type=float, default=1e-3)
@click.option('--rho', type=float, default=0.05)
@click.option('--r', type=float, default=0.05)
@click.option('--oracle', type=str, default="Transformer")
def gfp(local_dir, cpus, gpus, num_parallel, num_samples, lambda_, epsilon, eta_lambda_, rho, r, oracle):
    """Evaluate Naive Gradient Ascent on GFP-Transformer-v0
    """

    # Final Version

    from design_baselines.gradient_ascent_IGNITE import gradient_ascent_IGNITE
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent_IGNITE, config={
        "lambda_": lambda_,
        "eta_lambda_": eta_lambda_,
        "epsilon": epsilon,
        "rho": rho,
        "r": r,
        "logging_dir": "data",
        "task": f"GFP-{oracle}-v0",
        "task_kwargs": {"relabel": False},
        "normalize_ys": True,
        "normalize_xs": True,
        "model_noise_std": 0.0,
        "val_size": 200,
        "use_vae": False,
        "vae_beta": 0.005,
        "vae_epochs": 50,
        "vae_batch_size": 128,
        "vae_hidden_size": 64,
        "vae_latent_size": 256,
        "vae_activation": "relu",
        "vae_kernel_size": 3,
        "vae_num_blocks": 5,
        "vae_lr": 0.0003,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "aggregation_method": 'mean',
        "solver_samples": 128, "do_evaluation": True,
        "solver_lr": 0.01,
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-tf-bind-8')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--epsilon', type=float, default=1e-3)
@click.option('--eta_lambda_', type=float, default=1e-3)
@click.option('--rho', type=float, default=0.05)
@click.option('--r', type=float, default=0.05)
def tf_bind_8(local_dir, cpus, gpus, num_parallel, num_samples, lambda_, epsilon, eta_lambda_, rho, r):
    """Evaluate Naive Gradient Ascent on TFBind8-Exact-v0
    """

    # Final Version

    from design_baselines.gradient_ascent_IGNITE import gradient_ascent_IGNITE
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent_IGNITE, config={
        "lambda_": lambda_,
        "eta_lambda_": eta_lambda_,
        "epsilon": epsilon,
        "rho": rho,
        "r": r,
        "logging_dir": "data",
        "task": "TFBind8-Exact-v0",
        "task_kwargs": {"relabel": False},
        "normalize_ys": True,
        "normalize_xs": True,
        "model_noise_std": 0.0,
        "val_size": 200,
        "use_vae": False,
        "vae_beta": 0.01,
        "vae_epochs": 50,
        "vae_batch_size": 128,
        "vae_hidden_size": 64,
        "vae_latent_size": 256,
        "vae_activation": "relu",
        "vae_kernel_size": 3,
        "vae_num_blocks": 3,
        "vae_lr": 0.0003,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "aggregation_method": 'mean',
        "solver_samples": 128, "do_evaluation": True,
        "solver_lr": 0.01,
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-utr')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--epsilon', type=float, default=1e-3)
@click.option('--eta_lambda_', type=float, default=1e-3)
@click.option('--rho', type=float, default=0.05)
@click.option('--r', type=float, default=0.05)
def utr(local_dir, cpus, gpus, num_parallel, num_samples, lambda_, epsilon, eta_lambda_, rho, r):
    """Evaluate Naive Gradient Ascent on UTR-ResNet-v0
    """

    # Final Version

    from design_baselines.gradient_ascent_IGNITE import gradient_ascent_IGNITE
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent_IGNITE, config={
        "lambda_": lambda_,
        "eta_lambda_": eta_lambda_,
        "epsilon": epsilon,
        "rho": rho,
        "r": r,
        "logging_dir": "data",
        "task": "UTR-ResNet-v0",
        "task_kwargs": {"relabel": False},
        "normalize_ys": True,
        "normalize_xs": True,
        "model_noise_std": 0.0,
        "val_size": 200,
        "use_vae": False,
        "vae_beta": 0.01,
        "vae_epochs": 50,
        "vae_batch_size": 128,
        "vae_hidden_size": 64,
        "vae_latent_size": 256,
        "vae_activation": "relu",
        "vae_kernel_size": 3,
        "vae_num_blocks": 5,
        "vae_lr": 0.0003,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "aggregation_method": 'mean',
        "solver_samples": 128, "do_evaluation": True,
        "solver_lr": 0.01,
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-tf_bind_10')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--epsilon', type=float, default=1e-3)
@click.option('--eta_lambda_', type=float, default=1e-3)
@click.option('--rho', type=float, default=0.05)
@click.option('--r', type=float, default=0.05)
def tf_bind_10(local_dir, cpus, gpus, num_parallel, num_samples, lambda_, epsilon, eta_lambda_, rho, r):
    """Evaluate Naive Gradient Ascent on TFBind10-Exact-v0
    """

    # Final Version

    from design_baselines.gradient_ascent_IGNITE import gradient_ascent_IGNITE
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent_IGNITE, config={
        "lambda_": lambda_,
        "eta_lambda_": eta_lambda_,
        "epsilon": epsilon,
        "rho": rho,
        "r": r,
        "logging_dir": "data",
        "task": "TFBind10-Exact-v0",
        "task_kwargs": {"relabel": False, "dataset_kwargs": {"max_samples": 10000}},
        "normalize_ys": True,
        "normalize_xs": True,
        "model_noise_std": 0.0,
        "val_size": 200,
        "use_vae": False,
        "vae_beta": 0.01,
        "vae_epochs": 50,
        "vae_batch_size": 128,
        "vae_hidden_size": 64,
        "vae_latent_size": 256,
        "vae_activation": "relu",
        "vae_kernel_size": 3,
        "vae_num_blocks": 5,
        "vae_lr": 0.0003,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "aggregation_method": 'mean',
        "solver_samples": 128, "do_evaluation": True,
        "solver_lr": 0.01,
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='gradient-ascent-nas')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--epsilon', type=float, default=1e-3)
@click.option('--eta_lambda_', type=float, default=1e-3)
@click.option('--rho', type=float, default=0.05)
@click.option('--r', type=float, default=0.05)
def nas(local_dir, cpus, gpus, num_parallel, num_samples, lambda_, epsilon, eta_lambda_, rho, r):
    """Evaluate Naive Gradient Ascent on CIFARNAS-Exact-v0
    """

    # Final Version

    from design_baselines.gradient_ascent_IGNITE import gradient_ascent_IGNITE
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(gradient_ascent_IGNITE, config={
        "lambda_": lambda_,
        "eta_lambda_": eta_lambda_,
        "epsilon": epsilon,
        "rho": rho,
        "r": r,
        "logging_dir": "data",
        "task": "CIFARNAS-Exact-v0",
        "task_kwargs": {"relabel": False},
        "normalize_ys": True,
        "normalize_xs": True,
        "model_noise_std": 0.0,
        "val_size": 200,
        "use_vae": False,
        "vae_beta": 0.01,
        "vae_epochs": 50,
        "vae_batch_size": 128,
        "vae_hidden_size": 64,
        "vae_latent_size": 256,
        "vae_activation": "relu",
        "vae_kernel_size": 3,
        "vae_num_blocks": 5,
        "vae_lr": 0.0003,
        "batch_size": 128,
        "epochs": 100,
        "activations": [['leaky_relu', 'leaky_relu']],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "aggregation_method": 'mean',
        "solver_samples": 128, "do_evaluation": False,
        "solver_lr": 0.01,
        "solver_steps": 200},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})
