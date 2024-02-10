def main():
    import logging

    import os
    from importlib import import_module

    from qsprpred.benchmarks import BenchmarkSettings, BenchmarkRunner
    from qsprpred.logs import logger

    logger.setLevel(logging.INFO)

    if "QSPBENCH_SETTINGS" not in os.environ:
        raise ValueError(
            "Please set the QSPBENCH_SETTINGS environment "
            "variable with an import path to your settings module."
        )
    settings_module = os.environ["QSPBENCH_SETTINGS"]
    os.environ["QSPBENCH_SETTINGS"] = settings_module
    settings_module = import_module(settings_module)
    settings = BenchmarkSettings(
        name=settings_module.NAME,
        n_replicas=settings_module.N_REPLICAS,
        random_seed=settings_module.SEED,
        data_sources=settings_module.DATA_SOURCES,
        descriptors=settings_module.DESCRIPTORS,
        target_props=settings_module.TARGET_PROPS,
        prep_settings=settings_module.DATA_PREPS,
        models=settings_module.MODELS,
        assessors=settings_module.ASSESSORS,
    )
    if settings_module.RESET_MODELS and os.path.exists(settings_module.MODELS_DIR):
        logger.warning("Resetting models directory.")
        import shutil

        shutil.rmtree(settings_module.MODELS_DIR)
    if settings_module.RESET_MODELS and os.path.exists(settings_module.RESULTS_FILE):
        logger.warning("Resetting results file.")
        os.remove(settings_module.RESULTS_FILE)
    runner = BenchmarkRunner(
        settings,
        settings_module.N_PROC,
        settings_module.DATA_DIR,
        settings_module.RESULTS_FILE,
        settings_module.GPUS,
    )
    runner.run(raise_errors=True)
    print("Benchmark finished.")


if __name__ == "__main__":
    main()
