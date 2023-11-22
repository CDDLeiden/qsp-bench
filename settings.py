N_REPLICAS = 30  # number of repetitions per experiment
SEED = 42  # seed for random operations

# grids
PREP_GRID = {
    "splits": [
        {
            "split": "qsprpred.data.sampling.splits.RandomSplit",
            "kwargs": {
                "test_size": [0.1, 0.2, 0.3]
            }
        },
        {
            "split": "qsprpred.data.sampling.splits.TimeSplit",
            "kwargs": {
                "year": [2000, 2005, 2010]
            }
        }
    ],
    "datasets": {
        "acc_keys": [
            "P30542",  # A1
            "P29274",  # A2A
            "P29275",  # A2B
            "P0DMS8",  # A3
        ],
        "source": "papyrus",
        "settings": {
            "quality": "high",
            "papyrus_version": "05.6",
            "plus_only": True,
            "activity_types": "all"
        }
    }
}


