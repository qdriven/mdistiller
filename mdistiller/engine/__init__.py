from .trainer import BaseTrainer, CRDTrainer, DOT, CRDDOT, CTDKDTrainer
trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "dot": DOT,
    "crd_dot": CRDDOT,
    "ctdkd": CTDKDTrainer,
}
