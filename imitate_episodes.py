import argparse

from PACT.runners.eval_runner import main as eval_main
from PACT.runners.train_runner import main as train_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True)
    parser.add_argument("--policy_class", action="store", type=str, help="policy_class, capitalize", required=True)
    parser.add_argument("--task_name", action="store", type=str, help="task_name", required=True)
    parser.add_argument("--batch_size", action="store", type=int, help="batch_size", required=False)
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument("--num_epochs", action="store", type=int, help="num_epochs", required=False)
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)
    parser.add_argument("--resume_ckpt", action="store", type=str, default=None, help="checkpoint path for resume")
    parser.add_argument("--start_epoch", action="store", type=int, default=None, help="override start epoch when resuming")

    parser.add_argument("--kl_weight", action="store", type=int, help="KL Weight", required=False)
    parser.add_argument("--chunk_size", action="store", type=int, help="chunk_size", required=False)
    parser.add_argument("--hidden_dim", action="store", type=int, help="hidden_dim", required=False)
    parser.add_argument("--dim_feedforward", action="store", type=int, help="dim_feedforward", required=False)
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument("--num_workers", action="store", type=int, default=4, help="dataloader workers")
    parser.add_argument("--prefetch_factor", action="store", type=int, default=2, help="dataloader prefetch factor")
    parser.add_argument("--persistent_workers", action="store", type=int, default=1, help="1 to keep workers persistent")
    parser.add_argument("--pin_memory", action="store", type=int, default=1, help="1 to enable pin memory")

    parser.add_argument(
        "--equipment_model",
        action="store",
        type=str,
        default="vx300s_bimanual",
        help="equipment model folder under assets (e.g., vx300s_bimanual)",
    )
    parser.add_argument("--dataset_dir", action="store", type=str, default=None, help="override dataset dir for train")
    parser.add_argument("--num_episodes", action="store", type=int, default=None, help="override num_episodes for train")
    parser.add_argument("--num_rollouts", action="store", type=int, default=50, help="eval only: number of rollout episodes (default 50)")
    args = vars(parser.parse_args())
    if args.get("eval"):
        eval_main(args)
    else:
        train_main(args)
