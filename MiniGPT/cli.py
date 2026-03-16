"""
cli.py
======
Command-line interface for miniGPT.

Run with no arguments to execute the built-in quick demo::

    python cli.py

Train from scratch::

    python cli.py --train wiki_dataset.txt --epochs 100 --save gpt_weights.json

Resume training from a checkpoint (continues where you left off)::

    python cli.py --train wiki_dataset.txt --resume gpt_weights_v4.json --epochs 200 --lr 0.0003 --save gpt_weights_v5.json

Generate text::

    python cli.py --load gpt_weights.json --prompt "Democracy is" --length 300

All flags::

    python cli.py --help
"""

from __future__ import annotations
import argparse
from model import MiniGPT
from data import simplify_text

_DEMO_TEXT = (
    "democracy is the foundation of freedom. "
    "civil rights protect every person equally. "
    "the environment needs protection from pollution. "
    "history teaches us lessons about justice and equality. "
    "science reveals the truth about our natural world. "
) * 30


def _build_parser() -> argparse.ArgumentParser:
    """
    Build and return the argument parser for the miniGPT CLI.

    :return: Configured :class:`argparse.ArgumentParser`.
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        prog="miniGPT",
        description=(
            "miniGPT — character-level language model built on Neural_Network.py\n\n"
            "EXAMPLES\n"
            "--------\n"
            "  Quick demo:\n"
            "    python cli.py\n\n"
            "  Train from scratch:\n"
            "    python cli.py --train wiki_dataset.txt --epochs 100 --save gpt_weights.json\n\n"
            "  Resume training from checkpoint:\n"
            "    python cli.py --train wiki_dataset.txt --resume gpt_weights_v4.json --epochs 200 --lr 0.0003 --save gpt_weights_v5.json\n\n"
            "  Generate text:\n"
            "    python cli.py --load gpt_weights.json --prompt \"Democracy is\" --length 300\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    p.add_argument("--train",  metavar="TEXT_FILE",   help="Path to .txt file to train on.")
    p.add_argument("--resume", metavar="WEIGHTS_FILE",
                   help="Load existing weights and CONTINUE training from them.\n"
                        "Use together with --train.\n"
                        "E.g. --resume gpt_weights_v4.json")
    p.add_argument("--load",   metavar="WEIGHTS_FILE",help="Load saved weights for generation only.")
    p.add_argument("--save",   metavar="WEIGHTS_FILE", default="gpt_weights.json",
                   help="Where to save after training.\n(default: gpt_weights.json)")

    # ── Training ──────────────────────────────────────────────────────────────
    tg = p.add_argument_group("Training options")
    tg.add_argument("--epochs",     type=int,   default=100,
                    help="Training epochs. (default: 100)")
    tg.add_argument("--samples",    type=int,   default=20_000,
                    help="Max training samples. (default: 20000)")
    tg.add_argument("--context",    type=int,   default=12,
                    help="Context window size. (default: 12)")
    tg.add_argument("--hidden",     type=int,   nargs="+", default=[512, 256, 128],
                    help="Hidden layer sizes. E.g. --hidden 512 256 128\n(default: 512 256 128)")
    tg.add_argument("--activation", type=str,   default="relu",
                    help="relu | tanh | sigmoid | leaky_relu (default: relu)")
    tg.add_argument("--lr",         type=float, default=0.001,
                    help="Learning rate. (default: 0.001)")
    tg.add_argument("--embed_dim",  type=int,   default=64,
                    help="Embedding dimensions. 64/128/256 (default: 64)")
    tg.add_argument("--batch_size", type=int,   default=1024,
                    help="Samples per gradient step. (default: 1024)")
    tg.add_argument("--num_blocks", type=int,   default=2,
                    help="Number of transformer blocks. (default: 2)")
    tg.add_argument("--num_heads",  type=int,   default=4,
                    help="Attention heads. embed_dim must be divisible by this. (default: 4)")
    tg.add_argument("--dropout",    type=float, default=0.0,
                    help="Dropout rate 0.0-0.5. 0=disabled. (default: 0.0)")
    tg.add_argument("--no_weight_tying", action="store_true",
                    help="Disable weight tying (separate Wout). Default: tying ON.")
    tg.add_argument("--grad_clip",  type=float, default=1.0,
                    help="Gradient clipping norm. 0=disabled. (default: 1.0)")
    tg.add_argument("--max_chars",  type=int,   default=500_000,
                    help="Max chars to read from file. (default: 500000)")
    tg.add_argument("--log_every",  type=int,   default=1,
                    help="Print loss every N epochs. 0=silent. (default: 1)")
    tg.add_argument("--simple_vocab", action="store_true",
                    help="Strip text to lowercase a-z + space + basic punctuation (~36 chars). "
                         "Makes learning much easier for small models.")
    tg.add_argument("--save_every",  type=int, default=0,
                    help="Save a checkpoint every N epochs. 0 = disabled. (default: 0)")
    tg.add_argument("--force_lr",   type=float, default=None,
                    help="Force learning rate to this value on resume, even if Adam state\n"
                         "is present. Useful to restart from a higher lr after plateauing.\n"
                         "WARNING: pair with --reset_adam if boosting lr by more than ~2x.\n"
                         "E.g. --force_lr 0.0003")
    tg.add_argument("--reset_adam", action="store_true",
                    help="Wipe Adam momentum state on resume and start fresh.\n"
                         "Use with --force_lr when boosting lr significantly.\n"
                         "Without this, old momentum built at low lr causes loss spikes.")

    # ── Generation ────────────────────────────────────────────────────────────
    gg = p.add_argument_group("Generation options")
    gg.add_argument("--prompt",      type=str,   default="",
                    help="Seed text. E.g. --prompt \"Science is\"")
    gg.add_argument("--length",      type=int,   default=300,
                    help="Characters to generate. (default: 300)")
    gg.add_argument("--temperature", type=float, default=0.6,
                    help="< 1.0 focused, > 1.0 creative. (default: 0.6)")

    return p


def main() -> None:
    """
    Entry point for the miniGPT CLI.

    **Modes:**

    ``--train`` only
        Train a brand new model from scratch.

    ``--train`` + ``--resume``
        Load existing weights and continue training from them.
        Architecture (hidden layers, context size etc.) is loaded from
        the checkpoint — you only need to specify ``--lr``, ``--epochs``,
        ``--samples``, and ``--save``.

    ``--load``
        Load a saved model and generate text.

    No flags
        Run the built-in demo.

    **Resume example:**

    .. code-block:: bash

        python cli.py \\
          --train wiki_dataset.txt \\
          --resume gpt_weights_v4.json \\
          --epochs 200 \\
          --lr 0.0003 \\
          --save gpt_weights_v5.json
    """
    parser = _build_parser()
    args   = parser.parse_args()

    # ── Resume mode: load checkpoint then keep training ───────────────────────
    if args.train and args.resume:
        print(f"Resuming from '{args.resume}'...")
        model = MiniGPT.load(args.resume)

        # -- Learning rate override logic ------------------------------------
        # --force_lr: hard override regardless of Adam state (intentional restart)
        # --lr:       only applied if no Adam state (safe default behaviour)
        if args.force_lr is not None:
            old_lr = model.nn.learning_rate
            model.nn.learning_rate = args.force_lr
            print(f"Learning rate force-overridden: {old_lr:.6f} -> {args.force_lr:.6f}")
        elif args.lr != 0.001 and not model.nn._adam_init:
            model.nn.learning_rate = args.lr
            print(f"Learning rate overridden to {args.lr}")
        elif args.lr != 0.001 and model.nn._adam_init:
            print(f"Adam state restored -- keeping saved lr={model.nn.learning_rate:.6f}, ignoring --lr {args.lr}")

        # --reset_adam: wipe momentum state so old gradients don't cause spikes.
        # Critical when boosting lr by more than ~2x -- without this, momentum
        # built at the old (low) lr gets applied at the new (high) lr, making
        # the first few steps far too large and spiking the loss badly.
        if args.reset_adam and model.nn._adam_init:
            model.nn._adam_init = False  # forces _init_adam() on next train()
            print(f"Adam state reset -- fresh momentum at lr={model.nn.learning_rate:.6f}")

        model.train(
            args.train,
            epochs       = args.epochs,
            max_samples  = args.samples,
            max_chars    = args.max_chars,
            simple_vocab = args.simple_vocab,
            save_every   = args.save_every,
            save_path    = args.save,
            log_every    = args.log_every,
        )
        model.save(args.save)
        print("\n-- Sample generation --------------------------------------------------")
        print(model.generate(prompt=args.prompt, length=args.length, temperature=args.temperature))

    # ---- Train from scratch -------------------------------------------------
    elif args.train:
        model = MiniGPT(
            context_size  = args.context,
            hidden_layers = args.hidden,
            activation    = args.activation,
            learning_rate = args.lr,
            embed_dim     = args.embed_dim,
            batch_size    = args.batch_size,
            num_blocks    = args.num_blocks,
            num_heads     = args.num_heads,
            dropout       = args.dropout,
            weight_tying  = not args.no_weight_tying,
            grad_clip     = args.grad_clip,
        )
        model.train(
            args.train,
            epochs       = args.epochs,
            max_samples  = args.samples,
            max_chars    = args.max_chars,
            simple_vocab = args.simple_vocab,
            save_every   = args.save_every,
            save_path    = args.save,
            log_every    = args.log_every,
        )
        model.save(args.save)
        print("\n── Sample generation ──────────────────────────────────────────")
        print(model.generate(prompt=args.prompt, length=args.length, temperature=args.temperature))

    # ── Generate only ─────────────────────────────────────────────────────────
    elif args.load:
        model = MiniGPT.load(args.load)
        print(f"\nGenerating {args.length} chars  (temperature={args.temperature})...\n")
        print("── Output ─────────────────────────────────────────────────────")
        print(model.generate(prompt=args.prompt, length=args.length, temperature=args.temperature))
        print("───────────────────────────────────────────────────────────────")

    # ── Demo ──────────────────────────────────────────────────────────────────
    else:
        parser.print_help()
        print("\n── Quick demo (no dataset needed) ─────────────────────────────")
        print("Training on a built-in string to verify everything works...\n")
        model = MiniGPT(context_size=6, hidden_layers=[64, 32], learning_rate=0.001)
        model.train(_DEMO_TEXT, epochs=10, max_samples=2_000)
        print("\n── Generated text ─────────────────────────────────────────────")
        print(model.generate(prompt="demo", length=150, temperature=0.7))


if __name__ == "__main__":
    main()