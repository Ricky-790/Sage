import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from sage.tests.workflows.refund_agent.agent import run_refund_agent


def main():
    parser = argparse.ArgumentParser(description="Approve or reject a refund request.")
    parser.add_argument("--customer-id", required=True)
    parser.add_argument("--order-id", required=True)
    parser.add_argument(
        "--agent",
        choices=("deterministic", "llm"),
        default="deterministic",
        help="Use the fixed policy or a LangChain agent.",
    )
    args = parser.parse_args()

    result = run_refund_agent(args.customer_id, args.order_id, args.agent)
    print(json.dumps(result, indent=2))

    if result["reason"].startswith("llm_") or result["reason"] == "unknown_agent_mode":
        return 2
    return 0 if result["approved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
