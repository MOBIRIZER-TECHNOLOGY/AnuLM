"""Run python -m anulm COMMAND [options]."""
import argparse
import runpy
import sys
COMMANDS = {'train': 'anulm.training.train', 'sample': 'anulm.inference.sample', 'ask': 'anulm.inference.ask', 'serve': 'anulm.inference.serve', 'api': 'anulm.apps.api', 'export': 'anulm.inference.export_hf', 'finetune': 'anulm.training.finetune', 'quickstart': 'anulm.tools.quickstart'}

def main():
    parser = argparse.ArgumentParser(prog="anulm", description=__doc__)
    parser.add_argument("command", choices=COMMANDS, nargs="?")
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        return 0
    command = sys.argv[1]
    if command not in COMMANDS:
        parser.error("unknown command: " + command)
    sys.argv = ["anulm " + command] + sys.argv[2:]
    runpy.run_module(COMMANDS[command], run_name="__main__")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
