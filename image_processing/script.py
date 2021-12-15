from argparse import ArgumentParser
from distutils.command.config import config

from netron import __version__

parser = ArgumentParser(description="""Run this script under SUPER USER privilege. Create or remove git repository
under direcotry: {}. Currently only works for Linux. Most operations are only tested under Linux.
Script version : {}.
    """.format("zzl", __version__))

print(parser)
parser.add_argument("-V", "--version", help="Print this script version", action="store_true")
args = parser.parse_args()
if args.version:
	print("Script version is {}".format(__version__))