# encoding=utf-8
from fairseq import options
from fairseq_cli.generate import main


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
