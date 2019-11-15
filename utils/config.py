#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Generates user-specific config for current project.
'''

import os
import errno
import argparse
import subprocess
import configparser

project_root = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(project_root)
#project_root = os.path.dirname(project_root)

class Config:

    """
    Stores project config
    """

    def __init__(self):
        # Assign default paths
        config_path = os.path.join(project_root, 'user.conf')

        if not os.path.isfile(config_path):
            config_path = os.path.join(project_root, 'default.conf')

        if not os.path.isfile(config_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    config_path)

        config = configparser.ConfigParser()
        config.read(config_path)
        self.path = config_path
        self.cfg = config

    @property
    def cfg(self):
        return self._cfg

    @cfg.setter
    def cfg(self, cfg):
        paths = cfg['PATHS']
        for i in paths:
            path = os.path.abspath(paths[i])
            if not os.path.isdir(path):
                os.makedirs(path)
            paths[i] = path
        paths['root'] = project_root
        self._cfg = cfg

    def get(self, address):
        if isinstance(address, tuple):
            return self.cfg.get(*address)
        return self.cfg[address]

    def __getitem__(self, path):
        return self.get(path)

def show_list(args):
    c = Config()
    print('Config found at {0!s}'.format(c.path))
    if len(args.address) == 0:
        msg = 'Config contains following sections:\n'
        msg += '\n'.join(map(lambda x: '\t' + x, c.cfg.sections()))
        print(msg)

    elif len(args.address) == 1:
        print('Describing section {0!s}:'.format(*args.address))
        for p in c[args.address[0]]:
            print('\t', p, '=>', c[args.address[0], p])

    else:
        print('List content of {0!s} => {1!s}'.format(*args.address))
        path = c[tuple(args.address)]
        ls = subprocess.run(['ls', '-lh', path],
                            stdout = subprocess.PIPE)
        print(ls.stdout.decode('utf-8'))



def main():
    parser = argparse.ArgumentParser(
        description = 'Returns information on user config',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(help='sub-command help')
    parser_list = subparsers.add_parser('list', help='list content of config')
    parser_list.add_argument('address', type=str, nargs = '*', default = None)
    parser_list.set_defaults(func=show_list)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()


