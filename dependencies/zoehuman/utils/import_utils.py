import argparse
import importlib
import os
from inspect import getmembers, ismodule


def get_user_members(target_module_name):
    """Get all members defined by user from the target module.

    Args:
        target_module_name (str):
            Target module split by dot.

    Returns:
        list:
            A list with 2 items.
            Return[0] is the target module in module type.
            Return[1] is a list of (name, value) pairs.
    """
    target_module = importlib.import_module(target_module_name)
    all_member = getmembers(target_module)
    user_members = []
    for member in all_member:
        member_name = member[0]
        # filter builtins
        if 'builtins' in member_name:
            continue
        # filter __*__
        elif member_name.startswith('__') and \
                member_name.endswith('__'):
            continue
        # filter modules(defined somewhere else)
        elif ismodule(getattr(target_module, member_name)):
            continue
        else:
            user_members.append(member)
    return target_module, user_members


def main(args):
    # skip import_utils itself
    if 'import_utils' in args.dst_module_path:
        return 1
    src_module_file_path = os.path.join(args.src_module_path + '.py')
    dst_module_file_path = os.path.join(args.dst_module_path + '.py')
    # if either file does not exist
    if not os.path.exists(src_module_file_path) or \
            not os.path.exists(dst_module_file_path):
        return 2
    src_package_path = args.src_module_path.replace('/', '.')
    dst_package_path = args.dst_module_path.replace('/', '.')
    _, dst_members = get_user_members(dst_package_path)
    _, src_members = get_user_members(src_package_path)
    exist_list = []
    for member in dst_members:
        exist_list.append(member[0])
    missing_list = []
    for member in src_members:
        type_str = str(member[1])
        # function, yes
        if 'function' in type_str and \
                '0x' in type_str:
            pass
        # class definded in package, yes
        elif 'class' in type_str and \
                dst_package_path in type_str:
            pass
        else:
            continue
        if member[0] not in exist_list:
            missing_list.append(member[0])

    if len(missing_list) > 0:
        members_str = ', '.join(missing_list)

        if args.print_info:
            print('Missing import in %s:' % args.dst_module_path)
            print(members_str)

        if args.inplace_fix:
            import_str = 'from %s import %s  # noqa: F401\n' %\
                         (src_package_path, members_str)
            # print("Insert \n%s \ninto \n%s" %
            #       (import_str, dst_module_file_path))
            with open(dst_module_file_path, 'r') as f_read:
                lines = f_read.readlines()
            lines.insert(0, import_str)
            with open(dst_module_file_path, 'w+') as f_wirte:
                f_wirte.writelines(lines)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_module_path',
        type=str,
        help='Path to the source package, split by /',
        default='')
    parser.add_argument(
        '--dst_module_path',
        type=str,
        help='Path to the destination package, split by /',
        default='')
    parser.add_argument(
        '--print_info',
        type=bool,
        help='Whether to print info if import is missing',
        default=True)
    parser.add_argument(
        '--inplace_fix',
        type=bool,
        help='Whether to fix the missing import by ' +
        'editting the destination file',
        default=False)
    args = parser.parse_args()
    main(args)
