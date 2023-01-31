"""
Functions that are common to all rbrain modules.
"""
import nibabel as nib
import os
from core import rbrain
import subprocess as sp

DEF_CONFIG_FILE = os.path.join(rbrain.__path__[0], "rbrain.cfg")


def run_cmd(cmd_str, modules=[]):
    """Runs a command in a bash shell, loading modules as necessary in HPC.
    Assumes that bash is in the path, as well as any executables not listed. Runs in "./" by default.

    :param cmd_str: The command to be run
    :type cmd_str: str
    :param modules: List of modules to imported (via lmod on HPC). Provide as a list of strings
    :type modules: list, optional.

    :return: STDOUT, useful for example if you are collecting the output for analysis e.g. with fslstats
    :rtype: str
    """

    # Avoids a Nonetype error
    if modules is None:
        modules = []

    # Generate the string to load lmod modules
    module_str = ""
    for m in modules:
        module_str = module_str + "module load " + m + ' && '

    # Run the command with subprocess, including modules string if any, and return STDOUT
    output = sp.run(['bash', '-c', module_str + cmd_str],
                    universal_newlines=True,
                    stdout=sp.PIPE)

    return output.stdout.splitlines()


def read_nifti_file(filepath):
    """Reads a NIFTI file into something numpy can process.

    :param filepath: The full path and filename of the scan being read
    :type filepath: str

    :return: A numpy ndarray of sort (see nibabel docs for detail)
    :rtype: numpy.ndarray
    """
    return nib.load(filepath).get_fdata()


def is_nifti(filename, gz_ok=True, hidden_ok=False):
    """
    Reads in a filename and determines if it is a valid NIFTI file based on extension (i.e. whether it ends in
    .nii or optionally .nii.gz).

    This seems really obvious but having this method makes everything SO much cleaner.

    Args:
        filename (str): File name with/without full path.
        gz_ok (bool): Are .gz files OK? (default=True)
        hidden_ok (bool): Are dot-file (hidden files, i.e. begin with '.') OK? (default=False)

    Returns:
        bool: Whether is the file named like a nifti file.
    """

    if filename.startswith('.') and not hidden_ok:
        return False

    if filename.endswith('.gz') and not gz_ok:
        return False

    return filename.endswith('.nii.gz') or filename.endswith('.nii')
