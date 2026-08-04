"""LSF command runner that exempts shared-FS file_mounts from symlink-wrapping.

The base :class:`~sky.utils.command_runner.LsfCommandRunner` transfers files to
the LSF *login* node. For containerized steps (enroot), the job runs on a
*compute* node inside a container whose HOME and ``/tmp`` are node-local and are
NOT shared with the login node. Only the shared network filesystem roots (e.g.
``/proj``) are bind-mounted *identity* into the container, so a payload written
to such a path on the login node is visible to the job at the identical path.

The shared cloud-agnostic backend (``_execute_file_mounts`` in
``cloud_vm_ray_backend.py``) would otherwise sudo-symlink-wrap every absolute,
non-``~/``/non-``/tmp/`` destination — which both fails on the sudo-less login
node and redirects the payload away from the identity-mounted path. This
subclass exposes the shared-FS roots via :meth:`get_unwrapped_mount_prefixes`
so the backend leaves those destinations un-wrapped; the base login-node rsync
then writes them straight to the shared FS where the container can read them.

All other transfers (down-syncs, ``~``/``/tmp`` targets, runtime setup, job
driver) keep the inherited base-runner behavior.
"""
from typing import List, Optional

from sky.utils import command_runner


class LsfContainerCommandRunner(command_runner.LsfCommandRunner):
    """LsfCommandRunner that keeps shared-FS file_mounts un-wrapped.

    Instances are created by the LSF provisioner
    (:func:`sky.provision.lsf.instance.get_command_runners`), which passes the
    shared network-filesystem roots that are identity bind-mounted into the
    enroot container. Destinations under those roots are written directly on the
    login node and are visible to the containerized job at the same path.
    """

    def __init__(self,
                 *args,
                 shared_fs_roots: Optional[List[str]] = None,
                 **kwargs) -> None:
        """Record the identity-mounted shared-FS roots for wrap-exemption.

        :param args: positional args forwarded to
            :class:`~sky.utils.command_runner.LsfCommandRunner`.
        :param shared_fs_roots: absolute path prefixes that are bind-mounted
            identity into the enroot container (e.g. ``['/proj']``); file_mount
            destinations under these are exempt from the backend's symlink-wrap.
        :param kwargs: keyword args forwarded to the base runner.
        """
        super().__init__(*args, **kwargs)
        self._shared_fs_roots: List[str] = list(shared_fs_roots or [])

    def get_unwrapped_mount_prefixes(self) -> List[str]:
        """Return the shared-FS roots whose file_mounts must not be wrapped.

        Overrides :meth:`sky.utils.command_runner.CommandRunner.get_unwrapped_mount_prefixes`
        (which returns ``[]``) to declare this cluster's identity-mounted roots.

        Consumed by ``_execute_file_mounts`` in the backend: a destination equal
        to, or under, one of these prefixes bypasses ``make_safe_symlink_command``
        and is rsynced straight to the shared filesystem on the login node.

        :returns: a copy of the identity-mounted shared-FS root prefixes.
        """
        return list(self._shared_fs_roots)
