import io
from typing import Union

from mmhuman3d.data.data_structures.smc_reader import SMCReader as SMCReader_mm


class SMCReader(SMCReader_mm):

    def __init__(
        self,
        smc_file: Union[str, io.BytesIO],
        ceph_client=None,
        ceph_sdk_conf_path: str = '~/petreloss.conf',
    ):
        if not isinstance(smc_file, (io.BytesIO, str)):
            raise TypeError(f'{type(smc_file)} unsupported, \
                smc_file should be str or io.BytesIO')

        if isinstance(smc_file, io.BytesIO):
            super().__init__(smc_file)
            return None

        if smc_file.endswith('.7z'):
            import py7zr
        if 's3://' in smc_file and ceph_client is None:
            from petrel_client.client import Client
            ceph_client = Client(ceph_sdk_conf_path)

        if 's3://' not in smc_file and smc_file.endswith('.smc'):
            super().__init__(smc_file)
        elif 's3://' not in smc_file and smc_file.endswith('.7z'):
            zip_file = py7zr.SevenZipFile(smc_file)
            smc_key = next(x for x in zip_file.getnames()
                           if x.endswith('.smc'))
            iob = zip_file.read(smc_key)[smc_key]
            super().__init__(iob)
        elif 's3://' in smc_file and smc_file.endswith('.smc'):
            obj_bytes = ceph_client.get(smc_file, no_cache=True)
            super().__init__(io.BytesIO(obj_bytes))
        elif 's3://' in smc_file and smc_file.endswith('.7z'):
            obj_bytes = ceph_client.get(smc_file, no_cache=True)
            zip_file = py7zr.SevenZipFile(io.BytesIO(obj_bytes))
            smc_key = next(x for x in zip_file.getnames()
                           if x.endswith('.smc'))
            iob = zip_file.read(smc_key)[smc_key]
            super().__init__(iob)
        else:
            raise ValueError(f'cannot init SMCReader with smc_file={smc_file}')
