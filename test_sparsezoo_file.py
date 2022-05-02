import pytest
import numpy as np
import os
import tempfile
import onnx
import yaml
from onnx.helper import make_node, make_graph, make_model, make_tensor_value_info


from sparsezoo_file import File, Directory, SampleOriginals


def _create_sample_file(file_path):
    _, extension = os.path.splitext(file_path)
    if extension == ".npz":
        np.savez(file_path, np.arange(2))
    elif extension == ".onnx":
        node = make_node('MatMul', ['X', 'Y'], ['Z'], name='test_node')
        graph = make_graph([node], 'test_graph',
                           [make_tensor_value_info('X', onnx.TensorProto.FLOAT, (1,2)),
                            make_tensor_value_info('Y', onnx.TensorProto.FLOAT, (2,1))],
                           [make_tensor_value_info('Z', onnx.TensorProto.FLOAT, (1,1))])
        model = make_model(graph)
        onnx.checker.check_model(model)
        onnx.save_model(model, file_path)
    elif extension == ".yaml":
        test_dict = {'test_key': 'test_value'}
        with open(file_path, 'w') as outfile:
            yaml.dump(test_dict, outfile, default_flow_style=False)




@pytest.mark.parametrize(
    "extension, raise_error",
    [
        (".npz", False),
        (".onnx", False),
        (".yaml", False),
        (".json", True),

    ],
    scope = "function"
)
class TestFile:
    """
    Try the File object against different supported/prohibive file types.
    """
    @pytest.fixture()
    def setup(self, extension, raise_error):
        # setup
        _, path = tempfile.mkstemp(suffix=extension)
        _create_sample_file(path)

        yield path, raise_error

        # teardown
        os.remove(path)

    def test_validate(self, setup):
        path, raise_error = setup
        file = File(name = path)
        if not raise_error:
            assert file.validate()
        else:
            with pytest.raises(Exception):
                assert file.validate()

@pytest.mark.parametrize(
    "list_extensions, faulty_files",
    [
        ([".npz", ".onnx", ".yaml"], False, ),
        ([".npz", ".onnx", ".yaml"], True,),

    ],
    scope = "function"
)

class TestDirectory:
    @pytest.fixture()
    def setup(self, list_extensions, faulty_files):
        files = []
        for extension in list_extensions:
            _, path = tempfile.mkstemp(suffix=extension)
            files.append(File(name = path))
            _create_sample_file(path)

        if faulty_files:
            new_path = tempfile.NamedTemporaryFile(suffix=extension, dir=os.path.abspath(os.getcwd()), delete=False)
            files.append(File(name = new_path.name))

        yield files, faulty_files

        for file in files:
            os.remove(file.name)

    def test_directory(self, setup):
        files, raise_error = setup
        if not raise_error:
            directory = Directory(files = files)
            assert directory.dirname == os.path.dirname(files[0].name)
        else:
            with pytest.raises(Exception):
                Directory(files=files)

    #def test_gzip(self, setup):
    #    files, raise_error = setup
    #    if not raise_error:
    #        directory = Directory(files = files)
    #        gzip_dir = directory.gzip()
    #        assert gzip_dir == os.path.dirname(files[0].name) + '.zip'
    #        os.remove(gzip_dir)

@pytest.mark.parametrize(
    "list_extensions, raise_error",
    [
        ([".npz", ".npz", ".npz"], False),
        ([".npz", ".onnx", ".yaml"], True),

    ],
    scope = "function"
)
class TestSampleOriginals:
    @pytest.fixture()
    def setup(self, list_extensions, raise_error):
        files = []
        for extension in list_extensions:
            _, path = tempfile.mkstemp(suffix=extension)
            files.append(File(name = path))
            _create_sample_file(path)

        yield files, raise_error

        for file in files:
            os.remove(file.name)

    def test_validate(self, setup):
        files, raise_error = setup
        sample_originals = SampleOriginals(files=files)
        if not raise_error:
            sample_originals.validate()

        else:
            with pytest.raises(Exception):
                sample_originals.validate()








